from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

POLICY_DIR = Path("data/policies")
POLICY_EVAL_FILE = Path("data/raw/policy_eval.csv")
SCENARIOS_FILE = Path("data/raw/agent_scenarios.csv")
DATABASE = Path("data/instamart.db")
OUTPUT_DIR = Path("outputs")
LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"
AGENT_RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"
STATE_DIR = Path("state")
CHUNK_WORDS = 40
CHUNK_OVERLAP = 10

POLICY_SYSTEM_PROMPT = (
    "You are the POLICY ASSISTANT. Answer only from the supplied policy context. "
    "If the context does not contain the answer, say exactly: "
    "I could not find the information in the policy documents."
)

AGENT_SYSTEM_PROMPT = (
    "You are an AGENT for refund support. Use the supplied tools to look up the order, "
    "retrieve the refund policy, calculate the permitted refund, and issue or escalate it. "
    "Only completed orders qualify; over 45 minutes is a full refund, otherwise zero. "
    "Refunds above INR 1000 must be escalated to a human."
)


class SimulatedCrash(RuntimeError):
    """Raised by the resumability demonstration after a saved tool checkpoint."""


@dataclass
class PolicyRetriever:
    chunks: list[dict[str, Any]]
    vectorizer: TfidfVectorizer
    matrix: Any

    @classmethod
    def from_directory(
        cls, policy_dir: Path = POLICY_DIR, chunk_words: int = CHUNK_WORDS,
        overlap: int = CHUNK_OVERLAP,
    ) -> "PolicyRetriever":
        if chunk_words <= 0 or overlap < 0 or overlap >= chunk_words:
            raise ValueError("Require chunk_words > overlap >= 0")
        chunks: list[dict[str, Any]] = []
        stride = chunk_words - overlap
        for path in sorted(policy_dir.glob("*.txt")):
            words = path.read_text(encoding="utf-8").split()
            start = 0
            while start < len(words):
                end = min(start + chunk_words, len(words))
                chunks.append({
                    "source": path.name,
                    "text": " ".join(words[start:end]),
                })
                if end >= len(words):
                    break
                start += stride
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
        return cls(chunks, vectorizer, matrix)

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(scores[i]),
                "text": self.chunks[i]["text"],
            }
            for i in ranked_indices
        ]


_DEFAULT_RETRIEVER: PolicyRetriever | None = None


def get_retriever() -> PolicyRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = PolicyRetriever.from_directory()
    return _DEFAULT_RETRIEVER


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    return get_retriever().retrieve(query, k=k)


def answer(question: str, retriever: PolicyRetriever | None = None) -> dict[str, Any]:
    active_retriever = retriever or get_retriever()
    results = active_retriever.retrieve(question, k=3)
    context = "\n".join(
        f"[{chunk['source']}]\n{chunk['text']}" for chunk in results
    )
    user_content = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    response = mock_llm.chat([
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])
    return {
        "answer": response.choices[0].message.content or "",
        "sources": [chunk["source"] for chunk in results],
    }


def evaluate_policy(retriever: PolicyRetriever) -> dict[str, Any]:
    evaluation = pd.read_csv(POLICY_EVAL_FILE, keep_default_na=False)
    answerable = evaluation.loc[evaluation["expected_doc"].ne("NONE")]
    unanswerable = evaluation.loc[evaluation["expected_doc"].eq("NONE")]
    retrieval_hits = 0
    fact_hits = 0
    fallback_hits = 0
    failures: list[tuple[str, str, str]] = []

    for row in evaluation.itertuples(index=False):
        retrieved = retriever.retrieve(row.question, k=3)
        result = answer(row.question, retriever=retriever)
        if row.expected_doc != "NONE":
            hit = row.expected_doc in [item["source"] for item in retrieved]
            fact_hit = row.expected_fact in result["answer"]
            retrieval_hits += int(hit)
            fact_hits += int(fact_hit)
            if not fact_hit:
                failures.append((row.question, row.expected_fact, result["answer"]))
        else:
            fallback_hits += int(result["answer"].startswith("I could not find"))
            if not result["answer"].startswith("I could not find"):
                failures.append((row.question, "I could not find", result["answer"]))

    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "retrieval_hits": retrieval_hits,
        "fact_hits": fact_hits,
        "fallback_hits": fallback_hits,
        "failures": failures,
    }


def get_order(order_id: str) -> dict[str, Any]:
    with sqlite3.connect(DATABASE) as connection:
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]) if row[2] is not None else None,
        "delivery_minutes": float(row[3]) if row[3] is not None else None,
    }


def search_policy(query: str) -> dict[str, Any]:
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount: float, delivery_minutes: float) -> dict[str, float]:
    amount = round(float(order_amount), 2)
    if delivery_minutes is None:
        return {"refund_inr": 0.0}
    delivery = float(delivery_minutes)
    return {"refund_inr": amount if delivery > 45 else 0.0}


def _record_refund(tool_name: str, order_id: str, refund_inr: float) -> dict[str, float]:
    amount = round(float(refund_inr), 2)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": amount,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER_FILE.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"refund_inr": amount}


def issue_refund(order_id: str, refund_inr: float) -> dict[str, float]:
    order = get_order(order_id)
    amount = round(float(refund_inr), 2)
    if "error" in order:
        return order
    if order["status"] != "completed":
        return {"error": "only completed orders can be refunded"}
    allowed = calculate_refund(order["order_amount"], order["delivery_minutes"])["refund_inr"]
    if amount != allowed:
        return {"error": "refund amount does not match the late-delivery rule"}
    if amount <= 0 or amount > 1000:
        return {"error": "automatic refunds must be greater than zero and at most INR 1000"}
    return _record_refund("issue_refund", order_id, amount)


def escalate_to_human(order_id: str, refund_inr: float) -> dict[str, float]:
    order = get_order(order_id)
    amount = round(float(refund_inr), 2)
    if "error" in order:
        return order
    if order["status"] != "completed":
        return {"error": "only completed orders can be refunded"}
    allowed = calculate_refund(order["order_amount"], order["delivery_minutes"])["refund_inr"]
    if amount != allowed:
        return {"error": "refund amount does not match the late-delivery rule"}
    if amount <= 1000:
        return {"error": "refunds above INR 1000 are required for human escalation"}
    return _record_refund("escalate_to_human", order_id, amount)


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order and return its status, amount and delivery time.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "name": "search_policy",
        "description": "Retrieve the three policy chunks most relevant to a query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the full late-delivery refund or zero for an eligible completed order.",
        "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, "required": ["order_amount", "delivery_minutes"]},
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund of INR 1000 or less.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]},
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund above INR 1000 for human approval.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]},
    },
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def _tool_call_dict(call: Any) -> dict[str, Any]:
    name = call.name if hasattr(call, "name") else call["name"]
    call_id = call.id if hasattr(call, "id") else call["id"]
    arguments = call.arguments if hasattr(call, "arguments") else call["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return {"id": call_id, "name": name, "arguments": arguments}


def _checkpoint_path(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, underscores and hyphens")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"{run_id}.json"


def _save_checkpoint(state: dict[str, Any]) -> None:
    path = _checkpoint_path(state["run_id"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fallback(state: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    completed = ", ".join(state["tools_executed"]) if state["tools_executed"] else "none"
    state["status"] = status
    state["final_answer"] = f"{status}: {reason} Human handoff required; completed tools: {completed}."
    _save_checkpoint(state)
    return {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": status,
        "final_answer": state["final_answer"],
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "tools_executed": state["tools_executed"],
    }


def _continue_agent(
    state: dict[str, Any], crash_after_tool_executions: int | None = None
) -> dict[str, Any]:
    while state["llm_calls"] < state["max_iterations"]:
        elapsed = time.monotonic() - state["started_monotonic"]
        if elapsed >= state["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT", "Time limit reached before the next model call.")

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += int(response.usage.total_tokens)
        message = response.choices[0].message
        calls = [_tool_call_dict(call) for call in (message.tool_calls or [])]

        if state["tokens_used"] > state["token_limit"]:
            return _fallback(
                state, "FALLBACK_TOKEN_LIMIT",
                "Token limit exceeded; the requested tool was not executed.",
            )

        if not calls:
            final_answer = message.content or ""
            status = final_answer.split(":", 1)[0].strip() if ":" in final_answer else "ANSWER"
            state["status"] = status
            state["final_answer"] = final_answer
            _save_checkpoint(state)
            return {
                "run_id": state["run_id"],
                "scenario_id": state["scenario_id"],
                "status": status,
                "final_answer": final_answer,
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"],
                "tools_executed": state["tools_executed"],
            }

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": calls,
        })
        for call in calls:
            tool_name = call["name"]
            arguments = call["arguments"]
            tool_function = TOOL_MAP.get(tool_name)
            if tool_function is None:
                observation = {"error": f"unknown tool: {tool_name}"}
            else:
                try:
                    observation = tool_function(**arguments)
                except Exception as error:
                    observation = {"error": f"{type(error).__name__}: {error}"}
            observation_json = json.dumps(observation, ensure_ascii=False, sort_keys=True)
            state["messages"].append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": tool_name,
                "content": observation_json,
            })
            state["steps"].append({
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation,
            })
            state["tools_executed"].append(tool_name)
            state["step_number"] += 1
            _save_checkpoint(state)
            if crash_after_tool_executions is not None and len(state["tools_executed"]) >= crash_after_tool_executions:
                raise SimulatedCrash(
                    f"Simulated crash after tool {state['step_number']} checkpoint was saved"
                )

    return _fallback(
        state, "FALLBACK_MAX_ITERATIONS",
        f"Reached the limit of {state['max_iterations']} model calls without a final answer.",
    )


def run_agent(
    user_message: str,
    scenario_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    run_id: str | None = None,
    crash_after_tool_executions: int | None = None,
) -> dict[str, Any]:
    run_id = run_id or f"run_{uuid.uuid4().hex}"
    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "user_message": user_message,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
        "started_monotonic": time.monotonic(),
    }
    _checkpoint_path(run_id)
    return _continue_agent(state, crash_after_tool_executions=crash_after_tool_executions)


def resume_agent(run_id: str) -> dict[str, Any]:
    path = _checkpoint_path(run_id)
    state = json.loads(path.read_text(encoding="utf-8"))
    return _continue_agent(state)


def _refund_amount_from_result(result: dict[str, Any]) -> float:
    for step in result["steps"]:
        observation = step["observation"]
        if step["tool_name"] in {"calculate_refund", "issue_refund", "escalate_to_human"}:
            if "refund_inr" in observation:
                return float(observation["refund_inr"])
    return 0.0


def _append_json_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if LEDGER_FILE.exists():
        LEDGER_FILE.write_text("", encoding="utf-8")
    if AGENT_RUNS_FILE.exists():
        AGENT_RUNS_FILE.write_text("", encoding="utf-8")

    global _DEFAULT_RETRIEVER
    _DEFAULT_RETRIEVER = PolicyRetriever.from_directory()
    chunk_counts = pd.Series([chunk["source"] for chunk in _DEFAULT_RETRIEVER.chunks]).value_counts()
    print(f"Policy chunks: {len(_DEFAULT_RETRIEVER.chunks)}")
    print("Chunks by source:")
    print(chunk_counts.sort_index().to_string())
    assert len(_DEFAULT_RETRIEVER.chunks) == 15
    assert chunk_counts.to_dict() == {
        "refund_policy.txt": 4,
        "delivery_policy.txt": 3,
        "escalation_policy.txt": 3,
        "payment_policy.txt": 3,
        "privacy_policy.txt": 2,
    }
    top_sources = [item["source"] for item in retrieve("refund policy late delivery", k=3)]
    print(f"Top sources for 'refund policy late delivery': {top_sources}")
    assert top_sources == ["refund_policy.txt"] * 3

    mock_llm.reset()
    overlap_eval = evaluate_policy(_DEFAULT_RETRIEVER)
    zero_overlap_retriever = PolicyRetriever.from_directory(overlap=0)
    mock_llm.reset()
    no_overlap_eval = evaluate_policy(zero_overlap_retriever)
    assert (overlap_eval["retrieval_hits"], overlap_eval["fact_hits"], overlap_eval["fallback_hits"]) == (12, 12, 2)
    assert (no_overlap_eval["retrieval_hits"], no_overlap_eval["fact_hits"], no_overlap_eval["fallback_hits"]) == (12, 10, 2)
    assert any(
        question == "What delivery time makes a completed order eligible for a full refund?"
        for question, _, _ in no_overlap_eval["failures"]
    )
    print("\nPolicy evaluation (overlap 10):")
    print(f"Answerable expected source in top 3: {overlap_eval['retrieval_hits']}/{overlap_eval['answerable']}")
    print(f"Answerable expected fact in answer: {overlap_eval['fact_hits']}/{overlap_eval['answerable']}")
    print(f"Unanswerable fallback answers: {overlap_eval['fallback_hits']}/{overlap_eval['unanswerable']}")
    print("Policy evaluation (overlap 0):")
    print(f"Answerable expected source in top 3: {no_overlap_eval['retrieval_hits']}/{no_overlap_eval['answerable']}")
    print(f"Answerable expected fact in answer: {no_overlap_eval['fact_hits']}/{no_overlap_eval['answerable']}")
    print(f"Unanswerable fallback answers: {no_overlap_eval['fallback_hits']}/{no_overlap_eval['unanswerable']}")
    print(f"Zero-overlap answer failures: {no_overlap_eval['failures']}")

    scenarios = pd.read_csv(SCENARIOS_FILE, keep_default_na=False)
    scenario_results = []
    for scenario in scenarios.itertuples(index=False):
        mock_llm.reset()
        result = run_agent(
            scenario.message,
            scenario.scenario_id,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
            run_id=f"batch_{scenario.scenario_id}",
        )
        amount = _refund_amount_from_result(result)
        expected_amount = float(scenario.expected_refund_inr)
        if scenario.expected_outcome == "FALLBACK":
            assert result["status"] == "FALLBACK_MAX_ITERATIONS"
        else:
            assert result["status"] == scenario.expected_outcome
        assert abs(amount - expected_amount) <= 0.01
        assert result["tokens_used"] <= 6000
        if scenario.scenario_id == "S1":
            assert result["llm_calls"] == 5
        if scenario.scenario_id == "S7":
            assert result["status"] == "FALLBACK_MAX_ITERATIONS"
            assert result["llm_calls"] == 6
        record = {
            "scenario_id": scenario.scenario_id,
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"],
        }
        _append_json_line(AGENT_RUNS_FILE, record)
        scenario_results.append({
            "scenario_id": scenario.scenario_id,
            "status": result["status"],
            "refund_inr": amount,
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
        })

    # Limit demonstrations requested in task 7.
    s1_message = scenarios.loc[scenarios["scenario_id"].eq("S1"), "message"].iloc[0]
    mock_llm.reset()
    token_demo = run_agent(
        s1_message, "S1", token_limit=200, run_id="demo_s1_token_limit"
    )
    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
    assert token_demo["llm_calls"] == 2
    mock_llm.reset()
    timeout_demo = run_agent(
        s1_message, "S1", time_limit_seconds=0, run_id="demo_s1_timeout"
    )
    assert timeout_demo["status"] == "FALLBACK_TIMEOUT" and timeout_demo["llm_calls"] == 0

    # Crash only after tool 2's observation has been checkpointed, then resume.
    resume_run_id = "demo_s1_resume"
    try:
        mock_llm.reset()
        run_agent(
            s1_message,
            "S1",
            run_id=resume_run_id,
            crash_after_tool_executions=2,
        )
    except SimulatedCrash:
        crash_checkpoint = json.loads(_checkpoint_path(resume_run_id).read_text(encoding="utf-8"))
    else:
        raise AssertionError("The crash demonstration did not raise SimulatedCrash")
    mock_llm.reset()
    resumed = resume_agent(resume_run_id)
    expected_tools = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert crash_checkpoint["tools_executed"] == expected_tools[:2]
    assert resumed["tools_executed"] == expected_tools
    assert resumed["status"] == "REFUND"
    assert abs(_refund_amount_from_result(resumed) - 128.25) <= 0.01

    print("\nAgent scenario results:")
    results_df = pd.DataFrame(scenario_results)
    print(results_df.to_string(index=False, formatters={"refund_inr": "INR {:,.2f}".format}))
    print(f"\nS1 token-limit demo: {token_demo['status']}, calls={token_demo['llm_calls']}, tokens={token_demo['tokens_used']}")
    print(f"S1 timeout demo: {timeout_demo['status']}, calls={timeout_demo['llm_calls']}, tokens={timeout_demo['tokens_used']}")
    print(f"Crash/resume tools: {expected_tools}; final={resumed['status']} INR {_refund_amount_from_result(resumed):.2f}")


if __name__ == "__main__":
    main()
