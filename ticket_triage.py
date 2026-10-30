from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import mock_llm

TICKETS_FILE = Path("data/raw/tickets.csv")
SEGMENTS_FILE = Path("outputs/customer_segments.csv")
OUTPUT_DIR = Path("outputs")
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

REDACTION_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = (
    'You are a support ticket triage assistant. CLASSIFY the user message into exactly one '
    'of these allowed categories: billing, delivery, technical, fraud. Reply with JSON only '
    'in this form: {"category": "...", "reason": "..."}.'
)

FEW_SHOT_PROMPT = (
    'You are a support ticket triage assistant. CLASSIFY the user message into exactly one '
    'of these allowed categories: billing, delivery, technical, fraud. Reply with JSON only '
    'in this form: {"category": "...", "reason": "..."}.\n'
    'EXAMPLES:\n'
    'Example 1 (fraud): I see an unauthorized card transaction. '
    '{"category": "fraud", "reason": "unauthorized transaction"}\n'
    'Example 2 (delivery): My package arrived late. '
    '{"category": "delivery", "reason": "late arrival"}'
)


@dataclass
class RunResult:
    rows: pd.DataFrame
    total_tokens: int
    transient_errors: int
    retries: int
    permanent_errors: int


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Apply the required redaction expressions in the required order."""
    redacted = text or ""
    counts: dict[str, int] = {}
    for label, pattern, placeholder in REDACTION_PATTERNS:
        redacted, count = pattern.subn(placeholder, redacted)
        counts[label] = count
    return redacted, counts


def redact_tickets(tickets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    redacted_rows = []
    totals = {label: 0 for label, _, _ in REDACTION_PATTERNS}
    for text in tickets["text"]:
        redacted, counts = redact_text(str(text))
        redacted_rows.append(redacted)
        for label, count in counts.items():
            totals[label] += count
    prepared = tickets.copy()
    prepared["redacted_text"] = redacted_rows
    return prepared, totals


def assert_redacted(texts: pd.Series) -> int:
    remaining = 0
    for text in texts:
        for _, pattern, _ in REDACTION_PATTERNS:
            remaining += len(pattern.findall(text))
    assert remaining == 0, f"Found {remaining} sensitive pattern matches after redaction"
    return remaining


def validate_reply(content: Any) -> tuple[str, str]:
    """Return (status, category), with no guessing for invalid outputs."""
    if not isinstance(content, str):
        return "flagged_invalid_output", ""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", content.lower())
        hits = [word for word in words if word in ALLOWED_CATEGORIES]
        if len(hits) == 1:
            return "classified_keyword", hits[0]
        return "flagged_invalid_output", ""

    # A well-formed JSON answer with an out-of-list category is invalid; do not
    # fall back to keyword matching its reason or other JSON content.
    category = parsed.get("category") if isinstance(parsed, dict) else None
    if isinstance(category, str) and category in ALLOWED_CATEGORIES:
        return "classified_json", category
    return "flagged_invalid_output", ""


def route_ticket(
    status: str, category: str, segment_name: str, is_premium: bool
) -> tuple[str, str]:
    if status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def run_pipeline(
    tickets: pd.DataFrame,
    system_prompt: str,
    token_budget: int | None = None,
) -> RunResult:
    total_tokens = 0
    transient_errors = 0
    retries = 0
    rows = []

    # Iteration is directly over the input frame, preserving file order.
    for ticket in tickets.itertuples(index=False):
        text = ticket.redacted_text
        estimated = mock_llm.est_tokens(text)
        tokens_used = 0
        category = ""

        # Input guardrails run before budget evaluation and before every call.
        if not text.strip():
            status = "skipped_empty"
        elif estimated > MAX_INPUT_TOKENS:
            status = "flagged_long"
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                # Deliberately no wrapper, ID, category or other metadata.
                {"role": "user", "content": text},
            ]
            response = None
            for retry_number in range(MAX_RETRIES + 1):
                if token_budget is not None and total_tokens >= token_budget:
                    status = "budget_exhausted"
                    break
                try:
                    # Privacy assertion immediately before sending the payload.
                    assert text.strip() and estimated <= MAX_INPUT_TOKENS
                    assert messages[-1]["content"] == text
                    assert_redacted(pd.Series([message["content"] for message in messages]))
                    response = mock_llm.chat(messages)
                    tokens_used = int(response.usage.total_tokens)
                    total_tokens += tokens_used
                    break
                except mock_llm.TransientLLMError:
                    transient_errors += 1
                    if retry_number >= MAX_RETRIES:
                        raise RuntimeError(f"Retries exhausted for ticket {ticket.ticket_id}")
                    wait = BASE_DELAY * (2 ** retry_number) * random.uniform(0.9, 1.15)
                    minimum_wait = BASE_DELAY * (2 ** retry_number) * 0.9
                    maximum_wait = BASE_DELAY * (2 ** retry_number) * 1.15
                    assert minimum_wait <= wait <= maximum_wait
                    retries += 1
                    print(
                        f"retry ticket_id={ticket.ticket_id} retry_number={retry_number} "
                        f"wait_seconds={wait:.4f}"
                    )
                    time.sleep(wait)
            else:
                status = "flagged_invalid_output"

            if response is not None:
                status, category = validate_reply(response.choices[0].message.content)

        rows.append({
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "status": status,
            "category": category,
            "tokens_used": tokens_used,
            "true_category": ticket.true_category,
        })

    result = pd.DataFrame(rows)
    return RunResult(result, total_tokens, transient_errors, retries, 0)


def start_run(
    tickets: pd.DataFrame,
    prompt: str,
    token_budget: int | None = None,
) -> RunResult:
    random.seed(42)
    # Reset is immediately before each independent full pipeline run.
    mock_llm.reset()
    return run_pipeline(tickets, prompt, token_budget=token_budget)


def get_metrics(result: RunResult) -> dict[str, Any]:
    classified = result.rows["status"].isin({"classified_json", "classified_keyword"})
    df = result.rows.loc[classified]
    accuracy = (df["category"] == df["true_category"]).mean() if len(df) else 0.0
    status_counts = result.rows["status"].value_counts().to_dict()
    return {
        "classified": int(classified.sum()),
        "accuracy": float(accuracy),
        "correct": int((df["category"] == df["true_category"]).sum()),
        "total_tokens": result.total_tokens,
        "tokens_per_classified": result.total_tokens / int(classified.sum()) if classified.any() else 0.0,
        "status_counts": status_counts,
    }


def add_routing(rows: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    segment_lookup = segments.set_index("customer_id")[["segment_name", "is_premium"]]
    output = rows.copy()
    output = output.join(segment_lookup, on="customer_id")
    output["segment_name"] = output["segment_name"].fillna("")
    output["is_premium"] = output["is_premium"].fillna(False).astype(bool)
    routing = output.apply(
        lambda row: route_ticket(row["status"], row["category"], row["segment_name"], row["is_premium"]),
        axis=1,
    )
    output[["route", "priority"]] = pd.DataFrame(routing.tolist(), index=output.index)
    return output[[
        "ticket_id", "customer_id", "status", "category", "segment_name",
        "priority", "route", "tokens_used",
    ]]


def print_run_report(name: str, result: RunResult) -> None:
    metrics = get_metrics(result)
    print(f"\n{name}:")
    print(f"status counts: {metrics['status_counts']}")
    print(f"classified: {metrics['classified']}; accuracy: {metrics['correct']}/{metrics['classified']} ({metrics['accuracy']:.1%})")
    print(f"total response tokens: {metrics['total_tokens']}; tokens per classified ticket: {metrics['tokens_per_classified']:.2f}")
    print(f"transient errors: {result.transient_errors}; retries: {result.retries}; permanent errors: {result.permanent_errors}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assert "CLASSIFY" in ZERO_SHOT_PROMPT and "EXAMPLES:" not in ZERO_SHOT_PROMPT
    assert "CLASSIFY" in FEW_SHOT_PROMPT and "EXAMPLES:\n" in FEW_SHOT_PROMPT
    assert FEW_SHOT_PROMPT.count("Example ") >= 2 and "(fraud):" in FEW_SHOT_PROMPT
    assert validate_reply("The customer appears to have a billing-related problem.") == (
        "classified_keyword", "billing"
    )
    assert validate_reply('{"category": "shipping", "reason": "parcel movement"}') == (
        "flagged_invalid_output", ""
    )
    assert validate_reply("billing-related billing-related") == ("flagged_invalid_output", "")

    tickets = pd.read_csv(TICKETS_FILE, keep_default_na=False)
    tickets, redaction_counts = redact_tickets(tickets)
    remaining_sensitive = assert_redacted(tickets["redacted_text"])
    print(f"Tickets loaded: {len(tickets)}")
    print(f"Redactions: email={redaction_counts['email']}, phone={redaction_counts['phone']}, order_id={redaction_counts['order_id']}, total={sum(redaction_counts.values())}; remaining={remaining_sensitive}")

    eligible = tickets.loc[
        tickets["redacted_text"].str.strip().ne("")
        & tickets["redacted_text"].map(mock_llm.est_tokens).le(MAX_INPUT_TOKENS)
    ]
    empty_count = int(tickets["redacted_text"].str.strip().eq("").sum())
    long_count = int(tickets["redacted_text"].map(mock_llm.est_tokens).gt(MAX_INPUT_TOKENS).sum())
    print(f"Input guardrails: skipped_empty={empty_count}; flagged_long={long_count}; long-token estimates={sorted(tickets.loc[tickets['redacted_text'].map(mock_llm.est_tokens).gt(MAX_INPUT_TOKENS), 'redacted_text'].map(mock_llm.est_tokens).unique().tolist())}")
    assert len(tickets) == 164 and empty_count == 4 and long_count == 4
    assert sorted(tickets.loc[tickets["redacted_text"].map(mock_llm.est_tokens).gt(MAX_INPUT_TOKENS), "redacted_text"].map(mock_llm.est_tokens).unique()) == [734]
    assert redaction_counts == {"email": 55, "phone": 91, "order_id": 132}
    assert sum(redaction_counts.values()) == 278 and remaining_sensitive == 0

    # The two comparison runs have no token budget.
    zero_result = start_run(tickets, ZERO_SHOT_PROMPT)
    few_result = start_run(tickets, FEW_SHOT_PROMPT)
    print_run_report("Zero-shot run (unbudgeted)", zero_result)
    print_run_report("Few-shot run (unbudgeted)", few_result)

    zero_metrics = get_metrics(zero_result)
    few_metrics = get_metrics(few_result)
    assert zero_result.transient_errors == 19 and zero_result.permanent_errors == 0
    assert few_result.transient_errors == 19 and few_result.permanent_errors == 0
    assert zero_metrics["classified"] == 147 and zero_metrics["correct"] == 126
    assert few_metrics["classified"] == 147 and few_metrics["correct"] == 147
    assert abs(zero_metrics["accuracy"] - 0.857) <= 0.005
    assert abs(few_metrics["accuracy"] - 1.0) <= 0.005
    assert zero_result.rows["status"].value_counts().to_dict() == {
        "classified_json": 135, "classified_keyword": 12,
        "flagged_invalid_output": 9, "skipped_empty": 4, "flagged_long": 4,
    }
    assert few_result.rows["status"].value_counts().to_dict() == zero_result.rows["status"].value_counts().to_dict()

    zero_classified = zero_result.rows[
        zero_result.rows["status"].isin({"classified_json", "classified_keyword"})
    ]
    confusion = pd.crosstab(
        zero_classified["true_category"], zero_classified["category"],
        rownames=["true_category"], colnames=["predicted_category"],
    ).reindex(index=["billing", "delivery", "technical", "fraud"],
              columns=["billing", "delivery", "technical", "fraud"], fill_value=0)
    print("\nZero-shot confusion table (classified tickets only):")
    print(confusion.to_string())
    assert confusion.loc["fraud", "billing"] == 11
    assert confusion.loc["technical", "billing"] == 10
    extra_tokens = few_metrics["total_tokens"] - zero_metrics["total_tokens"]
    print(
        f"Few-shot uses {extra_tokens} extra total tokens ({few_metrics['tokens_per_classified'] - zero_metrics['tokens_per_classified']:.2f} extra per classified ticket) and raises accuracy from {zero_metrics['accuracy']:.1%} to {few_metrics['accuracy']:.1%}, correcting 21 errors."
    )

    # Budget demonstration, zero-shot, budget gate checked before every attempt.
    budget_demo = start_run(tickets, ZERO_SHOT_PROMPT, token_budget=6000)
    budget_count = int(budget_demo.rows["status"].eq("budget_exhausted").sum())
    budget_metrics = get_metrics(budget_demo)
    if budget_count == 0:
        budget_demo = start_run(tickets, ZERO_SHOT_PROMPT, token_budget=5000)
        budget_count = int(budget_demo.rows["status"].eq("budget_exhausted").sum())
        budget_used = 5000
    else:
        budget_used = 6000
    assert budget_count > 0
    segments = pd.read_csv(SEGMENTS_FILE)
    budget_out = add_routing(budget_demo.rows, segments)
    budget_out.to_csv(OUTPUT_DIR / "triage_budget_demo.csv", index=False)
    print(f"\nBudget demo: TOKEN_BUDGET={budget_used}; budget_exhausted={budget_count}; response tokens={budget_demo.total_tokens}")

    # Final few-shot run with the requested high budget and segment-based routing.
    final_result = start_run(tickets, FEW_SHOT_PROMPT, token_budget=100000)
    assert final_result.transient_errors == 19 and final_result.permanent_errors == 0
    final_metrics = get_metrics(final_result)
    assert final_metrics["status_counts"] == few_result.rows["status"].value_counts().to_dict()
    final_output = add_routing(final_result.rows, segments)
    expected_columns = [
        "ticket_id", "customer_id", "status", "category", "segment_name",
        "priority", "route", "tokens_used",
    ]
    assert list(final_output.columns) == expected_columns
    final_output.to_csv(OUTPUT_DIR / "triage_results.csv", index=False)

    print_run_report("Final few-shot run (TOKEN_BUDGET=100000)", final_result)
    print(f"Saved {OUTPUT_DIR / 'triage_budget_demo.csv'} and {OUTPUT_DIR / 'triage_results.csv'}")
    print(f"Final routes: {final_output['route'].value_counts().to_dict()}")
    print(f"Final priorities: {final_output['priority'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
