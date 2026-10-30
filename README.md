# Swiggy Customer Intelligence Capstone

## Overview

This project cleans and analyzes supplied synthetic Instamart customer, order, and support-ticket data. It includes SQL analysis, RFM segmentation, lapse prediction, privacy-aware ticket triage, policy retrieval, and a tool-using refund agent. Clean tables are saved as CSV and SQLite outputs.

## Reproduce

1. Create and activate a virtual environment: `python -m venv venv` followed by `.\venv\Scripts\Activate.ps1` in Windows PowerShell, `venv\Scripts\activate.bat` in Command Prompt, or `source venv/bin/activate` on macOS/Linux.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Generate all raw synthetic data: `python generate_data.py`.
4. Clean the tables and rebuild outputs: `python call.py`. This writes `customers.csv`, `products.csv`, and `orders.csv` to `data/clean/` and recreates `data/instamart.db` with `customers`, `products`, and `orders` tables.
5. Run `python run_sql_analysis.py` to execute all seven queries through pandas `read_sql_query` and print the results.
6. Run `python run_descriptive_analysis.py` to reproduce the pivot table and descriptive statistics.
7. Run `python customer_analytics.py` to build RFM segments, train and evaluate lapse models, and write the three outputs under `outputs/`.
8. Run `python ticket_triage.py` to redact and classify support tickets, compare prompts, and write the budget demo and routed results.
9. Run `python policy_agent.py` to evaluate policy retrieval, process refund scenarios, and write run, ledger, and checkpoint outputs.
10. Run `python validate.py`; it checks raw audit, cleaning, database, SQL, pivot, descriptive, segmentation, model, triage, and policy-agent outputs, then prints exactly `ALL CHECKPOINTS PASSED` if all assertions hold.

## Part A

### Cleaning rules and changes

### Customers

- Trim spaces and title-case `city`: 476 of 1,200 city values changed; the 24 raw spellings become the six expected cities.
- Map every `gender` spelling to `male`, `female`, or `unknown`: 950 of 1,200 values changed, including all 30 blanks mapped to `unknown`.
- Set ages outside 10-100 to missing: 6 values changed; the source also has 61 blank ages. Fill all 67 missing ages using the median age for that city with `groupby` and `transform`: 67 values filled, leaving zero missing ages and a final range of 18-60.
- Convert `signup_date` to datetime: all 1,200 values converted successfully. The saved CSV represents these dates as `YYYY-MM-DD` text.

### Orders and products

`clean_orders()` applies the following rules in order:

1. Convert `order_ts` to datetime: all 13,084 raw order rows converted successfully; saved/database values use `YYYY-MM-DD HH:MM:SS` text.
2. Remove exact duplicate rows: 119 rows removed.
3. Keep only the latest `order_ts` record for each `order_id`: 656 superseded status rows removed, including the 656 pending records; the result has 12,309 orders, one per ID.
4. Convert `order_amount` to numeric and turn unparseable values into missing values. The raw audit finds 120 unparseable values; after duplicate and superseded rows are removed, 112 remain and are filled from product price, quantity, and discount using `round(price_inr x quantity x (1 - discount_pct / 100), 2)`. This leaves zero missing amounts. The cleaned table has 11,110 completed orders and 1,199 cancelled orders, with no pending rows.
5. Set negative `delivery_minutes` to missing: 105 negative values remain after row deduplication and are set missing (the raw audit counted 106 before duplicates were removed). Fill 774 missing completed-order delivery times with the completed-order median for the product category using `groupby` and `transform`. Leave all 1,199 cancelled-order delivery times missing.

The 40 product rows are copied to the clean CSV and database unchanged; they provide catalogue prices and categories for the order cleaning and analysis.

The cleaned customer counts are Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, and Mumbai 184; genders are female 610, male 560, and unknown 30. The raw audit references are 13,084 order rows, 119 exact duplicates, 656 pending rows, 12,309 raw distinct order IDs, 120 unparseable raw amounts, 106 negative raw delivery times, and 2,541 blank raw delivery times. Customer raw audit references are 61 blank ages, 6 out-of-range ages, 24 city spellings, 10 non-blank gender spellings, and 30 blank genders.

### SQL analysis

The seven queries are stored in `sql/queries.sql`; `run_sql_analysis.py` runs each query against `data/instamart.db` with `pandas.read_sql_query`.

1. **Q1 - WHERE then HAVING:** 30 customers have over INR 8,000 in completed orders individually above INR 500.
2. **Q2 - three-table join:** 30 city/category revenue rows sum to INR 2,998,668.80; the largest cell is Bengaluru / Electronics at INR 233,909.45.
3. **Q3 - scalar subquery:** 11 products cost more than the average catalogue price.
4. **Q4 - LEFT JOIN:** 25 customers have no completed order.
5. **Q5 - CTE:** 550 customers exceed the average completed spend of INR 2,552.06.
6. **Q6 - ranking windows:** Top spenders are Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), and Pune C0426 (INR 10,377.10); eight rows have different `RANK()` and `ROW_NUMBER()` values because tied spend totals share a rank while `ROW_NUMBER()` still assigns each row a unique position.
7. **Q7 - running total:** 183 daily rows end at cumulative completed revenue of INR 2,998,668.80.

### Descriptive analysis

Run `python run_descriptive_analysis.py` to reproduce the Q2 pivot, city summaries, completed order amount statistics, outlier fences, and correlation matrix.

The completed-order Q2 pivot table (INR) has a grand total of INR 2,998,668.80, equal to the SQL Q2 total. Its city totals include Bengaluru INR 523,643.40 and Mumbai INR 479,058.50. Named aggregation reports Bengaluru sum / mean / count as INR 523,643.40 / INR 276.47 / 1,894 and Mumbai as INR 479,058.50 / INR 297.00 / 1,613.

For the 11,110 completed orders, `order_amount` mean is INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q3 INR 297.00, and IQR INR 232.20. The 1.5xIQR fences are INR -283.50 and INR 645.30; 0 values are below the lower fence and 897 are above the upper fence.

**Interpretation (four questions):** This dataset describes synthetic Instamart customer orders, their products, prices, discounts, and delivery times. Completed order amounts are strongly right-skewed: the mean (INR 269.91) is well above the median (INR 135.00), with Q1 INR 64.80 and Q3 INR 297.00. There are 897 high outliers above INR 645.30 and none below the lower fence. Amount and discount have the largest absolute correlation (-0.095), while amount and quantity correlate at +0.087; both relationships are weak, and correlation of this size does not establish causation.

The completed-order correlation matrix is:

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---:|---:|---:|---:|
| order_amount | 1.000000 | -0.095106 | 0.087294 | -0.003135 |
| discount_pct | -0.095106 | 1.000000 | -0.000359 | -0.009808 |
| quantity | 0.087294 | -0.000359 | 1.000000 | -0.012623 |
| delivery_minutes | -0.003135 | -0.009808 | -0.012623 | 1.000000 |





## Part B: RFM segmentation and lapse modeling

Run `python customer_analytics.py` after the clean-table steps above. It creates `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, and `outputs/retention_list.csv`. It uses the cleaned CSVs; clustering and lapse preprocessing are fit inside the script, with lapse preprocessing fit only on training folds/rows.

### RFM and K-means

RFM uses completed orders as of 2025-10-01 00:00:00. Recency is whole days since the most recent completed order, frequency is distinct completed order count, and monetary is completed order amount summed per customer. The resulting 1,175 customers have median recency 19 days (range 0-180), mean frequency 9.46 (maximum 33), and mean monetary value INR 2,552.06 (maximum INR 14,766.55).

After StandardScaler, K-means uses `n_init=10` and `random_state=42` for K=2-8:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected because the elbow is around four, K=4 has the highest silhouette score, and four profiles support distinct retention actions. Inertia decreases as K rises because more centroids can only reduce within-cluster squared distance; K=8 therefore has lower inertia than K=2. Inertia alone cannot select K because it keeps decreasing as clusters are added, so the elbow, silhouette and business usefulness also matter.

| Segment ID | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Name | Recommended action |
|---:|---:|---:|---:|---:|---|---|
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Occasional Customers | Nurture with relevant repeat-purchase promotions |
| 1 | 138 | 24.12 | 6.77 | 7,129.70 | Premium Loyalists - **Premium** | Offer high-value loyalty benefits |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 | Active Regulars | Send a timely reorder reminder |
| 3 | 182 | 109.04 | 2.20 | 685.99 | Dormant Customers | Use a targeted win-back offer |

Cluster numbers are arbitrary labels; they do not rank customers or segments. Premium marks the segment with the highest mean monetary value. PCA on scaled RFM explains 0.634 and 0.218 of variance in its first two components, 0.852 total.

### Lapse prediction

The lapse cohort includes every customer with a completed order before 2025-09-01. A customer is labeled lapsed when they have no completed order from 2025-09-01 through 2025-09-30. The cohort has 1,156 customers, including 443 lapsed (38.32%); the stratified split has 924 training and 232 test rows. Features are calculated only from orders before the cutoff plus the clean customer table: recency, distinct completed-order frequency, completed spend, completed-order average discount, cancellation rate over final-status orders, completed-order average delivery time, favourite completed-order product category, city, age, gender, and tenure. The script asserts that no feature value is missing. No order dated 2025-09-01 or later enters the feature matrix; September orders are used only to create the outcome labels. The task 3 segment labels are not included as features.

A baseline that predicts not lapsed for every test customer has 61.6% accuracy and lapsed-class F1 of 0.00. Both models use a training-only Pipeline with StandardScaler for numeric features and OneHotEncoder(drop="first") for city, gender and favourite category. No encoder, scaler, or resampler saw the test rows: encoders and scalers are fitted only inside training folds and then on the full training split, while resampling uses training rows only. The held-out test split is used only for final model evaluation. Logistic regression (`max_iter=1000`) used `GridSearchCV` over C = {0.01, 0.1, 1, 10, 100}, with `cv=5` and `scoring="f1"`; its best C is 0.1 and best CV F1 is 0.639. Random forest (`n_estimators=100`, `random_state=42`) used `GridSearchCV` over `max_depth` = {3, 5, 8} and `min_samples_leaf` = {5, 10, 20}, with `cv=5` and `scoring="f1"` (45 fits); its best parameters are depth 5 and leaf size 10, with best CV F1 0.676.

| Model | Test confusion matrix (actual rows: not lapsed, lapsed; predicted columns: not lapsed, lapsed) | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---:|---:|
| Logistic regression | `[[105, 38], [17, 72]]` | 0.655 | 0.809 | 0.724 |
| Random forest | `[[111, 32], [23, 66]]` | 0.673 | 0.742 | 0.706 |

All precision, recall and F1 scores in this table are for the lapsed class (`lapsed=1`).

The random forest is selected because its cross-validated F1 (0.676) is higher than logistic regression's (0.639), and its held-out F1 is 0.706. Missing a lapsing customer loses a chance to retain that customer and their future spend, so recall matters to the retention team. Wrongly targeting a loyal customer wastes offer budget and may annoy them; the forest's 0.673 precision and 0.742 recall provide a practical tradeoff.

The frequency mean learned by the fitted training scaler is 8.116, versus 8.029 for the full cohort; they differ because the scaler sees only training rows, while the latter includes held-out customers. Training-only oversampling (replacement sampling with `random_state=42`) keeps the original 570 not-lapsed and 354 lapsed rows, then duplicates lapsed rows with replacement to reach 570 in each class. On the unchanged test set, logistic precision/recall/F1 moves from 0.655/0.809/0.724 to 0.587/0.910/0.714; recall moves most, improving lapse detection while increasing false-positive outreach and reducing precision.

The five most important random-forest features and business readings are frequency (0.318; repeat cadence is the strongest signal), monetary (0.277; customer value helps prioritize retention effort), recency (0.098; purchase timing can guide outreach), favourite category Electronics (0.059; category preference can tailor offers), and cancellation rate (0.055; order friction can signal a need for service recovery). In logistic regression, the recency coefficient is negative (-0.0285), so higher recency is associated with lower predicted lapse odds after controlling for the other features; frequency is negative (-1.2811), so higher frequency is associated with lower predicted lapse odds; monetary is positive (+0.0295), so higher monetary value is associated with higher predicted lapse odds. These are conditional associations, not causal effects. The selected joblib model was reloaded and reproduced every test prediction exactly.

The Premium retention list contains the top 10 Premium test customers by predicted lapse probability; 2 of the 10 were actually lapsed. The list is written with `customer_id`, `lapse_probability`, and `actually_lapsed` columns.

## Part C: Support-ticket triage

Run `python ticket_triage.py` after running `python customer_analytics.py`. The script processes `data/raw/tickets.csv` in file order and calls only the supplied offline `mock_llm.py`. The only user message sent for a ticket is its redacted text, with no ID or other ticket metadata. No raw ticket text is sent to an outside model.

### Input guardrails, privacy and output checks

`est_tokens` is applied before any model call. Four empty/whitespace tickets are `skipped_empty`; four tickets estimated at 734 tokens are `flagged_long` because they exceed the 400-token limit. They are never sent to the mock model. Across all 164 tickets, the ordered email, phone and order-ID regexes made 55, 91 and 132 substitutions respectively (278 total); rechecking all redacted text found zero matches. The script asserts the redaction patterns find nothing immediately before each model call, and puts the redacted ticket text alone in the user message.

Both system prompts contain `CLASSIFY`, name billing, delivery, technical and fraud, and require a JSON-only `{"category": "...", "reason": "..."}` reply. The zero-shot prompt has no `EXAMPLES:` marker. The few-shot prompt includes an `EXAMPLES:` line with labelled fraud and delivery examples. A JSON category outside the allowed set is flagged invalid. For non-JSON replies, the pipeline extracts words using `re.findall(r"[a-z]+", text.lower())` and accepts only exactly one allowed category word; otherwise it flags the reply for human review. Only `classified_json` and `classified_keyword` outputs passed these category rules and are eligible for automatic routing. `skipped_empty`, `flagged_long`, `flagged_invalid_output`, and `budget_exhausted` outputs go to `human_review` at normal priority.

Each call retries `TransientLLMError` at most three times with exponential backoff and jitter using `BASE_DELAY=0.05`; every retry logs ticket ID, retry number and wait, and the code asserts the sampled wait is inside the range for that retry. In these runs every retry was number 0, with observed waits from 0.0453 to 0.0562 seconds, within the required 0.045 to 0.0575 range. Each unbudgeted full run had 19 transient errors, all recovered, and zero permanent errors. `mock_llm.reset()` runs immediately before each independent full run.

### Zero-shot and few-shot comparison

The comparison runs have no token budget and count accuracy only over tickets classified as JSON or keyword. Token totals sum each successful response's `usage.total_tokens`, including prompt and completion tokens; tokens per classified ticket divide that total by 147.

| Run | Classified | Correct / accuracy | Total tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 126 / 85.7% | 11,116 | 75.62 |
| Few-shot | 147 | 147 / 100.0% | 16,576 | 112.76 |

Zero-shot confusion table (true category by predicted category, classified tickets only):

| True \ Predicted | Billing | Delivery | Technical | Fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

Few-shot used 5,460 extra tokens, or 37.14 more per classified ticket, and raised accuracy from 126/147 (85.7%) to 147/147 (100%), correcting 21 errors. Those errors were 11 fraud and 10 technical tickets routed as billing by the zero-shot prompt. The extra prompt cost is worthwhile when preventing these misroutes, especially fraud, matters more than 5,460 tokens; for lower-risk use, zero-shot with human review may be the cheaper choice.

### Budget and segment routing

The zero-shot budget demo uses `TOKEN_BUDGET=6000` and writes `outputs/triage_budget_demo.csv`; 72 tickets reached `budget_exhausted`. The token gate is checked before every initial call and retry, so a successful response may take the running total just past the budget; after it reaches or exceeds the limit, no further call is made.

The final few-shot run uses `TOKEN_BUDGET=100000` and writes `outputs/triage_results.csv` with `ticket_id`, `customer_id`, `status`, `category`, `segment_name`, `priority`, `route`, and `tokens_used`. It yields 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `skipped_empty`, and 4 `flagged_long` statuses. Invalid/empty/long/budget-exhausted cases route to `human_review` at normal priority; valid fraud routes to `security_team` at high priority; other Premium customers route to `premium_desk` at high priority; remaining tickets route to `support_queue` at normal priority. A missing segment record is treated as not Premium. The final output routes 102 tickets to support, 34 to security, 11 to the Premium desk and 17 to human review; 45 have high priority.

## Part D: policy retrieval and refund agent

Run `python policy_agent.py` after generating `data/instamart.db` and `outputs/customer_segments.csv`. The program uses `mock_llm.py` for every model call; it makes no network calls and reads or uses no API key.

### Chunking, retrieval and grounded answers

The five policy files are read in filename order and split into 40-word chunks with a 10-word overlap: starts are 0, 30, 60, and so on, stopping as soon as a chunk reaches the document end. This produces 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2. `TfidfVectorizer()` is fit on those chunk texts and `cosine_similarity` ranks each query; for "refund policy late delivery" the top three sources are all `refund_policy.txt`.

TF measures how often a word appears in a chunk, while IDF reduces the weight of words common across chunks and gives more weight to distinctive words. TF-IDF is lexical, so two phrasings with the same meaning but different words can have low similarity.

`answer(question)` retrieves three chunks and sends the model a POLICY ASSISTANT system instruction plus a user message in the required `CONTEXT:` / `[source]` / `QUESTION:` format. It returns the answer text and the sources used. Evaluation is case-sensitive for `expected_fact` and checks whether unanswerable answers start with `I could not find`.

| Chunk overlap | Answerable expected source in top three | Expected fact in answer | Unanswerable fallbacks |
|---|---:|---:|---:|
| 10 words | 12/12 | 12/12 | 2/2 |
| 0 words | 12/12 | 10/12 | 2/2 |

With zero overlap, "What delivery time makes a completed order eligible for a full refund?" fails the fact check: one chunk ends with the threshold and the next starts with the refund consequence, so the mock extracts only "receives a full refund" without the condition "more than 45 minutes." The overlapping chunks retain the complete sentence together and restore the answer.

### Refund tools, ReAct runs and safeguards

`TOOLS_SPEC` describes the five functions in `TOOL_MAP`: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. Order lookup reads `data/instamart.db`; refund recording tools append timestamped JSON lines to `outputs/refund_ledger.jsonl`. The tools enforce completed-order eligibility, the 45-minute delivery rule, and the INR 1000 escalation threshold. Every observation is added to the message history and checkpointed under `state/` with the run ID, scenario ID, full messages, cumulative tokens, step number, and tools executed.

| Scenario | Status | Refund amount | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 | 5 | 1,172 |
| S2 | REFUND | INR 135.00 | 5 | 1,172 |
| S3 | ESCALATE | INR 1,999.00 | 5 | 1,159 |
| S4 | DENIED | INR 0.00 | 4 | 825 |
| S5 | DENIED | INR 0.00 | 3 | 502 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 182 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 3,330 |

The S1 token-limit demonstration (`token_limit=200`) ends as `FALLBACK_TOKEN_LIMIT` after 2 model calls and 212 cumulative tokens; the tool requested by the over-limit response is not executed. The S1 timeout demonstration (`time_limit_seconds=0`) ends as `FALLBACK_TIMEOUT` before its first model call. S7 consumes 3,330 tokens, completes all six calls under the default 6,000-token limit, then safely hands off after reaching the iteration limit.

The resumability demonstration crashes after the second tool observation has been checkpointed, then resumes S1. There are exactly four tool executions across the crash and resume, in this sequence: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`; no tool is repeated, and the resumed result is REFUND for INR 128.25. The seven normal scenario runs are stored as JSONL records in `outputs/agent_runs.jsonl`, with each step's tool, arguments and observation.

### ReAct versus plan-and-execute (under 250 words)

Measured ReAct usage was 5 calls and 1,172 tokens for S1, 3 calls and 502 tokens for S5, and 2 calls and 182 tokens for S6. A plan-and-execute agent could ask the model for a full action plan once and run routine tools without sending every observation back to the model, which may reduce calls and prompt tokens; a larger initial plan and any replan after an error add cost. ReAct responds to observations as they arrive: S6 returns NOT_FOUND after its lookup, while S5 sees a cancelled order and returns DENIED without calculating or issuing a refund. A fixed plan must explicitly branch on those observations or replan, and tools must still reject unsafe actions. For this refund workflow I would choose ReAct because the extra model calls support step-by-step decisions from live order and policy observations, while tool-side checks prevent the model from issuing an invalid refund.

## Limitations and Next Steps

The Premium flag is a post-hoc whole-window label because its RFM uses September, the same period used to define the lapse label, so the retention list is illustrative and not a leakage-free evaluation; the model features themselves still use only pre-cutoff orders. A production evaluation should define valuable customers using a historical window that ends before the prediction cutoff and validate the model on a later, untouched period.
