# Swiggy Customer Intelligence Capstone

This project cleans the supplied synthetic customer and order data and stores the results as CSV files and in SQLite.

## Reproduce from a clean clone

1. Create and activate a virtual environment: `python -m venv venv` followed by `.\venv\Scripts\Activate.ps1` in Windows PowerShell, `venv\Scripts\activate.bat` in Command Prompt, or `source venv/bin/activate` on macOS/Linux.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Generate all raw synthetic data: `python generate_data.py`.
4. Clean the tables and rebuild outputs: `python call.py`. This writes `customers.csv`, `products.csv`, and `orders.csv` to `data/clean/` and recreates `data/instamart.db` with `customers`, `products`, and `orders` tables.
5. Run `python run_sql_analysis.py` to execute all seven queries through pandas `read_sql_query` and print the results.
6. Run `python run_descriptive_analysis.py` to reproduce the pivot table and descriptive statistics.
7. Run `python validate.py`; it checks raw audit, cleaning, database, SQL, pivot, descriptive, and correlation checkpoints and prints exactly `ALL CHECKPOINTS PASSED` if all assertions hold.

## Cleaning rules and changes

### Customers

- Trim spaces and title-case `city`: 476 of 1,200 city values changed; the 24 raw spellings become the six expected cities.
- Map every `gender` spelling to `male`, `female`, or `unknown`: 950 of 1,200 values changed, including all 30 blanks mapped to `unknown`.
- Set ages outside 10–100 to missing: 6 values changed; the source also has 61 blank ages. Fill all 67 missing ages using the median age for that city with `groupby` and `transform`: 67 values filled, leaving zero missing ages and a final range of 18–60.
- Convert `signup_date` to datetime: all 1,200 values converted successfully. The saved CSV represents these dates as `YYYY-MM-DD` text.

### Orders and products

`clean_orders()` applies the following rules in order:

1. Convert `order_ts` to datetime: all 13,084 raw order rows converted successfully; saved/database values use `YYYY-MM-DD HH:MM:SS` text.
2. Remove exact duplicate rows: 119 rows removed.
3. Keep only the latest `order_ts` record for each `order_id`: 656 superseded status rows removed, including the 656 pending records; the result has 12,309 orders, one per ID.
4. Convert `order_amount` to numeric and turn unparseable values into missing values. The raw audit finds 120 unparseable values; after duplicate and superseded rows are removed, 112 remain and are filled from product price, quantity, and discount using `round(price_inr × quantity × (1 − discount_pct / 100), 2)`. This leaves zero missing amounts. The cleaned table has 11,110 completed orders and 1,199 cancelled orders, with no pending rows.
5. Set negative `delivery_minutes` to missing: 105 negative values remain after row deduplication and are set missing (the raw audit counted 106 before duplicates were removed). Fill 774 missing completed-order delivery times with the completed-order median for the product category using `groupby` and `transform`. Leave all 1,199 cancelled-order delivery times missing.

The 40 product rows are copied to the clean CSV and database unchanged; they provide catalogue prices and categories for the order cleaning and analysis.

The cleaned customer counts are Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, and Mumbai 184; genders are female 610, male 560, and unknown 30. The raw audit references are 13,084 order rows, 119 exact duplicates, 656 pending rows, 12,309 raw distinct order IDs, 120 unparseable raw amounts, 106 negative raw delivery times, and 2,541 blank raw delivery times. Customer raw audit references are 61 blank ages, 6 out-of-range ages, 24 city spellings, 10 non-blank gender spellings, and 30 blank genders.

## SQL analysis

The seven queries are stored in `sql/queries.sql`; `run_sql_analysis.py` runs each query against `data/instamart.db` with `pandas.read_sql_query`.

1. **Q1 — WHERE then HAVING:** 30 customers have over INR 8,000 in completed orders individually above INR 500.
2. **Q2 — three-table join:** 30 city/category revenue rows sum to INR 2,998,668.80; the largest cell is Bengaluru / Electronics at INR 233,909.45.
3. **Q3 — scalar subquery:** 11 products cost more than the average catalogue price.
4. **Q4 — LEFT JOIN:** 25 customers have no completed order.
5. **Q5 — CTE:** 550 customers exceed the average completed spend of INR 2,552.06.
6. **Q6 — ranking windows:** Top spenders are Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), and Pune C0426 (INR 10,377.10); eight rows have different `RANK()` and `ROW_NUMBER()` values because tied spend totals share a rank while `ROW_NUMBER()` still assigns each row a unique position.
7. **Q7 — running total:** 183 daily rows end at cumulative completed revenue of INR 2,998,668.80.

## Descriptive analysis

Run `python run_descriptive_analysis.py` to reproduce the Q2 pivot, city summaries, completed order amount statistics, outlier fences, and correlation matrix.

The completed-order Q2 pivot table (INR) has a grand total of INR 2,998,668.80, equal to the SQL Q2 total. Its city totals include Bengaluru INR 523,643.40 and Mumbai INR 479,058.50. Named aggregation reports Bengaluru sum / mean / count as INR 523,643.40 / INR 276.47 / 1,894 and Mumbai as INR 479,058.50 / INR 297.00 / 1,613.

For the 11,110 completed orders, `order_amount` mean is INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q3 INR 297.00, and IQR INR 232.20. The 1.5×IQR fences are INR −283.50 and INR 645.30; 0 values are below the lower fence and 897 are above the upper fence.

**Interpretation (four questions):** This dataset describes synthetic Instamart customer orders, their products, prices, discounts, and delivery times. Completed order amounts are strongly right-skewed: the mean (INR 269.91) is well above the median (INR 135.00), with Q1 INR 64.80 and Q3 INR 297.00. There are 897 high outliers above INR 645.30 and none below the lower fence. Amount and discount have the largest absolute correlation (−0.095), while amount and quantity correlate at +0.087; both relationships are weak, and correlation of this size does not establish causation.

The completed-order correlation matrix is:

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---:|---:|---:|---:|
| order_amount | 1.000000 | −0.095106 | 0.087294 | −0.003135 |
| discount_pct | −0.095106 | 1.000000 | −0.000359 | −0.009808 |
| quantity | 0.087294 | −0.000359 | 1.000000 | −0.012623 |
| delivery_minutes | −0.003135 | −0.009808 | −0.012623 | 1.000000 |




