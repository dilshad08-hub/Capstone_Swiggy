from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

DB_FILE = Path("data/instamart.db")
SQL_FILE = Path("sql/queries.sql")


def get_query(label: str) -> str:
    text = SQL_FILE.read_text(encoding="utf-8-sig")
    pieces = re.split(r"(?=-- Q[1-7] —)", text)
    for piece in pieces:
        match = re.match(r"-- (Q[1-7]) —[^\n]*\n", piece)
        if match and match.group(1) == label:
            return piece[match.end():].strip()
    raise ValueError(f"{label} not found in {SQL_FILE}")


with sqlite3.connect(DB_FILE) as connection:
    q2_sql = pd.read_sql_query(get_query("Q2"), connection)
    sql_total = pd.read_sql_query(
        "SELECT SUM(order_amount) AS total FROM orders WHERE status='completed'",
        connection,
    ).loc[0, "total"]
    completed = pd.read_sql_query(
        """
        SELECT o.order_amount, o.discount_pct, o.quantity, o.delivery_minutes,
               c.city, p.category
        FROM orders AS o
        JOIN customers AS c ON c.customer_id = o.customer_id
        JOIN products AS p ON p.product_id = o.product_id
        WHERE o.status = 'completed'
        """,
        connection,
    )

pivot = pd.pivot_table(
    completed,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    margins=True,
    margins_name="All",
    fill_value=0,
)
pivot_total = pivot.loc["All", "All"]
assert np.isclose(pivot_total, sql_total, atol=0.005)

city_summary = completed.groupby("city").agg(
    sum=("order_amount", "sum"),
    mean=("order_amount", "mean"),
    count=("order_amount", "count"),
)

amount = completed["order_amount"]
q1, q3 = amount.quantile([0.25, 0.75])
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
correlation = completed[
    ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
].corr()

print("Q2 pivot table (INR):")
print(pivot.round(2).to_string())
print(f"Pivot grand total: INR {pivot_total:,.2f}")
print(f"SQL Q2 grand total: INR {q2_sql['revenue'].sum():,.2f}")
print("\nCompleted order amount by city:")
city_summary = city_summary.rename(columns={"sum": "sum_inr", "mean": "mean_inr"})
print(city_summary.to_string(formatters={"sum_inr": "{:,.2f}".format, "mean_inr": "{:,.2f}".format}))
print("\nCompleted order_amount distribution:")
print(f"count: {len(amount)}")
print(f"mean: INR {amount.mean():.2f}")
print(f"median: INR {amount.median():.2f}")
print(f"skewness: {amount.skew():.2f}")
print(f"Q1: INR {q1:.2f}")
print(f"Q3: INR {q3:.2f}")
print(f"IQR: INR {iqr:.2f}")
print(f"lower 1.5×IQR fence: INR {lower_fence:.2f}")
print(f"upper 1.5×IQR fence: INR {upper_fence:.2f}")
print(f"outliers below lower fence: {(amount < lower_fence).sum()}")
print(f"outliers above upper fence: {(amount > upper_fence).sum()}")
print("\nCorrelation matrix:")
print(correlation.round(6).to_string())


