from pathlib import Path
import re
import sqlite3

import pandas as pd

DB_FILE = Path("data/instamart.db")
SQL_FILE = Path("sql/queries.sql")


def load_queries() -> dict[str, str]:
    text = SQL_FILE.read_text(encoding="utf-8-sig")
    pieces = re.split(r"(?=-- Q[1-7] —)", text)
    queries = {}
    for piece in pieces:
        match = re.match(r"-- (Q[1-7]) —[^\n]*\n", piece)
        if match:
            queries[match.group(1)] = piece[match.end():].strip()
    return queries


def main() -> None:
    raw_dir = Path("data/raw")
    clean_dir = Path("data/clean")
    raw_orders = pd.read_csv(raw_dir / "orders.csv")
    raw_customers = pd.read_csv(raw_dir / "customers.csv")
    customers_csv = pd.read_csv(clean_dir / "customers.csv")
    products_csv = pd.read_csv(clean_dir / "products.csv")
    orders_csv = pd.read_csv(clean_dir / "orders.csv")

    # Raw audit checkpoints.
    assert len(raw_orders) == 13084
    assert raw_orders.duplicated().sum() == 119
    assert raw_orders["status"].eq("pending").sum() == 656
    assert raw_orders["order_id"].nunique() == 12309
    assert pd.to_numeric(raw_orders["order_amount"], errors="coerce").isna().sum() == 120
    raw_delivery = pd.to_numeric(raw_orders["delivery_minutes"], errors="coerce")
    assert raw_delivery.lt(0).sum() == 106
    assert raw_delivery.isna().sum() == 2541
    raw_age = pd.to_numeric(raw_customers["age"], errors="coerce")
    assert raw_age.isna().sum() == 61
    assert (raw_age.notna() & ~raw_age.between(10, 100)).sum() == 6
    assert raw_customers["city"].nunique() == 24
    assert raw_customers["gender"].dropna().nunique() == 10
    assert raw_customers["gender"].isna().sum() == 30

    # Clean customer checkpoints.
    expected_cities = {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197,
        "Chennai": 191, "Hyderabad": 191, "Mumbai": 184,
    }
    assert customers_csv["city"].value_counts().to_dict() == expected_cities
    assert customers_csv["city"].nunique() == 6
    assert customers_csv["gender"].value_counts().to_dict() == {
        "female": 610, "male": 560, "unknown": 30,
    }
    assert customers_csv["age"].isna().sum() == 0
    assert customers_csv["age"].between(18, 60).all()
    assert pd.to_datetime(customers_csv["signup_date"], errors="coerce").notna().all()
    assert len(products_csv) == 40

    # Clean order checkpoints.
    assert len(orders_csv) == 12309
    assert orders_csv["order_id"].nunique() == 12309
    assert orders_csv["status"].value_counts().to_dict() == {
        "completed": 11110, "cancelled": 1199,
    }
    assert orders_csv["order_amount"].isna().sum() == 0
    completed_mask = orders_csv["status"].eq("completed")
    cancelled_mask = orders_csv["status"].eq("cancelled")
    assert orders_csv.loc[completed_mask, "delivery_minutes"].isna().sum() == 0
    assert orders_csv.loc[cancelled_mask, "delivery_minutes"].isna().sum() == 1199
    completed_revenue = orders_csv.loc[completed_mask, "order_amount"].sum()
    assert round(completed_revenue, 2) == 2998668.80

    queries = load_queries()
    assert set(queries) == {f"Q{i}" for i in range(1, 8)}
    with sqlite3.connect(DB_FILE) as connection:
        tables = set(pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", connection
        )["name"])
        sql_results = {
            label: pd.read_sql_query(query, connection)
            for label, query in queries.items()
        }
        order_ts_types = pd.read_sql_query(
            "SELECT DISTINCT typeof(order_ts) AS storage_type FROM orders", connection
        )["storage_type"].tolist()
        completed_join = pd.read_sql_query(
            """
            SELECT o.order_amount, o.discount_pct, o.quantity, o.delivery_minutes,
                   c.city, p.category
            FROM orders o
            JOIN customers c ON c.customer_id=o.customer_id
            JOIN products p ON p.product_id=o.product_id
            WHERE o.status='completed'
            """, connection,
        )

    assert {"customers", "products", "orders"}.issubset(tables)
    assert order_ts_types == ["text"]
    assert len(sql_results["Q1"]) == 30

    q2 = sql_results["Q2"]
    assert len(q2) == 30
    assert round(q2["revenue"].sum(), 2) == 2998668.80
    max_cell = q2.loc[q2["revenue"].idxmax()]
    assert (max_cell["city"], max_cell["category"], round(max_cell["revenue"], 2)) == (
        "Bengaluru", "Electronics", 233909.45,
    )
    assert len(sql_results["Q3"]) == 11
    assert len(sql_results["Q4"]) == 25
    q5 = sql_results["Q5"]
    assert len(q5) == 550
    assert round(q5["avg_spend"].iloc[0], 2) == 2552.06

    q6 = sql_results["Q6"]
    expected_tops = {
        "Bengaluru": ("C0082", 12856.20), "Chennai": ("C0930", 10151.40),
        "Delhi": ("C0291", 13742.60), "Hyderabad": ("C0743", 12730.05),
        "Mumbai": ("C0693", 14766.55), "Pune": ("C0426", 10377.10),
    }
    top_rows = q6.sort_values(["city", "spend_rank", "customer_id"]).groupby("city").first()
    assert set(top_rows.index) == set(expected_tops)
    for city, (customer_id, spend) in expected_tops.items():
        assert top_rows.loc[city, "customer_id"] == customer_id
        assert round(top_rows.loc[city, "total_spend"], 2) == spend
    assert (q6["spend_rank"] != q6["spend_row_number"]).sum() == 8

    q7 = sql_results["Q7"]
    assert len(q7) == 183
    assert round(q7["cumulative_revenue"].iloc[-1], 2) == 2998668.80

    # Pivot, per-city aggregation, distribution, outlier, and correlation references.
    pivot = pd.pivot_table(
        completed_join, values="order_amount", index="city", columns="category",
        aggfunc="sum", margins=True, fill_value=0,
    )
    assert round(pivot.loc["All", "All"], 2) == round(q2["revenue"].sum(), 2)
    city_summary = completed_join.groupby("city").agg(
        sum=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count")
    )
    assert round(city_summary.loc["Bengaluru", "sum"], 2) == 523643.40
    assert round(city_summary.loc["Bengaluru", "mean"], 2) == 276.47
    assert city_summary.loc["Bengaluru", "count"] == 1894
    assert round(city_summary.loc["Mumbai", "sum"], 2) == 479058.50
    assert round(city_summary.loc["Mumbai", "mean"], 2) == 297.00
    assert city_summary.loc["Mumbai", "count"] == 1613

    amount = completed_join["order_amount"]
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr = q3 - q1
    assert round(amount.mean(), 2) == 269.91
    assert round(amount.median(), 2) == 135.00
    assert round(amount.skew(), 2) == 4.61
    assert round(q1, 2) == 64.80 and round(q3, 2) == 297.00
    assert round(q1 - 1.5 * iqr, 2) == -283.50
    assert round(q3 + 1.5 * iqr, 2) == 645.30
    assert (amount < q1 - 1.5 * iqr).sum() == 0
    assert (amount > q3 + 1.5 * iqr).sum() == 897

    corr = completed_join[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    assert round(corr.loc["order_amount", "discount_pct"], 3) == -0.095
    assert round(corr.loc["order_amount", "quantity"], 3) == 0.087
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
