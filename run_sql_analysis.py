from pathlib import Path
import re
import sqlite3

import pandas as pd

SQL_FILE = Path("sql/queries.sql")
DB_FILE = Path("data/instamart.db")

sql_text = SQL_FILE.read_text(encoding="utf-8-sig")
statements = re.split(r"(?=-- Q[1-7] —)", sql_text)
queries = {}
for statement in statements:
    match = re.match(r"-- (Q[1-7]) —[^\n]*\n", statement)
    if match:
        queries[match.group(1)] = statement[match.end():].strip()

with sqlite3.connect(DB_FILE) as connection:
    results = {name: pd.read_sql_query(query, connection) for name, query in queries.items()}

q1 = results["Q1"]
q2 = results["Q2"]
q3 = results["Q3"]
q4 = results["Q4"]
q5 = results["Q5"]
q6 = results["Q6"]
q7 = results["Q7"]
print(f"Q1: {len(q1)} customers")
print(f"Q2: {len(q2)} city/category rows; revenue total INR {q2['revenue'].sum():,.2f}; largest: {q2.loc[q2['revenue'].idxmax(), 'city']} / {q2.loc[q2['revenue'].idxmax(), 'category']} INR {q2['revenue'].max():,.2f}")
print(f"Q3: {len(q3)} products above average price")
print(f"Q4: {len(q4)} customers with no completed order")
print(f"Q5: {len(q5)} customers above average completed spend; average INR {q5['avg_spend'].iloc[0]:,.2f}")
print("Q6 top customer per city:")
print(q6.sort_values(['city', 'spend_rank']).groupby('city', as_index=False).first()[['city','customer_id','total_spend']].to_string(index=False))
print(f"Q6 rows where RANK differs from ROW_NUMBER: {(q6['spend_rank'] != q6['spend_row_number']).sum()}")
print(f"Q7: {len(q7)} daily rows; last cumulative revenue INR {q7['cumulative_revenue'].iloc[-1]:,.2f}")
