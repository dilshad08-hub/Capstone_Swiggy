from pathlib import Path
import sqlite3

import pandas as pd


RAW_CUSTOMERS = Path("data/raw/customers.csv")
RAW_ORDERS = Path("data/raw/orders.csv")
RAW_PRODUCTS = Path("data/raw/products.csv")
CLEAN_DIR = Path("data/clean")
DATABASE = Path("data/instamart.db")


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Standardise customer fields and impute ages from city medians."""
    cleaned = customers.copy()
    cleaned["city"] = cleaned["city"].astype("string").str.strip().str.title()

    gender = cleaned["gender"].astype("string").str.strip().str.lower()
    cleaned["gender"] = gender.map(
        {"male": "male", "m": "male", "female": "female", "f": "female"}
    ).fillna("unknown")

    cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce")
    cleaned.loc[~cleaned["age"].between(10, 100), "age"] = pd.NA
    city_medians = cleaned.groupby("city")["age"].transform("median")
    cleaned["age"] = cleaned["age"].fillna(city_medians).astype("int64")
    cleaned["signup_date"] = pd.to_datetime(cleaned["signup_date"], errors="coerce")
    return cleaned


def clean_orders(orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, repair amounts, and impute completed delivery times."""
    cleaned = orders.copy()
    cleaned["order_ts"] = pd.to_datetime(cleaned["order_ts"], errors="coerce")
    cleaned = cleaned.drop_duplicates().copy()
    cleaned = cleaned.sort_values("order_ts", kind="stable")
    cleaned = cleaned.drop_duplicates(subset="order_id", keep="last").copy()

    product_info = products[["product_id", "price_inr", "category"]].drop_duplicates("product_id")
    cleaned = cleaned.merge(product_info, on="product_id", how="left", validate="many_to_one")

    cleaned["order_amount"] = pd.to_numeric(cleaned["order_amount"], errors="coerce")
    calculated_amount = (
        cleaned["price_inr"]
        * cleaned["quantity"]
        * (1 - cleaned["discount_pct"] / 100)
    ).round(2)
    cleaned["order_amount"] = cleaned["order_amount"].fillna(calculated_amount)

    cleaned["delivery_minutes"] = pd.to_numeric(cleaned["delivery_minutes"], errors="coerce")
    cleaned.loc[cleaned["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA
    completed = cleaned["status"].eq("completed")
    completed_medians = cleaned.loc[completed].groupby("category")["delivery_minutes"].transform("median")
    cleaned.loc[completed, "delivery_minutes"] = cleaned.loc[completed, "delivery_minutes"].fillna(completed_medians)
    return cleaned.drop(columns=["price_inr", "category"])


def load_clean_customers(path: Path = RAW_CUSTOMERS) -> pd.DataFrame:
    return clean_customers(pd.read_csv(path))


def load_clean_orders(
    orders_path: Path = RAW_ORDERS, products_path: Path = RAW_PRODUCTS
) -> pd.DataFrame:
    return clean_orders(pd.read_csv(orders_path), pd.read_csv(products_path))


def write_outputs(customers: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE.parent.mkdir(parents=True, exist_ok=True)

    customers_csv = customers.copy()
    customers_csv["signup_date"] = customers_csv["signup_date"].dt.strftime("%Y-%m-%d")
    orders_csv = orders.copy()
    orders_csv["order_ts"] = orders_csv["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    customers_csv.to_csv(CLEAN_DIR / "customers.csv", index=False)
    products.to_csv(CLEAN_DIR / "products.csv", index=False)
    orders_csv.to_csv(CLEAN_DIR / "orders.csv", index=False)

    with sqlite3.connect(DATABASE) as connection:
        customers_csv.to_sql("customers", connection, if_exists="replace", index=False)
        products.to_sql("products", connection, if_exists="replace", index=False)
        orders_csv.to_sql("orders", connection, if_exists="replace", index=False)


if __name__ == "__main__":
    customers = load_clean_customers()
    products = pd.read_csv(RAW_PRODUCTS)
    orders = clean_orders(pd.read_csv(RAW_ORDERS), products)
    write_outputs(customers, products, orders)
    print(f"Wrote clean CSVs to {CLEAN_DIR} and SQLite database to {DATABASE}")
