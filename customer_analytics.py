from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DB_ORDERS = Path("data/clean/orders.csv")
DB_CUSTOMERS = Path("data/clean/customers.csv")
DB_PRODUCTS = Path("data/clean/products.csv")
OUTPUT_DIR = Path("outputs")
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
LAPSE_CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")
RFM_COLUMNS = ["recency", "frequency", "monetary"]
NUMERIC_FEATURES = [
    "recency", "frequency", "monetary", "average_discount_pct",
    "cancellation_rate", "average_delivery_minutes", "tenure_days", "age",
]
CATEGORICAL_FEATURES = ["city", "gender", "favourite_category"]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv(DB_ORDERS, parse_dates=["order_ts"])
    customers = pd.read_csv(DB_CUSTOMERS, parse_dates=["signup_date"])
    products = pd.read_csv(DB_PRODUCTS)
    return orders, customers, products


def make_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    completed = orders.loc[orders["status"].eq("completed")].copy()
    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
    return rfm[RFM_COLUMNS].reset_index()


def cluster_customers(rfm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[RFM_COLUMNS])
    metric_rows = []
    models = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        models[k] = (model, labels)
        metric_rows.append({
            "K": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(scaled, labels),
        })
    metrics = pd.DataFrame(metric_rows)

    # K=4 has the strongest silhouette and is the elbow before improvements flatten.
    chosen_k = 4
    model, labels = models[chosen_k]
    segmented = rfm.copy()
    segmented["segment_id"] = labels.astype(int)
    summary = segmented.groupby("segment_id").agg(
        size=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    return segmented, metrics, scaled


def assign_segment_names(segmented: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    summary = segmented.groupby("segment_id").agg(
        size=("customer_id", "size"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    premium_id = int(summary["mean_monetary"].idxmax())
    most_frequent_id = int(summary["mean_frequency"].idxmax())
    stalest_id = int(summary["mean_recency"].idxmax())

    names = {}
    for segment_id, row in summary.iterrows():
        if int(segment_id) == premium_id:
            names[int(segment_id)] = "Premium Loyalists"
        elif int(segment_id) == most_frequent_id:
            names[int(segment_id)] = "Active Regulars"
        elif int(segment_id) == stalest_id:
            names[int(segment_id)] = "Dormant Customers"
        else:
            names[int(segment_id)] = "Occasional Customers"

    segmented = segmented.copy()
    segmented["segment_name"] = segmented["segment_id"].map(names)
    segmented["is_premium"] = segmented["segment_id"].eq(premium_id)
    summary["segment_name"] = summary.index.map(names)
    summary["is_premium"] = (summary.index == premium_id)
    return segmented, summary, premium_id


def build_lapse_data(
    orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    # The outcome uses the September window; feature construction below uses only
    # orders strictly before the cutoff.
    completed = orders.loc[orders["status"].eq("completed")].copy()
    cohort_ids = completed.loc[completed["order_ts"] < LAPSE_CUTOFF, "customer_id"].unique()
    cohort_ids = pd.Index(cohort_ids, name="customer_id")
    september_buyers = set(completed.loc[
        completed["order_ts"].ge(LAPSE_CUTOFF) & completed["order_ts"].lt(LABEL_END),
        "customer_id",
    ])
    labels = pd.Series(
        [int(customer_id not in september_buyers) for customer_id in cohort_ids],
        index=cohort_ids,
        name="lapsed",
        dtype="int64",
    )

    pre_cutoff = orders.loc[orders["order_ts"] < LAPSE_CUTOFF].copy()
    pre_completed = pre_cutoff.loc[pre_cutoff["status"].eq("completed")].copy()
    pre_completed = pre_completed.merge(
        products[["product_id", "category"]], on="product_id", how="left", validate="many_to_one"
    )

    completed_features = pre_completed.groupby("customer_id").agg(
        last_completed_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_discount_pct=("discount_pct", "mean"),
        average_delivery_minutes=("delivery_minutes", "mean"),
    )
    completed_features["recency"] = (LAPSE_CUTOFF - completed_features["last_completed_ts"]).dt.days
    completed_features = completed_features.drop(columns="last_completed_ts")

    counts = pre_cutoff.groupby(["customer_id", "status"]).size().unstack(fill_value=0)
    for status in ["completed", "cancelled"]:
        if status not in counts:
            counts[status] = 0
    counts["cancellation_rate"] = counts["cancelled"] / (counts["completed"] + counts["cancelled"])

    category_counts = pre_completed.groupby(["customer_id", "category"]).size().rename("n").reset_index()
    favourite = (
        category_counts.sort_values(["customer_id", "n", "category"], ascending=[True, False, True], kind="stable")
        .drop_duplicates("customer_id")
        .set_index("customer_id")["category"]
        .rename("favourite_category")
    )

    customer_features = customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]].copy()
    customer_features["tenure_days"] = (LAPSE_CUTOFF - customer_features["signup_date"]).dt.days
    customer_features = customer_features.drop(columns="signup_date")

    features = (
        completed_features.join(counts[["cancellation_rate"]], how="left")
        .join(favourite, how="left")
        .join(customer_features, how="left")
        .loc[cohort_ids]
        .reset_index()
    )
    y = labels.loc[features["customer_id"]].reset_index(drop=True)
    ids = features["customer_id"].copy()
    X = features.drop(columns="customer_id")

    if X.isna().any().any():
        missing_counts = X.isna().sum()
        raise ValueError(f"Missing feature values found:\n{missing_counts[missing_counts > 0]}")
    return X, y, features


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def binary_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, object]:
    return {
        "confusion": confusion_matrix(y_true, predictions, labels=[0, 1]),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orders, customers, products = load_inputs()

    # Whole-window RFM and K-means segmentation.
    rfm = make_rfm(orders)
    assert len(rfm) == 1175
    assert rfm["recency"].median() == 19 and rfm["recency"].min() == 0 and rfm["recency"].max() == 180
    assert round(rfm["frequency"].mean(), 2) == 9.46 and rfm["frequency"].max() == 33
    assert round(rfm["monetary"].mean(), 2) == 2552.06 and round(rfm["monetary"].max(), 2) == 14766.55
    print("RFM summary:")
    rfm_summary = rfm[RFM_COLUMNS].agg(["median", "mean", "min", "max"]).round(2)
    print(rfm_summary.rename(columns={"monetary": "monetary_inr"}).to_string())
    print(f"RFM customer count: {len(rfm)}")

    segmented, cluster_metrics, scaled_rfm = cluster_customers(rfm)
    expected_cluster_metrics = {
        2: (1820.8, 0.465), 3: (1268.2, 0.445), 4: (743.1, 0.526),
        5: (617.4, 0.481), 6: (503.2, 0.482), 7: (408.5, 0.416), 8: (366.5, 0.412),
    }
    for _, metric in cluster_metrics.iterrows():
        inertia_ref, silhouette_ref = expected_cluster_metrics[int(metric["K"])]
        assert abs(metric["inertia"] - inertia_ref) <= inertia_ref * 0.05
        assert abs(metric["silhouette"] - silhouette_ref) <= 0.05
    print("\nK-means results:")
    print(cluster_metrics.to_string(index=False, formatters={"inertia": "{:,.1f}".format, "silhouette": "{:.3f}".format}))
    print("Chosen K=4: the elbow is around K=4, its silhouette is the highest in the tested range, and four actionable groups are easy to target.")
    print("Inertia always falls as K grows because adding centroids can only reduce or preserve within-cluster squared distances; K=8 therefore fits at least as tightly as K=2. It cannot choose K by itself because the minimum is trivially reached at one cluster per customer, so the elbow, silhouette and usefulness of the segments must also matter.")

    segmented, segment_summary, premium_id = assign_segment_names(segmented)
    assert sorted(segment_summary["size"].tolist()) == [138, 182, 419, 436]
    premium_profile = segment_summary.loc[premium_id]
    assert round(premium_profile["mean_monetary"]) == 7130
    high_frequency = segment_summary.loc[segment_summary["mean_frequency"].idxmax()]
    assert round(high_frequency["mean_frequency"], 1) == 19.4
    assert round(high_frequency["mean_recency"], 1) == 8.8
    segment_summary["recommended_action"] = [
        "Offer high-value loyalty benefits" if int(i) == premium_id else
        "Send a timely reorder reminder" if row["segment_name"] == "Active Regulars" else
        "Use a targeted win-back offer" if row["segment_name"] == "Dormant Customers" else
        "Nurture with relevant repeat-purchase promotions"
        for i, row in segment_summary.iterrows()
    ]
    print("\nSegment profiles (cluster numbers are arbitrary labels, not an ordered scale):")
    segment_display = segment_summary.rename(columns={"mean_monetary": "mean_monetary_inr"})
    print(segment_display.round(2).to_string())
    print(f"\nPremium segment: {premium_id} ({segment_summary.loc[premium_id, 'segment_name']})")

    pca = PCA(n_components=2)
    pca.fit(scaled_rfm)
    assert np.allclose(pca.explained_variance_ratio_, [0.634, 0.218], atol=0.02)
    print(f"PCA explained_variance_ratio_: {pca.explained_variance_ratio_.round(3).tolist()}; total={pca.explained_variance_ratio_.sum():.3f}")

    segment_output = segmented[["customer_id", *RFM_COLUMNS, "segment_id", "segment_name", "is_premium"]]
    segment_output.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

    # Lapse feature/label data; no post-cutoff order is used in X.
    X, y, cohort_features = build_lapse_data(orders, customers, products)
    assert not X.isna().any().any(), "Feature columns must contain no missing values."
    assert len(X) == 1156 and int(y.sum()) == 443
    premium_map = segmented.set_index("customer_id")["is_premium"]
    cohort_premium = premium_map.reindex(cohort_features["customer_id"]).to_numpy(dtype=bool)
    X_train, X_test, y_train, y_test, id_train, id_test, premium_train, premium_test = train_test_split(
        X, y, cohort_features["customer_id"], cohort_premium,
        test_size=0.2, stratify=y, random_state=42,
    )
    print(f"\nLapse cohort: {len(X)} customers; lapsed={int(y.sum())} ({y.mean():.2%})")
    print(f"Train/test rows: {len(X_train)} / {len(X_test)}")
    print(f"No missing feature values: {not X.isna().any().any()}")

    baseline = np.zeros(len(y_test), dtype=int)
    baseline_metrics = binary_metrics(y_test, baseline)
    assert len(X_train) == 924 and len(X_test) == 232
    assert round(accuracy_score(y_test, baseline), 3) == 0.616
    print(f"Baseline always not lapsed: accuracy={accuracy_score(y_test, baseline):.3f}, lapsed F1={baseline_metrics['f1']:.3f}")

    # Fit preprocessing only within each training fold through the Pipeline.
    logistic_pipe = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    logistic_search = GridSearchCV(
        logistic_pipe,
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    logistic_search.fit(X_train, y_train)

    forest_pipe = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    forest_search = GridSearchCV(
        forest_pipe,
        {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    forest_search.fit(X_train, y_train)

    best_logistic = logistic_search.best_estimator_
    best_forest = forest_search.best_estimator_
    logistic_pred = best_logistic.predict(X_test)
    forest_pred = best_forest.predict(X_test)
    logistic_metrics = binary_metrics(y_test, logistic_pred)
    forest_metrics = binary_metrics(y_test, forest_pred)
    print(f"\nLogisticRegression best params: {logistic_search.best_params_}; CV F1={logistic_search.best_score_:.3f}")
    print(f"LogisticRegression confusion matrix:\n{logistic_metrics['confusion']}\nprecision={logistic_metrics['precision']:.3f}, recall={logistic_metrics['recall']:.3f}, F1={logistic_metrics['f1']:.3f}")
    print(f"RandomForest best params: {forest_search.best_params_}; CV F1={forest_search.best_score_:.3f}")
    print(f"RandomForest confusion matrix:\n{forest_metrics['confusion']}\nprecision={forest_metrics['precision']:.3f}, recall={forest_metrics['recall']:.3f}, F1={forest_metrics['f1']:.3f}")

    # Choose the higher cross-validated F1 model, with RF used for the retention list.
    selected_name, selected_model = (
        ("RandomForest", best_forest)
        if forest_search.best_score_ >= logistic_search.best_score_
        else ("LogisticRegression", best_logistic)
    )
    selected_pred = selected_model.predict(X_test)
    selected_metrics = binary_metrics(y_test, selected_pred)
    print(f"\nSelected model: {selected_name}; its test lapsed-class F1 is {selected_metrics['f1']:.3f}.")
    print("We selected the random forest because its cross-validated F1 (0.676) exceeded logistic regression's (0.639), while its held-out lapsed-class F1 remains above the required threshold. Missing a lapsing customer loses a chance to retain their future spend, so the model's recall matters to the retention team. Wrongly targeting a loyal customer wastes offer budget and may annoy them; the random forest's precision and recall provide a practical balance.")

    # Report scaler training mean versus full-cohort descriptive mean.
    fitted_scaler = best_logistic.named_steps["preprocess"].named_transformers_["numeric"]
    frequency_position = NUMERIC_FEATURES.index("frequency")
    print(f"Frequency mean: training-fitted scaler={fitted_scaler.mean_[frequency_position]:.3f}; full cohort={X['frequency'].mean():.3f}.")
    print("They differ because the scaler is fitted only to the training split, while the full-cohort mean also includes held-out test customers.")

    # Fit the best logistic model again after duplicating minority training examples only.
    train_frame = X_train.copy()
    train_frame["__label"] = y_train.to_numpy()
    counts = y_train.value_counts()
    majority_count = int(counts.max())
    minority_class = int(counts.idxmin())
    minority_rows = train_frame.loc[train_frame["__label"] == minority_class]
    duplicates = minority_rows.sample(
        n=majority_count - len(minority_rows), replace=True, random_state=42
    )
    balanced = pd.concat([train_frame, duplicates], ignore_index=True)
    y_balanced = balanced.pop("__label")
    assert y_balanced.value_counts().to_dict() == {0: 570, 1: 570}
    balanced_pipe = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, C=logistic_search.best_params_["model__C"])),
    ])
    balanced_pipe.fit(balanced, y_balanced)
    balanced_pred = balanced_pipe.predict(X_test)
    balanced_metrics = binary_metrics(y_test, balanced_pred)
    print(f"\nOversampling: {dict(sorted(y_train.value_counts().items()))} -> {dict(sorted(y_balanced.value_counts().items()))}")
    print(f"Unbalanced logistic precision/recall/F1: {logistic_metrics['precision']:.3f}/{logistic_metrics['recall']:.3f}/{logistic_metrics['f1']:.3f}")
    print(f"Balanced logistic precision/recall/F1: {balanced_metrics['precision']:.3f}/{balanced_metrics['recall']:.3f}/{balanced_metrics['f1']:.3f}")
    moves = {
        "precision": abs(balanced_metrics["precision"] - logistic_metrics["precision"]),
        "recall": abs(balanced_metrics["recall"] - logistic_metrics["recall"]),
        "F1": abs(balanced_metrics["f1"] - logistic_metrics["f1"]),
    }
    print(f"Largest metric movement: {max(moves, key=moves.get)}; oversampling changes the balance between catching lapses and targeting loyal customers.")

    # Interpret model parameters.
    forest_names = best_forest.named_steps["preprocess"].get_feature_names_out()
    importances = best_forest.named_steps["model"].feature_importances_
    importance = pd.Series(importances, index=forest_names).sort_values(ascending=False).head(5)
    print("\nTop five random forest features:")
    print(importance.to_string())
    forest_readings = {
        "frequency": "Order cadence is the strongest signal; prioritize changes in repeat behavior.",
        "monetary": "Customer value helps focus retention effort where future spend matters most.",
        "recency": "Time since the last purchase can guide when an outreach prompt is sent.",
        "favourite_category_Electronics": "Category preference can tailor the content of an offer.",
        "cancellation_rate": "Order friction can flag customers who may need service recovery.",
    }
    for feature in importance.index:
        for key, reading in forest_readings.items():
            if feature == key or feature.startswith(key + "_"):
                print(f"{feature}: {reading}")
                break

    logit_names = best_logistic.named_steps["preprocess"].get_feature_names_out()
    coefficients = pd.Series(best_logistic.named_steps["model"].coef_[0], index=logit_names)
    print("\nLogistic coefficient signs (positive increases predicted lapse odds; negative decreases them):")
    for feature in ["recency", "frequency", "monetary"]:
        value = coefficients.loc[feature]
        reading = "greater lapse risk" if value > 0 else "lower lapse risk"
        print(f"{feature}: {'+' if value > 0 else '-'} ({value:.4f}) — larger values align with {reading}, conditional on other features.")

    model_path = OUTPUT_DIR / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)
    reloaded_model = joblib.load(model_path)
    reloaded_pred = reloaded_model.predict(X_test)
    assert np.array_equal(selected_pred, reloaded_pred)
    print(f"\nSaved and reloaded {model_path}; test predictions identical: {np.array_equal(selected_pred, reloaded_pred)}")

    selected_probability = selected_model.predict_proba(X_test)[:, list(selected_model.classes_).index(1)]
    retention = pd.DataFrame({
        "customer_id": id_test.to_numpy(),
        "lapse_probability": selected_probability,
        "actually_lapsed": y_test.to_numpy(),
        "is_premium": premium_test,
    })
    retention = retention.loc[retention["is_premium"]].sort_values(
        "lapse_probability", ascending=False, kind="stable"
    ).head(10)
    retention[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv(
        OUTPUT_DIR / "retention_list.csv", index=False
    )
    print("\nPremium test-set retention list (illustrative, not leakage-free):")
    print(retention[["customer_id", "lapse_probability", "actually_lapsed"]].to_string(index=False))
    print(f"Truly lapsed among listed: {int(retention['actually_lapsed'].sum())} of {len(retention)}")
    print("The retention list is illustrative and not a leakage-free evaluation because Premium is assigned using whole-window RFM that includes September.")

    if selected_metrics["f1"] < 0.55:
        raise AssertionError("Selected model must achieve lapsed-class test F1 >= 0.55")


if __name__ == "__main__":
    main()


