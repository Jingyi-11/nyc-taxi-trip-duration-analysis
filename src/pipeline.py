"""Leakage-aware NYC taxi trip-duration modeling pipeline.

The raw TLC parquet file is not included in this repository. Download the
January 2026 Yellow Taxi Trip Records file from the NYC TLC data portal and
pass it with --data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LEAKAGE_COLUMNS = [
    "fare_amount",
    "tip_amount",
    "total_amount",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "Airport_fee",
]


def load_trips(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    df["tip_rate"] = np.where(
        df["fare_amount"] > 0, df["tip_amount"] / df["fare_amount"], np.nan
    )
    return df


def clean_base(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["trip_duration_min"] > 0)
        & (df["trip_distance"] > 0)
        & (df["fare_amount"] > 0)
        & (df["total_amount"] > 0)
    ].copy()


def build_duration_dataset(
    base_df: pd.DataFrame,
    zone_lookup_path: Path,
    max_duration: float = 180,
    max_distance: float = 100,
) -> pd.DataFrame:
    df = base_df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in base_df.columns]).copy()
    df = df[(df["trip_duration_min"] <= max_duration) & (df["trip_distance"] <= max_distance)]

    if zone_lookup_path.exists():
        zones = pd.read_csv(zone_lookup_path)
        pickup = zones[["LocationID", "Borough", "Zone", "service_zone"]].rename(
            columns={
                "LocationID": "PULocationID",
                "Borough": "PU_Borough",
                "Zone": "PU_Zone",
                "service_zone": "PU_service_zone",
            }
        )
        dropoff = zones[["LocationID", "Borough", "Zone", "service_zone"]].rename(
            columns={
                "LocationID": "DOLocationID",
                "Borough": "DO_Borough",
                "Zone": "DO_Zone",
                "service_zone": "DO_service_zone",
            }
        )
        df = df.merge(pickup, on="PULocationID", how="left")
        df = df.merge(dropoff, on="DOLocationID", how="left")
        df["same_borough"] = (df["PU_Borough"] == df["DO_Borough"]).astype(int)
        df["airport_trip"] = (
            df["PU_service_zone"].fillna("").str.contains("Airports", case=False)
            | df["DO_service_zone"].fillna("").str.contains("Airports", case=False)
        ).astype(int)

    model_cols = [
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "pickup_dayofweek",
        "is_weekend",
        "VendorID",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "PU_Borough",
        "DO_Borough",
        "PU_service_zone",
        "same_borough",
        "airport_trip",
        "trip_duration_min",
    ]
    return df[[c for c in model_cols if c in df.columns]].dropna().copy()


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_preprocessor(x: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    categorical = [
        c
        for c in [
            "VendorID",
            "RatecodeID",
            "PULocationID",
            "DOLocationID",
            "PU_Borough",
            "DO_Borough",
            "PU_service_zone",
        ]
        if c in x.columns
    ]
    numeric = [c for c in x.columns if c not in categorical]
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def train_models(model_df: pd.DataFrame, sample_size: int | None = None) -> pd.DataFrame:
    if sample_size and len(model_df) > sample_size:
        model_df = model_df.sample(sample_size, random_state=42)

    target = "trip_duration_min"
    x = model_df.drop(columns=[target])
    y = model_df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    results: list[dict[str, float | str]] = []

    distance_model = LinearRegression()
    distance_model.fit(x_train[["trip_distance"]], y_train)
    pred = distance_model.predict(x_test[["trip_distance"]])
    results.append({"Model": "Distance-only Linear Regression", **evaluate(y_test, pred)})

    linear_preprocessor = make_preprocessor(x_train, scale_numeric=True)
    linear_models = {
        "Multivariate Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=5000),
    }
    for name, estimator in linear_models.items():
        pipeline = Pipeline([("preprocessor", linear_preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        results.append({"Model": name, **evaluate(y_test, pred)})

    rf_preprocessor = make_preprocessor(x_train, scale_numeric=False)
    rf = Pipeline(
        [
            ("preprocessor", rf_preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    )
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    results.append({"Model": "Random Forest Regressor", **evaluate(y_test, pred)})

    return pd.DataFrame(results).sort_values("RMSE")


def permutation_test_mean_diff(
    group1: np.ndarray, group2: np.ndarray, n_perm: int = 1000, seed: int = 42
) -> tuple[float, float]:
    observed = group1.mean() - group2.mean()
    pooled = np.concatenate([group1, group2])
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(pooled)
        diffs[i] = shuffled[: len(group1)].mean() - shuffled[len(group1) :].mean()
    p_value = float(np.mean(np.abs(diffs) >= abs(observed)))
    return float(observed), p_value


def run(args: argparse.Namespace) -> None:
    raw = load_trips(args.data)
    print(f"Raw shape: {raw.shape}")

    featured = create_features(raw)
    base = clean_base(featured)
    print(f"Rows after base cleaning: {len(base):,}")

    model_df = build_duration_dataset(base, args.zone_lookup)
    print(f"Modeling shape: {model_df.shape}")

    results = train_models(model_df, sample_size=args.sample_size)
    print("\nModel comparison:")
    print(results.to_string(index=False))

    tip_df = base[
        base["tip_rate"].notna()
        & (base["tip_rate"] >= 0)
        & (base["tip_rate"] <= 1.0)
        & (base["trip_duration_min"] <= 180)
        & (base["trip_distance"] <= 100)
    ].copy()
    common_types = tip_df["payment_type"].value_counts().head(2).index.tolist()
    if len(common_types) == 2:
        g1 = tip_df.loc[tip_df["payment_type"] == common_types[0], "tip_rate"].values
        g2 = tip_df.loc[tip_df["payment_type"] == common_types[1], "tip_rate"].values
        if args.sample_size:
            rng = np.random.default_rng(42)
            g1 = rng.choice(g1, size=min(len(g1), args.sample_size), replace=False)
            g2 = rng.choice(g2, size=min(len(g2), args.sample_size), replace=False)
        diff, p_value = permutation_test_mean_diff(g1, g2, n_perm=args.n_perm)
        print("\nPayment-type tip-rate permutation test:")
        print(f"Observed mean difference: {diff:.4f}")
        print(f"p-value: {p_value:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to TLC parquet file.")
    parser.add_argument(
        "--zone-lookup",
        type=Path,
        default=Path("data/taxi_zone_lookup.csv"),
        help="Path to TLC taxi zone lookup CSV.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200_000,
        help="Optional row sample for faster local modeling.",
    )
    parser.add_argument("--n-perm", type=int, default=1000, help="Permutation iterations.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
