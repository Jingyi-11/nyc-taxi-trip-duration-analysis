# NYC Taxi Trip Duration Modeling

An end-to-end machine learning project for predicting New York City Yellow Taxi trip duration from structured trip, time, and location features. The project emphasizes leakage-aware feature engineering, scalable preprocessing on millions of records, model benchmarking, and interpretable analysis of urban mobility patterns.

## Project Overview

The main research question is:

> Can taxi trip duration be predicted using information available before or near the start of a trip?

Trip distance is clearly useful, but it is not sufficient on its own. Trips with similar distances can have very different durations depending on pickup time, borough flow, airport access, and other spatial-temporal conditions. This project combines data cleaning, feature engineering, regression modeling, and supplementary permutation-based hypothesis testing.

## Dataset

- Source: NYC Taxi and Limousine Commission (TLC) Yellow Taxi Trip Records
- Month: January 2026
- Raw size: 3,724,889 trips
- Modeling set: about 2.5M trips after filtering, feature selection, and missing-value handling
- Auxiliary data: TLC taxi zone lookup table for borough and service-zone decoding

The raw parquet file is not included in this repository because of size. Download it from the NYC TLC trip record data portal and place it in the project root as:

```text
yellow_tripdata_2026-01.parquet
```

## What This Repository Shows

- A reproducible Python pipeline in `src/pipeline.py`
- A cleaned exploratory notebook in `notebooks/`
- Data documentation and a taxi-zone lookup table in `data/`
- A concise project summary in `docs/`

## Pipeline

1. Load TLC Yellow Taxi trip records.
2. Create `trip_duration_min` from pickup and dropoff timestamps.
3. Remove invalid records and conservative extreme outliers.
4. Decode pickup and dropoff location IDs using the TLC zone lookup table.
5. Build leakage-aware features available before or near trip start.
6. Compare interpretable linear baselines with nonlinear ensemble models.
7. Evaluate models using RMSE, MAE, and R-squared.
8. Run permutation tests for supplementary tipping and timing analysis.

## Feature Engineering

The duration model excludes post-trip payment variables such as fare amount, tip amount, tolls, surcharges, and total amount to avoid target leakage.

Main feature groups:

- Trip context: trip distance, passenger count, vendor ID, rate code
- Time features: pickup hour, day of week, weekend indicator
- Location features: pickup/dropoff location IDs, pickup/dropoff borough, service zone
- Derived spatial features: same-borough indicator, airport-trip indicator

## Modeling Design

The project compares models with increasing complexity:

- Distance-only linear regression baseline
- Multivariate linear regression
- Ridge regression
- Lasso regression
- Random Forest regressor

The modeling design moves from an interpretable distance-only baseline to richer structured features and nonlinear ensemble methods. This makes it possible to separate the value of distance from the additional signal provided by time and location context.

## Statistical Analysis

Permutation tests evaluate whether observed group differences could plausibly occur under random reassignment of labels. The supplementary analysis studies:

- Payment type vs. tip rate
- Weekday vs. weekend trip duration
- Daytime vs. nighttime tip rate

The strongest behavioral signal was the relationship between payment type and tip rate.

## Key Takeaways

- Distance is the strongest single baseline predictor, but it does not fully explain duration variance.
- Time and real-zone location features improve the model beyond distance alone.
- Leakage control is essential: post-trip monetary variables should not be used for realistic duration prediction.
- Ensemble models better capture nonlinear interactions in urban mobility data.
- Payment type is strongly associated with tipping behavior in the filtered taxi records.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── pipeline.py
├── data/
│   ├── README.md
│   └── taxi_zone_lookup.csv
├── docs/
│   └── project_summary.md
└── notebooks/
    └── nyc_taxi_trip_duration_analysis.ipynb
```

## How to Run

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the January 2026 Yellow Taxi parquet file from the NYC TLC portal.
4. Place it in the project root as:

```text
yellow_tripdata_2026-01.parquet
```

5. Run the Python pipeline:

```bash
python src/pipeline.py --data yellow_tripdata_2026-01.parquet --sample-size 200000
```

6. For exploratory analysis, open:

```text
notebooks/nyc_taxi_trip_duration_analysis.ipynb
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, pyarrow, permutation testing
