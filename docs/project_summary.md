# Project Summary

## Problem

Predict NYC Yellow Taxi trip duration using trip information available before or near trip start. The goal is to build a realistic ETA-style model without relying on post-trip payment variables.

## Data

The project uses January 2026 NYC TLC Yellow Taxi trip records. The raw dataset contains 3,724,889 rows. After basic validity filtering, about 3.52M rows remain; after feature selection and missing-value handling, the duration modeling set contains about 2.5M trips.

## Cleaning

The pipeline removes invalid observations with non-positive duration, distance, fare, or total amount. It also filters extreme outliers by keeping trips with duration at most 180 minutes and distance at most 100 miles.

## Feature Engineering

Features are organized into trip context, time, and location groups:

- Trip context: distance, passenger count, vendor, rate code
- Time: pickup hour, day of week, weekend indicator
- Location: pickup/dropoff ID, borough, service zone
- Derived spatial features: same-borough and airport-trip indicators

Fare, tip, total amount, tolls, and surcharges are excluded from duration prediction to avoid data leakage.

## Modeling

The notebook compares interpretable linear baselines and nonlinear ensemble models. The baseline uses distance only. Additional models add structured temporal and location features, and Random Forest improves over the linear baselines. A natural future extension is to add gradient boosting models for richer nonlinear interactions.

## Statistical Testing

Permutation tests are used for supplementary analysis:

- Payment type and tip rate
- Weekday/weekend trip duration
- Daytime/nighttime tip rate

The payment-type comparison shows the strongest practical difference in tipping behavior.

## Practical Interpretation

The project shows that realistic trip-duration prediction should combine distance, time, and location context. Better duration estimates can support passenger ETA reliability, driver route and shift planning, dispatch operations, and transportation analysis.
