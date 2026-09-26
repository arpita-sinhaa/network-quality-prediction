# Network Quality Prediction

B.Tech final year research project — predicting mobile network channel quality (CQ) at different physical locations on campus, for different service providers (Jio, Airtel, Vi), using crowdsourced survey ratings and spatial interpolation.

## Problem Statement

Instead of using expensive signal-measuring equipment, we collect user-reported signal ratings (1-5) at fixed campus locations via a Google Form, then use spatial interpolation (Inverse Distance Weighting) to predict channel quality at any location on campus, for any provider — based on how close that location is to points we already have ratings for.

## Dataset

Source: Google Form survey, 87 raw responses → 86 after cleaning.

Six fixed campus locations were rated by each respondent for their mobile provider:
- Classrooms
- Labs
- Sports Ground
- Staircases
- Campus Gate/Badminton Court
- Library

Each location was manually mapped to (a, b) grid coordinates from a campus satellite image.

Final.xlsx contains:
- Master_Dataset — Long-format data: RespondentID, Location, Provider, Rating, a, b (515 rows)
- Provider_Summary — Avg rating + variance per (Provider, Location)
- Coordinate_Lookup — Location → (a, b) mapping
- Checks — Data validation counts

## Model: Inverse Distance Weighting (IDW)

IDW predicts a rating at any query point (a, b) as a distance-weighted average of known ratings — closer points contribute more:

predicted_rating = Σ(weight_i × rating_i) / Σ(weight_i)
weight_i = 1 / distance(query_point, point_i) ^ p

p is the power parameter (default p=2), controlling how sharply weight drops off with distance.

## Usage

idw_model.py

This runs 3 sanity checks against the actual dataset:
1. Leave-one-out test — hides each known location, predicts it from the rest, compares to the real average rating (per provider).
2. New-point prediction — predicts a rating at a location never in the dataset.
3. p-value sensitivity — shows how predictions change as p varies (1 to 4).

## Known Limitations

- Only 6 spatial points per provider — limited data for spatial interpolation to learn from.
- Vi has only 11 respondents vs. Jio's 46 and Airtel's 30 — Vi predictions are less statistically reliable.
- Coordinates for 5 of 6 locations are grid-estimated from a campus image, not surveyed GPS.
