"""
Stage 2 — Person B: IDW Parameter Tuning
Network Quality Prediction Project

Tunes:
1. p = 1, 2, 3
2. neighbour count k = 1, 2, 3, 4, 5
3. radius using the actual distinct pairwise grid distances

Evaluation:
Leave-one-out (LOO): hide one known location, predict it from the
remaining locations, and compute MAE.

For radius tuning, only configurations with 100% LOO coverage are used
for the final comparison.
"""

import math
import numpy as np
import pandas as pd

INPUT_FILE = "Network _Prediction_Data.xlsx"
SHEET_NAME = "Provider_Summary"


def idw_predict(query_point, known_points, p=2, k=None, radius=None):
    qa, qb = query_point
    distances = []

    for a, b, rating in known_points:
        dist = math.hypot(qa - a, qb - b)

        if dist == 0:
            return float(rating)

        distances.append((dist, float(rating)))

    if k is not None:
        distances = sorted(distances, key=lambda x: x[0])[:k]

    if radius is not None:
        # Small tolerance handles floating-point representation at a
        # radius that is exactly equal to a pairwise distance.
        distances = [
            x for x in distances
            if x[0] <= radius + 1e-9
        ]

    if not distances:
        return np.nan

    weights = np.array([1.0 / (d ** p) for d, _ in distances])
    ratings = np.array([r for _, r in distances])

    return float(np.sum(weights * ratings) / np.sum(weights))


def loo_eval(summary, p=2, k=None, radius=None):
    rows = []

    for provider, group in summary.groupby("Provider", sort=True):
        points = list(
            group[["Location", "a", "b", "avg_rating"]]
            .itertuples(index=False, name=None)
        )

        for i, (location, a, b, actual) in enumerate(points):
            other_points = [
                (oa, ob, rating)
                for _, oa, ob, rating in (points[:i] + points[i+1:])
            ]

            predicted = idw_predict(
                (a, b),
                other_points,
                p=p,
                k=k,
                radius=radius
            )

            rows.append({
                "Provider": provider,
                "Location": location,
                "Actual_Rating": actual,
                "Predicted_Rating": predicted,
                "Absolute_Error": (
                    abs(predicted - actual)
                    if not np.isnan(predicted) else np.nan
                )
            })

    return pd.DataFrame(rows)


summary = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# Baseline reproduction of Person A.
baseline = loo_eval(summary, p=2)
print("\nBASELINE — p=2, all 5 neighbours")
print(baseline.groupby("Provider")["Absolute_Error"].mean())
print("Overall MAE:", baseline["Absolute_Error"].mean())

# -------------------------------------------------
# A. Neighbour-count tuning
# -------------------------------------------------
results = []

for p in [1, 2, 3]:
    for k in [1, 2, 3, 4, 5]:
        r = loo_eval(summary, p=p, k=k)
        results.append({
            "p": p,
            "k": k,
            "Overall_MAE": r["Absolute_Error"].mean()
        })

k_results = pd.DataFrame(results).sort_values("Overall_MAE")

print("\nNEIGHBOUR-COUNT TUNING")
print(k_results.to_string(index=False))

best_k = k_results.iloc[0]

print("\nBEST NEIGHBOUR-COUNT CONFIGURATION")
print("p =", int(best_k["p"]))
print("k =", int(best_k["k"]))
print("Overall MAE =", round(best_k["Overall_MAE"], 4))

# -------------------------------------------------
# B. Radius tuning
# -------------------------------------------------
locations = summary[["Location", "a", "b"]].drop_duplicates().reset_index(drop=True)

pairwise_distances = []

for i in range(len(locations)):
    for j in range(i + 1, len(locations)):
        pairwise_distances.append(
            math.hypot(
                locations.loc[i, "a"] - locations.loc[j, "a"],
                locations.loc[i, "b"] - locations.loc[j, "b"]
            )
        )

# Keep full precision so a boundary point is not accidentally excluded.
radius_candidates = sorted(set(pairwise_distances))

radius_results = []

for p in [1, 2, 3]:
    for radius in radius_candidates:
        r = loo_eval(summary, p=p, radius=radius)
        valid = r["Absolute_Error"].notna()

        if valid.all():
            radius_results.append({
                "p": p,
                "radius": radius,
                "Overall_MAE": r["Absolute_Error"].mean()
            })

radius_results = pd.DataFrame(radius_results).sort_values("Overall_MAE")

print("\nRADIUS TUNING — FULL LOO COVERAGE ONLY")
print(radius_results.to_string(index=False))

best_radius = radius_results.iloc[0]

print("\nBEST FULL-COVERAGE RADIUS CONFIGURATION")
print("p =", int(best_radius["p"]))
print("radius =", best_radius["radius"])
print("Overall MAE =", round(best_radius["Overall_MAE"], 4))

print("\nFINAL STAGE 2 SELECTION")
print(
    "Use neighbour-count tuning: "
    f"p={int(best_k['p'])}, k={int(best_k['k'])}"
)
