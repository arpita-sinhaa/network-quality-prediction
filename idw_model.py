"""
Inverse Distance Weighting (IDW) model for predicting network channel
quality (CQ) at any campus location, for a given service provider.

Input: known points as (a, b, rating) for one provider.
Output: predicted rating at any query point (a, b).
"""

import pandas as pd
import numpy as np


def idw_predict(query_point, known_points, p=2):
    """
    Predict a rating at query_point using Inverse Distance Weighting.

    query_point : tuple (a, b)
    known_points : list of tuples (a, b, rating)
    p : power parameter (default 2 = inverse-square weighting)

    Returns the predicted rating (float).
    """
    qa, qb = query_point
    weighted_sum = 0.0
    weight_total = 0.0

    for a, b, rating in known_points:
        dist = np.sqrt((qa - a) ** 2 + (qb - b) ** 2)

        if dist == 0:
            # Query point exactly matches a known point -> return it directly,
            # otherwise weight = 1/0 blows up.
            return rating

        weight = 1 / (dist ** p)
        weighted_sum += weight * rating
        weight_total += weight

    return weighted_sum / weight_total


def leave_one_out_test(points_df, p=2):
    """
    Sanity check: for each known point, hide it, predict its rating using
    the other known points, and compare to its real average rating.
    points_df must have columns: a, b, avg_rating
    """
    results = []
    points = list(points_df[['a', 'b', 'avg_rating']].itertuples(index=False, name=None))

    for i, (a, b, actual) in enumerate(points):
        other_points = points[:i] + points[i + 1:]
        predicted = idw_predict((a, b), other_points, p=p)
        error = predicted - actual
        results.append({
            'a': a, 'b': b,
            'actual_rating': round(actual, 2),
            'predicted_rating': round(predicted, 2),
            'error': round(error, 2)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    xls = pd.ExcelFile('/mnt/user-data/uploads/Final.xlsx')
    summary = pd.read_excel(xls, sheet_name='Provider_Summary')

    print("=" * 70)
    print("SANITY CHECK 1: Leave-one-out test (p=2), per provider")
    print("Predict each known location's rating using the OTHER 5 locations.")
    print("=" * 70)

    for provider in summary['Provider'].unique():
        prov_df = summary[summary['Provider'] == provider].merge(
            summary[summary['Provider'] == provider][['a', 'b']].assign(
                Location=summary[summary['Provider'] == provider]['Location'].values
            ),
            on=['a', 'b'], how='left'
        )
        prov_df = summary[summary['Provider'] == provider].reset_index(drop=True)
        loo = leave_one_out_test(prov_df, p=2)
        loo.insert(0, 'Location', prov_df['Location'].values)
        loo.insert(0, 'Provider', provider)
        print(f"\n--- {provider} ---")
        print(loo.to_string(index=False))
        mae = loo['error'].abs().mean()
        print(f"Mean Absolute Error ({provider}, p=2): {mae:.3f}")

    print("\n" + "=" * 70)
    print("SANITY CHECK 2: Predict at a brand-new, never-rated point")
    print("Query point (6, 7) -- roughly between Library (6,8) and Labs (6,6)")
    print("=" * 70)

    for provider in summary['Provider'].unique():
        prov_df = summary[summary['Provider'] == provider]
        known = list(prov_df[['a', 'b', 'avg_rating']].itertuples(index=False, name=None))
        pred = idw_predict((6, 7), known, p=2)
        print(f"{provider}: predicted rating at (6,7) = {pred:.2f}")

    print("\n" + "=" * 70)
    print("SANITY CHECK 3: Effect of different p values on the same query point")
    print("Query point (6, 7), Provider = Jio")
    print("=" * 70)

    jio_known = list(summary[summary['Provider'] == 'Jio'][['a', 'b', 'avg_rating']]
                      .itertuples(index=False, name=None))
    for p_val in [1, 2, 3, 4]:
        pred = idw_predict((6, 7), jio_known, p=p_val)
        print(f"p={p_val}: predicted rating = {pred:.3f}")
