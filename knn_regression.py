"""
KNN Regression baseline for the Network Quality Prediction project.

Task
    Predict the average perceived network rating (1-5) of a (provider, campus
    location) pair from the location's grid coordinates (a, b) and the provider.

Data
    "Provider_Summary" sheet of the project workbook: one row per (provider,
    location), i.e. 18 rows (3 providers x 6 locations). IDW is evaluated on
    this same table, so both models predict the same 18 targets.

Validation
    Leave-one-location-out. All rows of one location are hidden together, the
    model is trained on the other five locations and the hidden rows are
    predicted. Repeating this for the six locations gives one out-of-fold
    prediction for each of the 18 observations. Holding out a single row instead
    would leave the same (a, b) in the training set under the other two
    providers, which leaks the answer (see leakage_demo).

Hyperparameter tuning
    k is chosen by an inner leave-one-location-out search that only sees the
    training locations of each outer fold (nested CV), so the reported error is
    not tuned on the points it is measured on.

Run
    python knn_regression.py --data Network_Prediction_Data.xlsx
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

try:
    from idw_model import idw_predict
except ImportError:
    def idw_predict(query_point, known_points, p=2):
        qa, qb = query_point
        weighted_sum, weight_total = 0.0, 0.0
        for a, b, rating in known_points:
            dist = np.sqrt((qa - a) ** 2 + (qb - b) ** 2)
            if dist == 0:
                return rating
            weight = 1 / (dist ** p)
            weighted_sum += weight * rating
            weight_total += weight
        return weighted_sum / weight_total

SHEET = "Provider_Summary"
COORDS = ["a", "b"]
PROVIDER = "Provider"
LOCATION = "Location"
TARGET = "avg_rating"
IDW_P_GRID = (1, 2, 3)
DATA_CANDIDATES = ("Network_Prediction_Data.xlsx", "Final.xlsx")


# ---------------------------------------------------------------- data

def find_data_file(cli_path):
    if cli_path:
        return cli_path
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (os.getcwd(), here):
        for name in DATA_CANDIDATES:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f"Data file not found; pass --data (looked for {DATA_CANDIDATES}).")


def load_summary(path):
    df = pd.read_excel(path, sheet_name=SHEET)
    needed = COORDS + [PROVIDER, LOCATION, TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {SHEET}: {missing}")
    if df[needed].isna().any().any():
        raise ValueError("Missing values in the modelling columns.")
    if df.duplicated([PROVIDER, LOCATION]).any():
        raise ValueError("Expected exactly one row per (provider, location).")
    if not df[TARGET].between(1, 5).all():
        raise ValueError("Average ratings must lie in [1, 5].")
    return df[needed].reset_index(drop=True)


# ---------------------------------------------------------------- model

def make_pipeline(k=5, scaler="standard", use_provider=True, weights="uniform"):
    """Scaling and one-hot encoding live inside the pipeline, so they are fitted
    on the training fold only (no leakage from the held-out location)."""
    scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "none": "passthrough"}
    parts = [("coords", scalers[scaler], COORDS)]
    if use_provider:
        parts.append(("provider", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [PROVIDER]))
    return Pipeline([
        ("prep", ColumnTransformer(parts)),
        ("knn", KNeighborsRegressor(n_neighbors=k, weights=weights)),
    ])


def metrics(y_true, y_pred):
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def units(df, per_provider):
    """Combined model: one unit holding all rows. Per-provider model: one unit
    per provider (the information set IDW uses)."""
    return [g for _, g in df.groupby(PROVIDER)] if per_provider else [df]


def max_k(unit):
    """Largest k that is valid when one whole location is held out."""
    return len(unit) - unit.groupby(LOCATION).size().max()


# ---------------------------------------------------------------- KNN evaluation

def knn_k_curve(df, per_provider=False, **cfg):
    """MAE/RMSE for every k under leave-one-location-out. k is selected on the
    same points it is scored on, so this is for analysis, not an honest estimate."""
    cfg = dict(cfg, use_provider=not per_provider)
    rows = []
    for k in range(1, min(max_k(u) for u in units(df, per_provider)) + 1):
        pred = pd.concat([_lolo_predict(u, make_pipeline(k=k, **cfg)) for u in units(df, per_provider)])
        rows.append({"k": k, **metrics(df[TARGET], pred.loc[df.index])})
    return pd.DataFrame(rows)


def knn_nested(df, per_provider=False, **cfg):
    """Outer loop: hold out one location. Inner loop: choose k by
    leave-one-location-out on the remaining locations only."""
    cfg = dict(cfg, use_provider=not per_provider)
    preds, chosen = [], {}
    for unit in units(df, per_provider):
        for held_out in unit[LOCATION].unique():
            train, test = unit[unit[LOCATION] != held_out], unit[unit[LOCATION] == held_out]
            search = GridSearchCV(
                make_pipeline(**cfg),
                {"knn__n_neighbors": list(range(1, max_k(train) + 1))},
                cv=LeaveOneGroupOut(),
                scoring="neg_mean_absolute_error",
                error_score="raise",
            )
            search.fit(train, train[TARGET], groups=train[LOCATION])
            preds.append(pd.Series(search.predict(test), index=test.index))
            label = f"{held_out}" if not per_provider else f"{unit[PROVIDER].iloc[0]}/{held_out}"
            chosen[label] = search.best_params_["knn__n_neighbors"]
    return pd.concat(preds).loc[df.index], chosen


def _lolo_predict(unit, model):
    pred = pd.Series(index=unit.index, dtype=float)
    for tr, te in LeaveOneGroupOut().split(unit, groups=unit[LOCATION]):
        model.fit(unit.iloc[tr], unit[TARGET].iloc[tr])
        pred.iloc[te] = model.predict(unit.iloc[te])
    return pred


def leakage_demo(df, k_values):
    """Leave-one-ROW-out in the combined model keeps the same (a, b) in training
    under the other providers. Shown only to quantify how misleading it is."""
    rows = []
    for k in k_values:
        pred = pd.Series(index=df.index, dtype=float)
        for tr, te in LeaveOneOut().split(df):
            pred.iloc[te] = make_pipeline(k=k).fit(df.iloc[tr], df[TARGET].iloc[tr]).predict(df.iloc[te])
        rows.append({"k": k, **metrics(df[TARGET], pred)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- IDW on the same 18 targets

def idw_fixed(df, p):
    """Per-provider leave-one-location-out IDW (all other locations as neighbours)."""
    pred = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby(PROVIDER):
        for idx in g.index:
            others = g.drop(index=idx)
            pred[idx] = idw_predict((g.at[idx, "a"], g.at[idx, "b"]),
                                    list(others[COORDS + [TARGET]].itertuples(index=False, name=None)), p=p)
    return pred


def idw_nested(df):
    """Same outer split as KNN; p is chosen on the training locations only."""
    preds, chosen = [], {}
    for held_out in df[LOCATION].unique():
        train, test = df[df[LOCATION] != held_out], df[df[LOCATION] == held_out]
        inner = {p: float(np.mean(np.abs(idw_fixed(train, p) - train[TARGET]))) for p in IDW_P_GRID}
        best_p = min(IDW_P_GRID, key=inner.get)
        out = pd.Series(index=test.index, dtype=float)
        for idx in test.index:
            known = train[train[PROVIDER] == test.at[idx, PROVIDER]]
            out[idx] = idw_predict((test.at[idx, "a"], test.at[idx, "b"]),
                                   list(known[COORDS + [TARGET]].itertuples(index=False, name=None)), p=best_p)
        preds.append(out)
        chosen[held_out] = best_p
    return pd.concat(preds).loc[df.index], chosen


def mean_baseline(df):
    """Predict the mean rating of the provider's other locations (no spatial information)."""
    pred = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby(PROVIDER):
        for idx in g.index:
            pred[idx] = g.drop(index=idx)[TARGET].mean()
    return pred


# ---------------------------------------------------------------- reporting

def show(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def table(frame):
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    ap = argparse.ArgumentParser(description="KNN regression baseline")
    ap.add_argument("--data", help="Path to the project workbook (.xlsx)")
    ap.add_argument("--out", default="knn_results", help="Folder for the saved result tables")
    args = ap.parse_args()

    df = load_summary(find_data_file(args.data))
    y = df[TARGET]
    show("DATA")
    print(f"{len(df)} rows = {df[PROVIDER].nunique()} providers x {df[LOCATION].nunique()} locations "
          f"(one average rating per pair)")
    print(f"Target: {TARGET} | Features: {COORDS} (scaled) + {PROVIDER} (one-hot)")

    show("1. IDW REFERENCE (existing IDW function, same 18 held-out targets)")
    idw2, idw3 = idw_fixed(df, 2), idw_fixed(df, 3)
    idw1 = idw_fixed(df, 1)
    print(f"p=1, k=5 : MAE {metrics(y, idw1)['MAE']:.4f}")
    print(f"p=2, k=5 : MAE {metrics(y, idw2)['MAE']:.4f}   (Person B baseline: 0.5941)")
    print(f"p=3, k=5 : MAE {metrics(y, idw3)['MAE']:.4f}   (Person B best:     0.5781)")
    if abs(metrics(y, idw2)["MAE"] - 0.5941) > 1e-3 or abs(metrics(y, idw3)["MAE"] - 0.5781) > 1e-3:
        print("WARNING: IDW numbers do not match the tuning script; check the data file.")
    else:
        print("Reproduced: evaluation set and protocol match the IDW tuning script.")

    show("2. KNN k-CURVE, combined model (leave-one-location-out; for analysis only)")
    curve = knn_k_curve(df)
    table(curve)
    best_k = int(curve.loc[curve["MAE"].idxmin(), "k"])
    print(f"Lowest MAE at k = {best_k}")

    show("3. KNN, k TUNED INSIDE THE VALIDATION LOOP (nested CV, the honest estimate)")
    knn_pred, knn_k = knn_nested(df)
    print("k chosen in each outer fold:", knn_k)
    print(metrics(y, knn_pred))

    show("4. KNN PER PROVIDER (same information set as IDW; provider one-hot is constant, so dropped)")
    pp_curve = knn_k_curve(df, per_provider=True)
    table(pp_curve)
    pp_pred, pp_k = knn_nested(df, per_provider=True)
    print("k chosen in each outer fold:", pp_k)
    print(metrics(y, pp_pred))

    show("5. IDW WITH p CHOSEN INSIDE THE SAME NESTED LOOP (fair counterpart to section 3)")
    idw_n_pred, idw_n_p = idw_nested(df)
    print("p chosen in each outer fold:", idw_n_p)
    print(metrics(y, idw_n_pred))

    show("6. WEIGHTING: uniform vs distance-weighted KNN (per provider, fixed k, no selection)")
    uniform_curve = knn_k_curve(df, per_provider=True)
    distance_curve = knn_k_curve(df, per_provider=True, weights="distance")
    table(uniform_curve.merge(distance_curve, on="k", suffixes=("_uniform", "_distance")))
    print(f"IDW p=1, k=5 for reference: MAE {metrics(y, idw1)['MAE']:.4f}, RMSE {metrics(y, idw1)['RMSE']:.4f} "
          "(distance-weighted KNN is IDW with p=1 on k neighbours)")

    show("7. SCALING CHOICE (combined KNN, nested CV)")
    rows = []
    for scaler in ("none", "standard", "minmax"):
        pred, _ = knn_nested(df, scaler=scaler)
        rows.append({"scaler": scaler, **metrics(y, pred)})
    table(pd.DataFrame(rows))

    show("8. LEAKAGE CHECK: leave-one-row-out (INVALID, shown only for contrast)")
    leaky = leakage_demo(df, k_values=range(1, 7))
    table(leaky)
    print(f"Best leaky MAE {leaky['MAE'].min():.4f} vs best leave-one-location-out MAE "
          f"{curve['MAE'].min():.4f}")

    show("9. SIDE-BY-SIDE ON THE SAME 18 HELD-OUT OBSERVATIONS")
    knn_dist5 = pd.concat([_lolo_predict(u, make_pipeline(k=5, use_provider=False, weights="distance"))
                           for u in units(df, True)]).loc[df.index]
    models = {
        "Provider mean (no spatial info)": (mean_baseline(df), "fixed"),
        "IDW p=1, k=5": (idw1, "fixed"),
        "IDW p=2, k=5": (idw2, "fixed (default)"),
        "IDW p=3, k=5": (idw3, "p picked on these 18 points"),
        "IDW, p tuned in nested CV": (idw_n_pred, "nested"),
        "KNN per provider, distance-weighted, k=5": (knn_dist5, "fixed"),
        f"KNN combined, uniform, k={best_k}": (_lolo_predict(df, make_pipeline(k=best_k)),
                                              "k picked on these 18 points"),
        "KNN combined, uniform, k tuned in nested CV": (knn_pred, "nested"),
        "KNN per provider, uniform, k tuned in nested CV": (pp_pred, "nested"),
    }
    summary = pd.DataFrame([{"Model": n, "Selection": s, **metrics(y, p)} for n, (p, s) in models.items()])
    table(summary)

    by_provider = pd.DataFrame({
        name: {prov: float(np.mean(np.abs(pred[df[PROVIDER] == prov] - y[df[PROVIDER] == prov])))
               for prov in sorted(df[PROVIDER].unique())}
        for name, (pred, _) in models.items()
    }).T
    print("\nMAE by provider")
    print(by_provider.to_string(float_format=lambda v: f"{v:.4f}"))

    err_knn, err_idw = np.abs(knn_pred - y), np.abs(idw_n_pred - y)
    wins, losses = int((err_knn < err_idw - 1e-9).sum()), int((err_knn > err_idw + 1e-9).sum())
    print(f"\nNested KNN vs nested IDW, per observation: KNN better on {wins}, IDW better on {losses}, "
          f"tied on {len(df) - wins - losses} of {len(df)}")

    os.makedirs(args.out, exist_ok=True)
    detail = df.assign(**{n: p for n, (p, _) in models.items()})
    detail.to_csv(os.path.join(args.out, "knn_vs_idw_predictions.csv"), index=False)
    curve.to_csv(os.path.join(args.out, "knn_k_curve.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "knn_vs_idw_summary.csv"), index=False)
    print(f"\nSaved result tables to ./{args.out}/")


if __name__ == "__main__":
    main()
