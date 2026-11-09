"""
Random Forest Regression baseline for the Network Quality Prediction project.

Task
    Predict the average perceived network rating (1-5) of a (provider, campus
    location) pair from the location's grid coordinates (a, b) and the provider.

Data
    "Provider_Summary" sheet of the project workbook: one row per (provider,
    location), i.e. 18 rows (3 providers x 6 locations). Same table and same 18
    targets as the IDW and KNN evaluations.

Preprocessing
    a, b are passed through unscaled (trees split on thresholds, so monotonic
    rescaling changes nothing). Provider is one-hot encoded inside the pipeline.

Validation
    Leave-one-location-out, identical to knn_regression.py: all three providers'
    rows for one location are hidden together, so the held-out (a, b) never
    appears in training. Six folds give one out-of-fold prediction per
    observation.

Hyperparameter tuning (n_estimators, max_depth, min_samples_leaf)
    Nested CV: for each outer fold, the configuration is chosen by an inner
    leave-one-location-out search over the five training locations only, then
    refitted on those locations and used once on the held-out location. This is
    the only number that is not tuned on the points it is scored on.
    Scoring the whole grid on all 18 points and keeping the best configuration is
    also reported, but labelled as tuned on the same data (optimistic).

Randomness
    Forests are stochastic. random_state is fixed, and stability across other
    seeds is reported so a result is not one lucky seed.

Run
    python random_forest.py --data Network_Prediction_Data.xlsx
    (keep knn_regression.py and idw_model.py in the same folder; their data
    loading, IDW and KNN evaluation are reused so the comparison is like for like)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, LeaveOneOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import knn_regression as base
except ImportError as exc:
    raise SystemExit("random_forest.py reuses the data loading, IDW and KNN evaluation from "
                     "knn_regression.py. Keep both files in the same folder.") from exc

COORDS, PROVIDER, LOCATION, TARGET = base.COORDS, base.PROVIDER, base.LOCATION, base.TARGET
metrics = base.metrics

SEED = 42
FIXED_SEEDS = range(10)
# Simplest options first: GridSearchCV breaks exact ties in favour of the earliest
# candidate, and ParameterGrid varies max_depth slowest, n_estimators fastest.
PARAM_GRID = {
    "rf__max_depth": [1, 2, 3, None],
    "rf__min_samples_leaf": [5, 3, 2, 1],
    "rf__n_estimators": [100, 300],
}
DEFAULT_PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}


# ---------------------------------------------------------------- model

def make_rf(seed=SEED, use_provider=True, **params):
    """One-hot encoding lives inside the pipeline, so it is fitted on the training
    fold only. max_features keeps scikit-learn's default (all features per split)."""
    parts = [("coords", "passthrough", COORDS)]
    if use_provider:
        parts.append(("provider", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [PROVIDER]))
    return Pipeline([
        ("prep", ColumnTransformer(parts)),
        ("rf", RandomForestRegressor(random_state=seed, n_jobs=1, **params)),
    ])


def short_params(search_params):
    return {key.split("__")[1]: value for key, value in search_params.items()}


def readable(params):
    """Printable copy of a parameter dict (max_depth=None stays readable in tables and CSVs)."""
    return {key: ("None" if value is None else value) for key, value in params.items()}


def lolo_predict(unit, model):
    """Out-of-fold predictions with whole locations held out."""
    pred = pd.Series(index=unit.index, dtype=float)
    for tr, te in LeaveOneGroupOut().split(unit, groups=unit[LOCATION]):
        model.fit(unit.iloc[tr], unit[TARGET].iloc[tr])
        pred.iloc[te] = model.predict(unit.iloc[te])
    return pred


# ---------------------------------------------------------------- evaluation

def rf_grid_lolo(df, seed=SEED):
    """Whole grid scored on all 18 points. The best row is selected on the data it
    is scored on, so it is exploratory, not an honest estimate."""
    rows = []
    for config_id, params in enumerate(ParameterGrid(PARAM_GRID)):
        chosen = short_params(params)
        rows.append({"config_id": config_id, **readable(chosen),
                     **metrics(df[TARGET], lolo_predict(df, make_rf(seed, **chosen)))})
    return pd.DataFrame(rows)


def rf_nested(df, seed=SEED, n_jobs=1):
    """Outer loop: hold out one location. Inner loop: choose the configuration by
    leave-one-location-out on the remaining locations only."""
    preds, chosen = [], []
    for held_out in df[LOCATION].unique():
        train, test = df[df[LOCATION] != held_out], df[df[LOCATION] == held_out]
        search = GridSearchCV(
            make_rf(seed),
            PARAM_GRID,
            cv=LeaveOneGroupOut(),
            scoring="neg_mean_absolute_error",
            error_score="raise",
            n_jobs=n_jobs,
        )
        search.fit(train, train[TARGET], groups=train[LOCATION])
        preds.append(pd.Series(search.predict(test), index=test.index))
        chosen.append({"seed": seed, "held_out_location": held_out, **readable(short_params(search.best_params_)),
                       "inner_MAE": -search.best_score_})
    return pd.concat(preds).loc[df.index], pd.DataFrame(chosen)


def rf_default_per_provider(df, seed=SEED):
    """Information-equivalent diagnostic: one forest per provider, coordinates only
    (the same information IDW uses). Fixed default settings, no tuning."""
    return pd.concat([lolo_predict(g, make_rf(seed, use_provider=False, **DEFAULT_PARAMS))
                      for g in base.units(df, True)]).loc[df.index]


def leakage_demo(df, seed=SEED):
    """Leave-one-ROW-out keeps the same (a, b) in training under the other two
    providers. Shown only to quantify how misleading that split is."""
    pred = pd.Series(index=df.index, dtype=float)
    for tr, te in LeaveOneOut().split(df):
        pred.iloc[te] = make_rf(seed, **DEFAULT_PARAMS).fit(df.iloc[tr], df[TARGET].iloc[tr]).predict(df.iloc[te])
    return metrics(df[TARGET], pred)


# ---------------------------------------------------------------- reference models (existing code)

def reference_models(df):
    """IDW and KNN predictions for the same 18 observations, computed with the
    functions already in the project (same protocol, nothing re-implemented)."""
    idw1, idw2, idw3 = (base.idw_fixed(df, p) for p in (1, 2, 3))
    idw_nested, _ = base.idw_nested(df)
    curve = base.knn_k_curve(df)
    best_k = int(curve.loc[curve["MAE"].idxmin(), "k"])
    knn_nested, _ = base.knn_nested(df)
    knn_pp_nested, _ = base.knn_nested(df, per_provider=True)
    knn_dist5 = pd.concat([lolo_predict(g, base.make_pipeline(k=5, use_provider=False, weights="distance"))
                           for g in base.units(df, True)]).loc[df.index]
    knn_best = lolo_predict(df, base.make_pipeline(k=best_k))
    return {
        "Provider mean (no spatial info)": (base.mean_baseline(df), "untuned"),
        "IDW p=1, k=5": (idw1, "untuned"),
        "IDW p=2, k=5": (idw2, "untuned"),
        "IDW p=3, k=5": (idw3, "tuned on same data"),
        "IDW, p tuned in nested CV": (idw_nested, "nested CV"),
        "KNN per provider, distance-weighted, k=5": (knn_dist5, "untuned"),
        f"KNN combined, uniform, k={best_k}": (knn_best, "tuned on same data"),
        "KNN combined, uniform, k tuned in nested CV": (knn_nested, "nested CV"),
        "KNN per provider, uniform, k tuned in nested CV": (knn_pp_nested, "nested CV"),
    }


def check_against_saved(summary, path):
    """Recomputed IDW/KNN rows should equal the ones saved by knn_regression.py."""
    if not os.path.exists(path):
        print(f"(no saved KNN summary at {path}; consistency check skipped)")
        return
    saved = pd.read_csv(path).set_index("Model")
    shared = [m for m in summary["Model"] if m in saved.index]
    mine = summary.set_index("Model").loc[shared]
    gap = max(float((mine["MAE"] - saved.loc[shared, "MAE"]).abs().max()),
              float((mine["RMSE"] - saved.loc[shared, "RMSE"]).abs().max()))
    status = "consistent" if gap < 1e-9 else "MISMATCH, check data file and code versions"
    print(f"Recomputed IDW/KNN rows vs {path}: {len(shared)} models, max difference {gap:.1e} ({status})")


# ---------------------------------------------------------------- reporting

def show(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def table(frame):
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    ap = argparse.ArgumentParser(description="Random Forest regression baseline")
    ap.add_argument("--data", help="Path to the project workbook (.xlsx)")
    ap.add_argument("--out", default="rf_results", help="Folder for the saved result tables")
    ap.add_argument("--knn-summary", default=os.path.join("knn_results", "knn_vs_idw_summary.csv"),
                    help="Saved KNN summary used for a consistency check")
    ap.add_argument("--nested-seeds", type=int, default=2,
                    help="Extra seeds for the nested-CV stability check (0 to skip)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for the inner search")
    args = ap.parse_args()
    start = time.time()

    df = base.load_summary(base.find_data_file(args.data))
    y = df[TARGET]
    n_configs = len(ParameterGrid(PARAM_GRID))

    show("DATA AND SETUP")
    features = make_rf().fit(df, y).named_steps["prep"].get_feature_names_out().tolist()
    print(f"{len(df)} rows = {df[PROVIDER].nunique()} providers x {df[LOCATION].nunique()} locations "
          f"(one average rating per pair)")
    print(f"Target: {TARGET} | model inputs after preprocessing: {features}")
    grid_text = ", ".join(f"{key.split('__')[1]} in {values}" for key, values in PARAM_GRID.items())
    print(f"Grid: {n_configs} configurations ({grid_text})")
    print(f"Nested CV: 6 outer folds x {n_configs} configurations x 5 inner folds | seed {SEED}")

    show("1. REFERENCE MODELS FROM THE EXISTING CODE (same 18 held-out targets)")
    references = reference_models(df)
    idw2, idw3 = references["IDW p=2, k=5"][0], references["IDW p=3, k=5"][0]
    print(f"IDW p=2, k=5: MAE {metrics(y, idw2)['MAE']:.4f} (Person B baseline: 0.5941)")
    print(f"IDW p=3, k=5: MAE {metrics(y, idw3)['MAE']:.4f} (Person B best:     0.5781)")
    if abs(metrics(y, idw2)["MAE"] - 0.5941) > 1e-3 or abs(metrics(y, idw3)["MAE"] - 0.5781) > 1e-3:
        print("WARNING: IDW numbers do not match the tuning script; check the data file.")
    else:
        print("Reproduced: evaluation set and protocol match the IDW tuning script.")

    show("2. RANDOM FOREST, DEFAULT SETTINGS (untuned; 100 trees, no depth limit, leaf size 1)")
    default_pred = lolo_predict(df, make_rf(SEED, **DEFAULT_PARAMS))
    print(f"seed {SEED}: {metrics(y, default_pred)}")
    stability = pd.DataFrame([{"setting": "RF default, untuned", "seed": s,
                               **metrics(y, lolo_predict(df, make_rf(s, **DEFAULT_PARAMS)))} for s in FIXED_SEEDS])
    print(f"across seeds {FIXED_SEEDS.start}-{FIXED_SEEDS.stop - 1}: "
          f"MAE {stability['MAE'].mean():.4f} +/- {stability['MAE'].std():.4f} "
          f"(min {stability['MAE'].min():.4f}, max {stability['MAE'].max():.4f})")

    show("3. WHOLE GRID SCORED ON ALL 18 POINTS (tuned on the same data: exploratory, optimistic)")
    grid = rf_grid_lolo(df)
    ranked = grid.sort_values("MAE", kind="stable").reset_index(drop=True)
    table(ranked.head(8))
    best_params = short_params(list(ParameterGrid(PARAM_GRID))[int(ranked.iloc[0]["config_id"])])
    print(f"Best grid configuration: {readable(best_params)}  (MAE spread over the whole grid: "
          f"{grid['MAE'].min():.4f} to {grid['MAE'].max():.4f})")
    grid_pred = lolo_predict(df, make_rf(SEED, **best_params))

    show("4. RANDOM FOREST, TUNED INSIDE THE VALIDATION LOOP (nested CV, the honest estimate)")
    nested_pred, chosen = rf_nested(df, SEED, args.n_jobs)
    table(chosen)
    print(metrics(y, nested_pred))
    runs = [chosen]
    seed_rows = [{"setting": "RF nested CV", "seed": SEED, **metrics(y, nested_pred)}]
    for s in [s for s in range(args.nested_seeds) if s != SEED]:
        pred_s, chosen_s = rf_nested(df, s, args.n_jobs)
        runs.append(chosen_s)
        seed_rows.append({"setting": "RF nested CV", "seed": s, **metrics(y, pred_s)})
    nested_stability = pd.DataFrame(seed_rows)
    if len(nested_stability) > 1:
        print("\nNested CV across seeds:")
        table(nested_stability)

    show("5. DIAGNOSTIC: ONE FOREST PER PROVIDER (same information as IDW, default settings)")
    pp_pred = rf_default_per_provider(df)
    print(metrics(y, pp_pred))

    show("6. LEAKAGE CHECK: leave-one-row-out (INVALID, shown only for contrast)")
    leaky = leakage_demo(df)
    print(f"leave-one-row-out MAE {leaky['MAE']:.4f} vs leave-one-location-out MAE "
          f"{metrics(y, default_pred)['MAE']:.4f} (same default forest, seed {SEED})")

    show("7. SIDE-BY-SIDE ON THE SAME 18 HELD-OUT OBSERVATIONS")
    models = dict(references)
    models["RF combined, default settings"] = (default_pred, "untuned")
    models["RF combined, best grid config (tuned on same 18)"] = (grid_pred, "tuned on same data")
    models["RF combined, tuned in nested CV"] = (nested_pred, "nested CV")
    models["RF per provider, default settings (diagnostic)"] = (pp_pred, "untuned")
    rows = []
    for name, (pred, evidence) in models.items():
        per_provider = {f"MAE_{p}": float(np.mean(np.abs(pred[df[PROVIDER] == p] - y[df[PROVIDER] == p])))
                        for p in sorted(df[PROVIDER].unique())}
        rows.append({"Model": name, "Evidence": evidence, **metrics(y, pred), **per_provider})
    summary = pd.DataFrame(rows)
    notes = {
        "RF combined, default settings": str(readable(DEFAULT_PARAMS)),
        "RF combined, best grid config (tuned on same 18)": str(readable(best_params)),
        "RF combined, tuned in nested CV": "chosen per outer fold, see rf_nested_selected_params.csv",
        "RF per provider, default settings (diagnostic)": str(readable(DEFAULT_PARAMS)),
    }
    summary["Config"] = summary["Model"].map(notes).fillna("")
    table(summary[["Model", "Evidence", "MAE", "RMSE"]])
    check_against_saved(summary, args.knn_summary)

    print("\nNested CV only (the like-for-like comparison to quote), best MAE first:")
    honest = summary[summary["Evidence"] == "nested CV"].sort_values("MAE")
    table(honest[["Model", "MAE", "RMSE"]])

    print("\nMAE by held-out location (nested models):")
    nested_models = {"IDW nested": "IDW, p tuned in nested CV", "KNN nested": "KNN combined, uniform, k tuned in nested CV",
                     "RF nested": "RF combined, tuned in nested CV"}
    by_location = pd.DataFrame({label: (models[name][0] - y).abs().groupby(df[LOCATION]).mean()
                                for label, name in nested_models.items()})
    print(by_location.to_string(float_format=lambda v: f"{v:.3f}"))

    rf_err = (nested_pred - y).abs()
    for label, name in (("IDW nested", nested_models["IDW nested"]), ("KNN nested", nested_models["KNN nested"])):
        other = (models[name][0] - y).abs()
        wins, losses = int((rf_err < other - 1e-9).sum()), int((rf_err > other + 1e-9).sum())
        print(f"RF nested vs {label}: RF better on {wins}, {label.split()[0]} better on {losses}, "
              f"tied on {len(df) - wins - losses} of {len(df)} observations")

    os.makedirs(args.out, exist_ok=True)
    df.assign(**{name: pred for name, (pred, _) in models.items()}).to_csv(
        os.path.join(args.out, "rf_vs_baselines_predictions.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "rf_vs_baselines_summary.csv"), index=False)
    ranked.to_csv(os.path.join(args.out, "rf_grid_exploratory.csv"), index=False)
    pd.concat(runs).to_csv(os.path.join(args.out, "rf_nested_selected_params.csv"), index=False)
    pd.concat([stability, nested_stability]).to_csv(os.path.join(args.out, "rf_seed_stability.csv"), index=False)
    print(f"\nSaved result tables to ./{args.out}/  (total time {time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
