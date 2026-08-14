#TODO: warning, wenn Werte für MSE bzw. Complexity fehlen oder folds nicht vollständig sind


from __future__ import annotations

import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

CSV_DIR = "./csv_all"

RUNNAME_COL = "tags.mlflow.runName"                 #concrete_strength__ni32__nr16__nir32__tune.seed-7.fold-30-of-30
FOLD_COL = "tags.fold"                              #TRUE/empty

MSE_COL = "metrics.test_neg_mean_squared_error"     
COMPLEXITY_COL = "metrics.elitist_complexity"   
FIT_TIME_COL = "metrics.fit_time"  

EXPECTED_FOLDS_PER_SEED = 20
EXPECTED_SEEDS = 5

group_cols = ["dataset", "label", "n_iter", "n_rules", "n_initial_rules"]

#tags.grid_point = true -> params.grid_n_inital_rules, params.grid_n_rules, params.grid_n_iter
gridpoint_pattern = re.compile(
    r"^(?P<dataset>.+?)__ni(?P<n_iter>\d+)__nr(?P<n_rules>\d+)__nir(?P<n_initial_rules>\d+)__(?P<label>.+?)\.seed-"
)


def load_from_csv() -> pd.DataFrame:
    """Liest und konkateniert alle *.csv im angegebenen Verzeichnis."""
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Keine CSV-Dateien in '{CSV_DIR}' gefunden.")

    print(f"[info] Lade {len(csv_files)} CSV-Datei(en):")
    for f in csv_files:
        print(f"       - {f}")

    df_list = [pd.read_csv(f) for f in csv_files]

    df = pd.concat(df_list, ignore_index=True)

    if "run_id" in df.columns:
        df = df.drop_duplicates(subset=["run_id"])

    #only keep relevant columns
    relevant_columns = [RUNNAME_COL, FOLD_COL, FIT_TIME_COL, MSE_COL, COMPLEXITY_COL]
    df = df[relevant_columns]

    print(f"[info] {len(csv_files)} CSV-Dateien eingelesen, insgesamt {len(df)} Zeilen.")
    return df

def filter_fold_runs(df: pd.DataFrame) -> pd.DataFrame:
    fold_mask = (df[FOLD_COL] == "True") | (df[FOLD_COL] == True)
    return df[fold_mask].copy()


def filter_seed_and_fold_index(df: pd.DataFrame) -> pd.DataFrame:
    if RUNNAME_COL not in df.columns:
        raise KeyError(f"Spalte '{RUNNAME_COL}' fehlt im DataFrame. ")

    seed_pattern = re.compile(r"seed-(\d+)")
    fold_pattern = re.compile(r"fold-(\d+)-of-")

    df["seed_index"] = df[RUNNAME_COL].apply(lambda x: extract_number(seed_pattern, x))
    df["fold_index"] = df[RUNNAME_COL].apply(lambda x: extract_number(fold_pattern, x))
    
    filtered_df = df[
        df["seed_index"].between(0, 4) 
        & df["fold_index"].between(1, 20)
    ].copy()

    print(f"Nach Filter: {len(filtered_df)} Zeilen übrig.")

    return filtered_df

def extract_number(pattern:re.Pattern, text:str):
    if not isinstance(text, str):
        return None
    match = pattern.search(text)
    return int(match.group(1)) if match else None

def extract_grid_point_info(run_name: str) -> pd.Series:
    if not isinstance(run_name, str):
        return pd.Series({
            "dataset": None, "n_iter": None, "n_rules": None, "n_initial_rules": None, "label": None
        })
    match = gridpoint_pattern.search(run_name)
    if not match:
        return pd.Series({
            "dataset": None, "n_iter": None, "n_rules": None, "n_initial_rules": None, "label": None
        })
    return pd.Series({
        "dataset": match.group("dataset"),
        "n_iter": int(match.group("n_iter")),
        "n_rules": int(match.group("n_rules")),
        "n_initial_rules": int(match.group("n_initial_rules")),
        "label": match.group("label")
    })



def check_missing_values(df: pd.DataFrame) -> None:
    """Prüft NaNs in den relevanten Metrik-Spalten, aufgeschlüsselt nach Grid-Punkt/Seed."""

    metric_cols = [FIT_TIME_COL, MSE_COL, COMPLEXITY_COL]
    check_cols = group_cols + ["seed_index", "fold_index"]

    missing_mask = df[metric_cols].isna().any(axis=1)
    if missing_mask.any():
        bad_rows = df.loc[missing_mask, check_cols + metric_cols]
        warnings.warn(
            f"{missing_mask.sum()} Zeile(n) mit fehlenden Metrik-Werten gefunden."
        )
        print(bad_rows.to_string(index=False))


def check_fold_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Prüft pro Grid-Punkt und Seed, ob genau EXPECTED_FOLDS_PER_SEED Folds vorhanden sind,
    und ob genau EXPECTED_SEEDS Seeds vorhanden sind."""

    fold_counts = (
        df.groupby(group_cols + ["seed_index"], dropna=False)
        .size()
        .reset_index(name="n_folds")
    )

    # Seeds mit falscher Fold-Anzahl
    bad_folds = fold_counts[fold_counts["n_folds"] != EXPECTED_FOLDS_PER_SEED]
    if not bad_folds.empty:
        warnings.warn(
            f"{len(bad_folds)} Grid-Punkt/Seed-Kombination(en) haben nicht "
            f"{EXPECTED_FOLDS_PER_SEED} Folds:"
        )
        print(bad_folds.to_string(index=False))

    # Grid-Punkte mit falscher Seed-Anzahl
    seed_counts = (
        fold_counts.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_seeds")
    )
    bad_seeds = seed_counts[seed_counts["n_seeds"] != EXPECTED_SEEDS]
    if not bad_seeds.empty:
        warnings.warn(
            f"{len(bad_seeds)} Grid-Punkt(e) haben nicht {EXPECTED_SEEDS} Seeds:"
        )
        print(bad_seeds.to_string(index=False))

    return fold_counts  # optional zurückgeben, falls man's weiterverwenden will


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    grid_info = df[RUNNAME_COL].apply(extract_grid_point_info)
    df = pd.concat([df, grid_info], axis=1)

    unparsed = df["dataset"].isna().sum() 
    if unparsed:
        print(f"[warn] {unparsed} Zeilen konnten nicht geparst werden (RunName passt nicht zum erwarteten Muster).")

    #group_cols = ["dataset", "label", "n_iter", "n_rules", "n_initial_rules"]

    check_missing_values(df)
    df = df.drop_duplicates(subset=group_cols + ["seed_index", "fold_index"])
    check_fold_counts(df)

    #check for missing values, redundant?
    metric_cols = [FIT_TIME_COL, MSE_COL, COMPLEXITY_COL]
    for col in metric_cols:
        if col not in df.columns:
            warnings.warn(f"Spalte '{col}' fehlt komplett im DataFrame.")
            continue
        n_missing = df[col].isna().sum()
        if n_missing:
            warnings.warn(f"{n_missing} fehlende Werte in Spalte '{col}' gefunden.")


    #flat mean
    """ result_df = (
        df.groupby(group_cols, dropna=False)
        .agg(
            fit_time_mean=(FIT_TIME_COL, "mean"),
            elitist_complexity_mean=(COMPLEXITY_COL, "mean"),
            MSE_mean=(MSE_COL, "mean"),
        )
        .reset_index()
    ) """


    # nested mean
    per_seed = (
        df.groupby(group_cols + ["seed_index"], dropna=False)
        .agg(
            fit_time_mean=(FIT_TIME_COL, "mean"),
            elitist_complexity_mean=(COMPLEXITY_COL, "mean"),
            MSE_mean=(MSE_COL, "mean"),
        )
        .reset_index()
    )

    result_df = (
        per_seed.groupby(group_cols, dropna=False)
        .agg(
            fit_time_mean=("fit_time_mean", "mean"),
            elitist_complexity_mean=("elitist_complexity_mean", "mean"),
            MSE_mean=("MSE_mean", "mean"),
            #MSE_std=("MSE_mean", "std"),
        )
        .reset_index()
    )


    result_df["MSE_mean"] = -result_df["MSE_mean"]  # Vorzeichen umdrehen für MSE

    summary = (
        result_df.groupby(["dataset", "label"])
        .size()
        .reset_index(name="row_count")
    )

    print(summary)
    
    print(f"[info] Zusammenfassung erstellt: {len(result_df)} Zeilen.")

    return result_df


def main():
    df_all = load_from_csv()
    print(f"df_all:")
    print(df_all.head(20))

    df_just_fold_runs = filter_fold_runs(df_all)
    print(f"df_just_fold_runs:")
    print(df_just_fold_runs.head(20))

    df = filter_seed_and_fold_index(df_just_fold_runs)
    print(f"df:")
    print(df.head(20))

    summary = build_summary(df)
    print(f"summary:")
    print(summary.head(50))

    summary.to_csv("summary.csv", index=False)
    print(f"[info] {len(summary)} Zeile(n) geschrieben nach 'summary.csv'")


if __name__ == "__main__":
    main()