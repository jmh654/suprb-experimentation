"""
extract_learning_curves.py
===========================
Extrahiert Pro-Iterations-Metriken (MSE/MAE/Complexity über die SupRB-Iterationen)
aus MLflow, um Lernkurven-Analysen (Performance vs. Trainingsbudget) zu ermoeglichen.

Hintergrund
-----------
`mlflow.search_runs()` (wie in mlflow2csv.py / summary_csv.py verwendet) liefert pro Run
nur den *letzten* geloggten Wert einer Metrik zurueck ("test_neg_mean_squared_error" etc.
sind Endwerte nach der letzten Iteration).

Die Werte PRO Iteration werden aber schon waehrend des Trainings geloggt, siehe
`_log_estimator_run()` in mlflow_logging.py:

    for step, metrics in sorted(by_step.items()):
        mlflow.log_metrics(metrics, step=step)

Diese Historie ist in MLflow vorhanden, aber `search_runs()` zeigt nur den letzten step.
Um die volle History zu bekommen, muss man pro Run und Metrik explizit
`MlflowClient.get_metric_history(run_id, key)` aufrufen.

Dieses Skript:
  1) sucht alle Fold-Runs (tags.fold == "True"), genau wie summary_csv.py
  2) parsed dataset/n_iter/n_rules/n_initial_rules/label/seed/fold aus dem Run-Namen
  3) erkennt automatisch, welche Metriken PRO ITERATION geloggt wurden
     (Kriterium: get_metric_history() liefert mehr als 1 Eintrag)
  4) baut daraus eine Long-Format-Tabelle mit einer Zeile pro (Run, Iteration)

WICHTIG - bitte vor dem Lauf pruefen/anpassen:
  - MLFLOW_TRACKING_URI  (Pfad zu deinem mlruns-Ordner bzw. Tracking-Server)
  - EXPERIMENT_IDS / EXPERIMENT_NAMES (welche Experimente durchsucht werden sollen)
  - Die tatsaechlichen Metrik-Keys, die der SupRB DefaultLogger loggt (z.B. "mse",
    "complexity", "fitness", ...). Das Skript nimmt automatisch ALLE Metriken mit
    Step-Historie > 1 mit, filtert also nichts nach Namen - falls dir das zu viel ist,
    trage die gewuenschten Keys in METRIC_KEYS_WHITELIST ein.
"""

#config_id,  dataset,    repetition_id,  seed_index, fold_index, iteration,  n_rules_in_pool, 
#   
# population_fitness_mean,    population_fitness_percentile10,    population_fitness_percentile90,    
# population_complexity_percentile90, pool_fitness_percentile10,  pool_fitness_percentile25,  population_error_min,   
# population_complexity_percentile75, training_score, pool_fitness_max,   population_complexity_min,  pool_fitness_mean,  
# elitist_error,  pool_fitness_median,    population_fitness_median,  pool_fitness_min,   population_error_mean,  
# population_complexity_max,  pool_fitness_percentile75,  population_fitness_min, population_complexity_mean, 
# pool_fitness_percentile90,  population_complexity_percentile10, population_complexity_median,   
# population_fitness_percentile25,    pool_size,  elitist_fitness,    population_error_percentile10,  
# population_error_median,    population_error_percentile75,  population_fitness_percentile75,    
# population_error_percentile90,  population_error_max,   population_error_percentile25,  elitist_complexity, 
# population_complexity_percentile25, population_size,    population_fitness_max

from __future__ import annotations

import re
import warnings
from typing import Optional
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------------
# Konfiguration - hier anpassen
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MLFLOW_URI = os.path.join(_HERE, "mlruns")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_IDS: Optional[list[str]] = ['485440579075042350']      # z.B. ['482653026699552589', ...]
EXPERIMENT_NAMES: Optional[list[str]] = None    # alternativ per Name, z.B. ["SupRB | problem=airfoil_self_noise | tune"]

# Falls leer -> alle Metriken mit >1 Step-Eintrag mitgenommen.
METRIC_KEYS_WHITELIST: list[str] = ['elitist_error', 'elitist_complexity']

RUNNAME_COL = "tags.mlflow.runName"
FOLD_COL = "tags.fold"


# gridpoint_pattern = re.compile(
#     r"^(?P<dataset>.+?)__ni(?P<n_iter>\d+)__nr(?P<n_rules>\d+)__nir(?P<n_initial_rules>\d+)__(?P<label>.+?)"
#     r"\.seed-(?P<seed>\d+)\.fold-(?P<fold>\d+)-of-\d+"
# )

# asn_notune_pattern = re.compile(
#     r"^(?P<dataset>.+?)__ni(?P<n_iter>\d+)__nr(?P<n_rules>\d+)__nir(?P<n_initial_rules>\d+)"
#     r"\.seed-(?P<seed>\d+)\.fold-(?P<fold>\d+)-of-\d+"
# )

pattern = re.compile(
    r"^(?P<dataset>.+?)"
    r"__ni(?P<n_iter>\d+)"
    r"__nr(?P<n_rules>\d+)"
    r"__nir(?P<n_initial_rules>\d+)"
    r"(?:__(?P<label>.+?))?"
    r"\.seed-(?P<seed>\d+)\.fold-(?P<fold>\d+)-of-\d+"

)


def get_fold_runs() -> pd.DataFrame:
    kwargs = {}
    if EXPERIMENT_IDS:
        print("Search experiment with id(s)")
        kwargs["experiment_ids"] = EXPERIMENT_IDS
    elif EXPERIMENT_NAMES:
        print("Search experiment with names")
        exp_ids = []
        for name in EXPERIMENT_NAMES:
            exp = mlflow.get_experiment_by_name(name)
            if exp is not None:
                exp_ids.append(exp.experiment_id)
        kwargs["experiment_ids"] = exp_ids
    else:
        print("Search all experiments")
        kwargs["search_all_experiments"] = True

    df = mlflow.search_runs(filter_string=f'{FOLD_COL} = "True"', **kwargs)
    if df.empty:
        raise RuntimeError("Keine Fold-Runs gefunden - EXPERIMENT_IDS/EXPERIMENT_NAMES pruefen.")
    if "run_id" in df.columns:
            df = df.drop_duplicates(subset=["run_id"])
    return df


def parse_run_name(run_name: str) -> Optional[dict]:
    match = pattern.search(run_name)
    if not match:
        return None
    return {
        "dataset": match.group("dataset"),
        "n_iter": int(match.group("n_iter")),
        "n_rules": int(match.group("n_rules")),
        "n_initial_rules": int(match.group("n_initial_rules")),
        "label": match.group("label"),          #None for parts of asn notune /"tune" / "notune" / "moo" / "ns_..."
        "seed_index": int(match.group("seed")),
        "fold_index": int(match.group("fold")),
    }

def add_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Wendet parse_run_name() auf jede Zeile an und hängt die geparsten Felder
    (dataset, n_iter, n_rules, n_initial_rules, label, seed_index, fold_index) an."""
    if RUNNAME_COL not in df.columns:
        raise KeyError(f"Spalte '{RUNNAME_COL}' fehlt im DataFrame.")

    parsed = df[RUNNAME_COL].apply(
        lambda name: parse_run_name(name) if isinstance(name, str) else None
    )
    parsed_df = pd.json_normalize(parsed)  # None -> Zeile mit NaNs

    out = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)

    n_unparsed = parsed_df["seed_index"].isna().sum() if "seed_index" in parsed_df else len(out)
    if n_unparsed:
        print(f"[warn] {n_unparsed} Run(s) konnten nicht anhand des Run-Namens geparst werden.")

    return out


def filter_seed_and_fold_index(df: pd.DataFrame) -> pd.DataFrame:
    """Filtert auf bereits geparste seed_index/fold_index-Spalten (siehe add_parsed_columns)."""
    if "seed_index" not in df.columns or "fold_index" not in df.columns:
        raise KeyError(
            "'seed_index'/'fold_index' fehlen im DataFrame. "
            "Vorher add_parsed_columns() aufrufen."
        )

    filtered_df = df[
        df["seed_index"].between(0, 4)
        & df["fold_index"].between(1, 20)
    ].copy()

    filtered_df["seed_index"] = filtered_df["seed_index"].astype(int)
    filtered_df["fold_index"] = filtered_df["fold_index"].astype(int)

    print(f"Nach Filter: {len(filtered_df)} Zeilen übrig.")
    return filtered_df


def build_config_id(info: dict) -> str:
    # Format: f"suprb_{n_iter}_{n_rules}_{n_initial_rules}_{label}"
    return f"suprb_{info['n_iter']}_{info['n_rules']}_{info['n_initial_rules']}_{info['label']}"


def extract_per_iteration_rows(client: MlflowClient, run_id: str, run_info: dict) -> list[dict]:
    """Liest fuer einen Fold-Run alle Metriken mit Step-Historie aus und baut Zeilen."""
    run = client.get_run(run_id)
    metric_keys = list(run.data.metrics.keys())

    if METRIC_KEYS_WHITELIST:
        metric_keys = [k for k in metric_keys if k in METRIC_KEYS_WHITELIST]

    # Historie pro Metrik holen; nur Metriken mit >1 Eintrag sind "pro Iteration"
    histories: dict[str, dict[int, float]] = {}
    for key in metric_keys:
        hist = client.get_metric_history(run_id, key)
        if len(hist) > 1:
            histories[key] = {m.step: (m.value, m.timestamp) for m in hist}

    if not histories:
        warnings.warn(f"Run {run_id}: keine Pro-Iterations-Metriken gefunden (nur Endwerte).")
        return []

    all_steps = sorted({step for h in histories.values() for step in h.keys()})
    #start_time_ms = run.info.start_time

    n_iter = run_info["n_iter"]
    n_rules = run_info["n_rules"]
    n_initial_rules = run_info["n_initial_rules"]
    config_id = build_config_id(run_info)
    # ein einzelner int fuer "Seed/Wiederholung/CV-Fold" - Annahme: max. 100 Folds pro Seed.
    # Falls das nicht passt, einfach seed_index/fold_index als getrennte Spalten lassen.
    #repetition_id = run_info["seed_index"] * 100 + run_info["fold_index"]

    rows = []
    for step in all_steps:
        row = {
            "config_id": config_id,
            "dataset": run_info["dataset"],
            #"repetition_id": repetition_id,
            "seed_index": run_info["seed_index"],
            "fold_index": run_info["fold_index"],
            "iteration": step,
            "n_rules_in_pool": (step + 1) * n_rules + n_initial_rules, #step starts at 0
        }
        #elapsed = None
        for key, hist in histories.items():
            if step in hist:
                value, ts_ms = hist[step]
                row[key] = value
                #elapsed = ts_ms / 1000.0
                #elapsed = (ts_ms - start_time_ms) / 1000.0  # Sekunden seit Run-Start
        #row["elapsed_seconds"] = elapsed
        rows.append(row)

    return rows

def validate_row_counts(df: pd.DataFrame, n_iter: int, expected_seeds: int = 5, expected_folds: int = 20) -> None:
    """Prüft, ob die erwartete Anzahl an Zeilen pro Run vorhanden ist (n_iter)."""
    grouped = df.groupby(["config_id", "dataset"])

    for (config_id, dataset), group in grouped:
        
        expected_rows = n_iter * expected_seeds * expected_folds

        actual_rows = len(group)
        if actual_rows != expected_rows:
            warnings.warn(
                f"Run {config_id} / {dataset}: "
                f"erwartet {expected_rows} Zeilen, gefunden {actual_rows}."
            )


def main():
    print("mlrun4Analysis")
    client = MlflowClient()
    fold_runs_df = get_fold_runs()
    parsed_df = add_parsed_columns(fold_runs_df)
    filtered_fold_runs_df = filter_seed_and_fold_index(parsed_df)
    print("Found and Filtered Fold Runs")

    all_rows: list[dict] = []
    skipped = 0

    for _, row in filtered_fold_runs_df.iterrows():
        info = {
            "dataset": row["dataset"],
            "n_iter": int(row["n_iter"]),
            "n_rules": int(row["n_rules"]),
            "n_initial_rules": int(row["n_initial_rules"]),
            "label": row["label"],
            "seed_index": int(row["seed_index"]),
            "fold_index": int(row["fold_index"]),
        }
        all_rows.extend(extract_per_iteration_rows(client, row["run_id"], info))

    if skipped:
        print(f"[warn] {skipped} Run(s) konnten nicht anhand des Run-Namens geparst werden.")
    print("parsed all Fold Runs")

    if not all_rows:
        raise RuntimeError(
            "Keine Pro-Iterations-Daten gefunden."
        )

    out_df = pd.DataFrame(all_rows)
    print("new DataFrame created")

    validate_row_counts(out_df, info["n_iter"])

    out_df = out_df.sort_values(["config_id", "dataset", "seed_index", "fold_index", "iteration"]) #"repetition_id", "iteration"])
    out_df.to_csv(f"{_HERE}/learning_curves.csv", index=False)
    print(f"[info] {len(out_df)} Zeilen geschrieben nach 'learning_curves.csv'")
    print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()