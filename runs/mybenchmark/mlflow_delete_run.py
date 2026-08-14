""" 
finds root from run_id or run_name and
deletes recursicely all runs in the subtree of the given run_id, including the root run itself.
"""
import mlflow
import pandas as pd

PARENT_TAG = "tags.mlflow.parentRunId"
NAME_TAG = "tags.mlflow.runName"


def load_run(experiment_id: str) -> pd.DataFrame:
    """Alle Runs aus allen Experimenten als DataFrame, ueber die fluent API."""
    df = mlflow.search_runs(experiment_ids=[experiment_id])
    
    if df.empty:
        raise SystemExit(f"Keine Runs im Experiment '{experiment_id}' gefunden.")

    for col in ("tags.mlflow.runName", "tags.mlflow.parentRunId"):
        if col not in df.columns:
            df[col] = None
    return df

def _find_root_run_id_by_child_run_id(df: pd.DataFrame, run_id: str) -> str:
    """Findet die run_id des Root-Runs (parentRunId ist NaN/leer) fuer den gegebenen Kind-Run."""

    # find child run
    child_row = df[df["run_id"] == run_id]
    if child_row.empty:
        raise SystemExit(f"Run mit ID '{run_id}' nicht gefunden.")

    # get parentRunId of the child run
    parent_run_id = child_row.iloc[0][PARENT_TAG]

    # If the parentRunId is NaN or empty, the child run is the root run itself
    if pd.isna(parent_run_id) or parent_run_id == "":
        return run_id

    # Recursively find the root run_id by looking up the parent run
    return _find_root_run_id_by_child_run_id(df, parent_run_id)


def _find_root_run_id_by_name(df: pd.DataFrame, run_name: str) -> str:
    """Findet die run_id des Root-Runs (parentRunId ist NaN/leer) mit dem gegebenen Namen."""

    is_root = df[PARENT_TAG].isna() | (df[PARENT_TAG] == "")
    roots = df[is_root]

    print(f"{len(roots)} Root-Run(s) im DataFrame gefunden:")
    for _, row in roots.iterrows():
        print(f"  - {row['run_id']}  ({row[NAME_TAG]}) [{row['start_time']}]")

    matches = roots[roots[NAME_TAG] == run_name]
    #matches = df[is_root & (df[NAME_TAG] == run_name)]
 
    if matches.empty:
        raise SystemExit(
            f"Kein Root-Run (ohne parentRunId) mit run_name='{run_name}' gefunden."
        )
    if len(matches) > 1:
        ids = ", ".join(matches["run_id"].tolist())
        raise SystemExit(
            f"Mehrdeutig: mehrere Root-Runs mit run_name='{run_name}' gefunden: {ids}. "
            f"Bitte stattdessen --run-id verwenden."
        )
 
    return matches.iloc[0]["run_id"]


def _collect_descendant_ids(df: pd.DataFrame, root_run_id: str) -> list[str]:
    to_visit = [root_run_id]
    collected: list[str] = []
 
    while to_visit:
        current = to_visit.pop()
        if current in collected:
            continue
        collected.append(current)
 
        children = df[df[PARENT_TAG] == current]["run_id"].tolist()
        to_visit.extend(children)
 
    return collected

def print_duplicated_roots (df: pd.DataFrame) -> None:
    is_root = df[PARENT_TAG].isna() | (df[PARENT_TAG] == "")
    roots = df[is_root]
    dupes = roots[roots.duplicated(subset=[NAME_TAG], keep=False)]
    if not dupes.empty:
        print("WARNUNG: doppelte Root-Run-Namen gefunden:")
        with pd.option_context("display.max_colwidth", None, "display.width", None):
            print(dupes[["run_id", "start_time", NAME_TAG]].sort_values(NAME_TAG))

def delete_run(run_id: str, df: pd.DataFrame) -> None:
    run_ids_to_delete = _collect_descendant_ids(df, run_id)

    for run_id in run_ids_to_delete:
        match = df.loc[df["run_id"] == run_id, "tags.mlflow.runName"]
        if match.empty:
            print(f"WARNUNG: run_id {run_id} nicht im DataFrame gefunden, überspringe.")
            continue
        run_name_to_print = match.iloc[0]
        print(f"Deleting run {run_id} ({run_name_to_print})")
        mlflow.delete_run(run_id)

def delete_folds_6_7_of_run(run_id: str, df: pd.DataFrame) -> None:
    # Collect all descendant run IDs
    run_ids_to_delete = _collect_descendant_ids(df, run_id)

    # Filter out runs with names containing "fold_6" or "fold_7"
    filtered_run_ids = []
    for run_id in run_ids_to_delete:
        match = df.loc[df["run_id"] == run_id, "tags.mlflow.runName"]
        if match.empty:
            print(f"WARNUNG: run_id {run_id} nicht im DataFrame gefunden, überspringe.")
            continue
        run_name_to_print = match.iloc[0]
        if "fold_6" in run_name_to_print or "fold_7" in run_name_to_print:
            filtered_run_ids.append(run_id)

    # Delete the filtered runs
    for run_id in filtered_run_ids:
        match = df.loc[df["run_id"] == run_id, "tags.mlflow.runName"]
        if match.empty:
            print(f"WARNUNG: run_id {run_id} nicht im DataFrame gefunden, überspringe.")
            continue
        run_name_to_print = match.iloc[0]
        print(f"Deleting run {run_id} ({run_name_to_print})")
        #mlflow.delete_run(run_id)

def print_subtree(run_id: str, df: pd.DataFrame) -> None:
    #get the root run id for the given run_id
    root_run_id = _find_root_run_id_by_child_run_id(df, run_id)

    run_ids_to_print = _collect_descendant_ids(df, root_run_id)
    print(f"Subtree of run {root_run_id}:")
    for current_id in run_ids_to_print:
        _print_run_info(current_id, df)


def _print_run_info(run_id: str, df: pd.DataFrame) -> None:
    match = df.loc[df["run_id"] == run_id, "tags.mlflow.runName"]
    if match.empty:
        print(f"WARNUNG: run_id {run_id} nicht im DataFrame gefunden, überspringe.")
        return
    run_name_to_print = match.iloc[0]
    print(f"  - {run_id} ({run_name_to_print})")




def main() -> None:
    #run_id = "532f8b65489d43d9a016bb72e6241b08"  
    run_ids = [
        "cee5138e07cc4201981e026439d91835",
        ]
    #run_name = "airfoil_self_noise__ni64__nr16__nir24__tune"


    experiment_id = '514426764301360574'
        #'485440579075042350'    #asn notune,
        #'482653026699552589',   #asn tune,
        #'796385774745469510',   #ccpp notune,
        #'514426764301360574',   #ccpp tune,
        #'197531564208076366',   #cs notune
        #'517600193416760241',   #cs tune
        #'368928382700897566',   #ec notune
        #'450111382996578006',    #ps notune

    df = load_run(experiment_id)
    df = df.drop_duplicates(subset=["run_id"])
    print(f"Anzahl Runs im Experiment {experiment_id}: {len(df)}")

    #print_duplicated_roots(df)

    #print(df[["run_id", "tags.mlflow.runName", "start_time"]].sort_values("start_time"))

    if not run_ids:
        print("Kein run_id angegeben, Suche nach Root-Run mit run_name='%s'" % run_name)
        df_runs = df[df["tags.mlflow.runName"].str.contains(run_name, case=False, na=False)]
        root_id = _find_root_run_id_by_name(df_runs, run_name)
        print(f"Root-Run für run_name='{run_name}' gefunden: run_id={root_id}")
        #delete_folds_6_7_of_run(root_id, df) #deletes only runs with fold_6 or fold_7 in the name
        #delete_run(root_id, df)
    else:
        print(f"{len(run_ids)} run_id(s) angegeben: {run_ids}")
        for current_id in run_ids:
            print(f"--- Verarbeite run_id: {current_id} ---")
            #print_subtree(current_id, df) #finds root and prints whole subtree
            
            root_id = _find_root_run_id_by_child_run_id(df, current_id)
            #print(f"Root-Run für run_id='{current_id}' gefunden: run_id={root_id}")
            _print_run_info(root_id, df) #prints only the run itself

            #delete_folds_6_7_of_run(root_id, df) #deletes only runs with fold_6 or fold_7 in the name
            delete_run(root_id, df) #deletes subtree of id, does NOT find root 


if __name__ == "__main__":
    main()
