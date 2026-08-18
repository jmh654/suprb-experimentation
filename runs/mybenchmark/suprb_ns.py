"""
suprb_default.py
================
Hauptskript für einen SupRB Grid-Punkt:
  1. Daten laden & skalieren
  2. Estimator bauen
  3. Optionales Hyperparameter-Tuning (Optuna)
  4. Finale Cross-Validation
  5. Ergebnisse in MLflow loggen
 
Aufruf (direkt):
    python suprb_ns.py -p airfoil_self_noise -n 32 -r 4 -i 4
 
Aufruf (via SLURM):
    sbatch default.sbatch  (setzt DATASET, N_ITER, N_RULES, N_INITIAL_RULES via --export)
"""
from __future__ import annotations
import numpy as np
import click
import time
import mlflow

from datetime import datetime

from optuna import Trial
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from sklearn.linear_model import Ridge
from sklearn.utils import Bunch, shuffle
from sklearn.model_selection import ShuffleSplit

from problems import scale_X_y

import suprb
from suprb import rule, SupRB

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es, origin, mutation, ns
from suprb.solution.initialization import RandomInit

from evaluation import run_evaluation
from mlflow_logging import log_results, log_results_multi_seed
from tuning import run_tuning, suprb_param_space, suprb_param_space_ns

import os
from joblib import Parallel, delayed
from joblib import parallel_backend

from functools import partial

RANDOM_STATE = 42
NUM_SEEDS = 5
N_CPU = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) 

_HERE = os.path.dirname(os.path.abspath(__file__))

MLFLOW_URI    = os.path.join(_HERE, "mlruns")
OPTUNA_DB_DIR = os.path.join(_HERE, "optuna_dbs")
os.makedirs(OPTUNA_DB_DIR, exist_ok=True)


# ============================================================
# TEMPORÄRER DEBUG-PATCH - vor Merge wieder entfernen!
# Patcht RouletteWheel.__call__, um zu loggen, WANN und WARUM
# sum(fitnesses) == 0 bzw. NaN auftritt. Verhalten bleibt
# unverändert - der Original-Call wird danach ganz normal
# ausgeführt (inkl. des bestehenden try/except-Verhaltens).
# ============================================================
""" import numpy as np
from suprb.optimizer.rule.selection import RouletteWheel

_orig_roulette_call = RouletteWheel.__call__
_debug_call_counter = {"n": 0}


def _debug_roulette_call(self, rules, random_state, size=1):
    rules_ = [rule for rule in rules if rule.fitness_ != -np.inf]

    if rules_:
        fitnesses = np.array([rule.fitness_ for rule in rules_])
        total = np.sum(fitnesses)
        problematic = (total == 0) or np.isnan(total) or np.any(np.isnan(fitnesses))

        if problematic:
            _debug_call_counter["n"] += 1
            n = _debug_call_counter["n"]
            print(f"\n=== DEBUG RouletteWheel #{n} ===")
            print(f"  rules total (vor -inf-Filter): {len(rules)}")
            print(f"  rules_ (nach -inf-Filter):      {len(rules_)}")
            print(f"  fitness_ Werte in rules_:        {fitnesses}")
            print(f"  sum(fitnesses) = {total}")
            print(f"  alle 0?   {np.all(fitnesses == 0)}")
            print(f"  alle -inf gefiltert? ursprüngliche fitness_ aller rules:")
            print(f"    {[r.fitness_ for r in rules]}")
            print("=" * 45)

    return _orig_roulette_call(self, rules, random_state, size)

def _debug_roulette_call(self, rules, random_state, size=1):
    raw_fitnesses = np.array([r.fitness_ for r in rules])
    print(f"[DEBUG] raw fitness_ (unfiltered): {raw_fitnesses}", flush=True)
    print(f"[DEBUG] sum: {np.sum(raw_fitnesses)}", flush=True)
    return _orig_roulette_call(self, rules, random_state, size)


RouletteWheel.__call__ = _debug_roulette_call
print(
    "DEBUG PATCH:",
    RouletteWheel.__call__.__name__,
    flush=True
)
print(">>> DEBUG PATCH ACTIVE <<<", flush=True)   # <- diese Zeile ergänzen """
# ============================================================
# ENDE DEBUG-PATCH
# ============================================================

""" import suprb.optimizer.rule.selection as sel_mod
print("selection.py Pfad:", sel_mod.__file__, flush=True)
print("RouletteWheel is patched class?",
      RouletteWheel is sel_mod.RouletteWheel, flush=True)
print("RouletteWheel.__call__ patched?",
      RouletteWheel.__call__ is _debug_roulette_call, flush=True)
print("id(RouletteWheel):", id(RouletteWheel), flush=True)
 """

def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)
    raise ValueError(f"Kein Dataset '{name}' gefunden (erwartet: problems.datasets.load_{name})")
  

def build_estimator(n_iter: int, n_rules: int, n_initial_rules: int) -> SupRB: #use_current_population: bool) -> SupRB:
    estimator = SupRB(
        rule_discovery=ns.NoveltySearch(
            init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(),
                                              model=Ridge(alpha=0.01,
                                                          random_state=RANDOM_STATE)),
            origin_generation=origin.SquaredError(),
            mutation=mutation.HalfnormIncrease(),
            #use_population_for_archive=use_current_population,
        ),
        #solution_composition=ga.GeneticAlgorithm(n_iter=32, population_size=32, selection=ga.selection.Tournament()),
        solution_composition=ga.GeneticAlgorithm(n_iter=32, population_size=32), #TODO: so? tunen? 
        n_iter=n_iter,
        n_rules=n_rules,
        n_initial_rules=n_initial_rules,
        verbose=1,
        logger=CombinedLogger(
            [('stdout', StdoutLogger()), ('default', DefaultLogger())]),
    )
    #sel_instance = getattr(suprb.optimizer.rule.selection, "RouletteWheel")()
    #print("Instanz-Klasse id:", id(type(sel_instance)), flush=True)
    #print("Ist gepatcht?", type(sel_instance).__call__ is _debug_roulette_call, flush=True)
    return estimator

    
    """ SupRB(
        rule_discovery=es.ES1xLambda(
            operator="&",
            n_iter=1000,
            delay=30,
            init=rule.initialization.MeanInit(
                fitness=rule.fitness.VolumeWu(),
                model=Ridge(alpha=0.01, random_state=RANDOM_STATE),
            ),
            mutation=mutation.HalfnormIncrease(),
            origin_generation=origin.SquaredError(),
        ),

        solution_composition=ga.GeneticAlgorithm(
            n_iter=32,
            population_size=32,
            selection=ga.selection.Tournament(),
        ),

        n_iter=n_iter,
        n_rules=n_rules,
        n_initial_rules=n_initial_rules,
        verbose=1, 
        logger=CombinedLogger(
            [("stdout", StdoutLogger()), ("default", DefaultLogger())]
        ),
    ) """

def _evaluate_one_seed(estimator, X, y, tuned_params, rs):
        cv_splitter = ShuffleSplit(n_splits=20, test_size=0.25, random_state=int(rs))
        print(f"[evaluation] [{datetime.now():%Y-%m-%d %H:%M:%S}] Seed {rs} gestartet", flush=True)
        result = run_evaluation(
            estimator=estimator,
            X=X, y=y,
            tuned_params=tuned_params,
            cv=cv_splitter,
            n_jobs=1,          
            random_state=int(rs),
            verbose=2, 
        )
        print(f"[evaluation] [{datetime.now():%Y-%m-%d %H:%M:%S}] Seed {rs} abgeschlossen", flush=True)
        return result


@click.command()
@click.option("-p", "--problem",          type=str, default="airfoil_self_noise", show_default=True)
@click.option("-j", "--job_id",           type=str, default="NA",                 show_default=True)
@click.option("-n", "--n_iter",           type=int, default=32,                   show_default=True)
@click.option("-r", "--n_rules",          type=int, default=4,                    show_default=True)
@click.option("-i", "--n_initial_rules",  type=int, default=4,                    show_default=True)
#@click.option('-a', '--use_current_population', type=click.BOOL, default=False                     )
def run(
    problem: str,
    job_id: str,
    n_iter: int,
    n_rules: int,
    n_initial_rules: int,
    #use_current_population: bool
):
    ns_type = 'MCNS'
    #label = "ns_population" if use_current_population else "ns_generational"

    print(f"[run] Problem={problem}  job_id={job_id}  n_iter={n_iter}  "
        f"n_rules={n_rules}  n_initial_rules={n_initial_rules}  {ns_type}") #use_current_population={use_current_population}")


    

    t0 = time.perf_counter()

    #-----------------------------------------------------------------------
    # Data
    #-----------------------------------------------------------------------

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y, _ = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=RANDOM_STATE)

    #-----------------------------------------------------------------------
    # Estimator
    #-----------------------------------------------------------------------

    grid_params = dict(n_iter=n_iter, n_rules=n_rules, n_initial_rules=n_initial_rules) #use_current_population=use_current_population)
    estimator = build_estimator(**grid_params)

    #-------------------------------------------------------------------------
    # Optinal Tuning (optional)
    # trails sequentiell, cv parallel (innerhalb jedes trials)
    #--------------------------------------------------------------------------
    tuned_params: dict = {}
    tuning_walltime: float = 0.0

    study_name = f"{problem}__ni{n_iter}__nr{n_rules}__nir{n_initial_rules}__{ns_type}" #__{label}"
    
    if True: 
        sub_dir = f"{problem}_{ns_type}"
        os.makedirs(os.path.join(OPTUNA_DB_DIR, sub_dir), exist_ok=True)
        db_url = f"sqlite:///{OPTUNA_DB_DIR}/{sub_dir}/{study_name}.db" #keine gemeinsame DB der SLURM jobs, da parallele Optuna-Trials zu Konflikten führen würden. Stattdessen: separate DB pro Job/Studie.

        param_space_fn = partial(
            suprb_param_space_ns,   #ns prameter space
            ns_type=ns_type,                              # z.B. "MCNS"
            #use_current_population=use_current_population, # kommt aus dem CLI-Flag -a
        )

        print(f"Starting tuning for {study_name}")
        tuned_params = run_tuning(
            estimator=estimator,
            X=X,
            y=y,
            param_space_fn=param_space_fn, 
            study_name=study_name,
            storage_url=db_url,
            n_trials=1000, 
            timeout=60*60*24,  # 24 hours
            cv=4,
            n_jobs_cv=N_CPU, #parallelität innerhalb cv jedes trials 
            n_jobs=1, #prallelität der trials, sqlite -> n_jobs=1
            random_state=RANDOM_STATE,
            scoring="neg_mean_squared_error",
            verbose=1,
        )

        tuning_walltime = time.perf_counter() - t0
        print(f"[tuning] Tuning completed in {tuning_walltime:.2f} seconds")

        print(f"[tuning] Best Params: {tuned_params}")
    
    #-------------------------------------------------------------------------
    # Evaluation
    # Seeds parallel, cv sequentiell
    #--------------------------------------------------------------------------
    t1 = time.perf_counter()

    random_states = np.random.SeedSequence(RANDOM_STATE).generate_state(NUM_SEEDS)
    
    seed_results: list[tuple] = []

    seed_results = Parallel(n_jobs=N_CPU)(
        delayed(_evaluate_one_seed)(estimator, X, y, tuned_params, rs)
        for rs in random_states
    )

    evaluation_walltime = time.perf_counter() - t1
    print(f"[evaluation] [{datetime.now():%Y-%m-%d %H:%M:%S}] Evaluation completed in {evaluation_walltime:.2f} seconds")
    
    walltime_metrics = dict(
        tuning_walltime_s=round(tuning_walltime, 2),
        evaluation_walltime_s=round(evaluation_walltime, 2),
    )

    #-------------------------------------------------------------------------
    # MLflow Logging
    #--------------------------------------------------------------------------
    
    experiment_name = f"SupRB | problem={problem} | {ns_type}" #| {label}" 
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    log_results_multi_seed(
        run_name=study_name,
        grid_params=grid_params,
        tuned_params=tuned_params,
        seed_results=seed_results,          # liste der (estimators, cv_results)-Tupel
        random_states=random_states,
        walltime_metrics=walltime_metrics,
    )
    print(f"[run] MLflow-Ergebnisse geloggt unter Experiment '{experiment_name}'")


if __name__ == "__main__":
    run()



