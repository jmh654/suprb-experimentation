from __future__ import annotations
from typing import Optional
from numbers import Integral
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, ShuffleSplit, cross_validate

from suprb.logging.metrics import hypervolume
from suprb.solution.fitness import c_norm, pseudo_accuracy

def _check_scoring(extra: Optional[str | list[str]] = None) -> list[str]:
    base = {"r2", "neg_mean_squared_error", "neg_mean_absolute_error"}
    if extra is not None:
        base.update([extra] if isinstance(extra, str) else extra)
    return list(base)

def _moo_hv_pf_from_scores(scores: dict, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Berechnet pro CV-Fold die Pareto-Front (als (c_norm, PACC)-Punkte) und das
    Hypervolume auf dem jeweiligen Test-Split und schreibt beides unter
    "test_pf_fitness" bzw. "test_hypervolume" in `scores`.

    Erwartet, dass `scores` "estimator" (return_estimator=True) und
    "indices" (return_indices=True) aus cross_validate enthält.
    """
    estimators = scores["estimator"]
    test_indices = scores["indices"]["test"]

    pf_test_fitnesses = []
    test_hvs = []

    for i in range(len(estimators)):
        test_X = X[test_indices[i]]
        test_y = y[test_indices[i]]

        pf = deepcopy(estimators[i].solution_composition_.pareto_front())
        for solution in pf:
            solution.fit(test_X, test_y, cache=False)

        n = pf[0].fitness.max_genome_length_
        reference = pf[0].fitness.hv_reference if hasattr(pf[0].fitness, "hv_reference") else np.array([1.0, 1.0])

        pf_points = [
            [1 - c_norm(solution.complexity_, n), 1 - pseudo_accuracy(solution.error_)] for solution in pf
        ]
        pf_points = sorted(pf_points, key=lambda p: p[0], reverse=True)

        pf_test_fitnesses.append(pf_points)
        test_hvs.append(hypervolume(pf_points, reference))

    scores["test_pf_fitness"] = pf_test_fitnesses
    scores["test_hypervolume"] = np.array(test_hvs)
    return scores

def _soo_fitness_from_scores(scores: dict, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Berechnet die Elitist-Fitness auf dem Test-Split (z.B. für die
    GA/TwoStage-Baseline im MOO-Vergleich) und schreibt sie unter
    "test_elitist_fitness" in `scores`.
    """
    estimators = scores["estimator"]
    test_indices = scores["indices"]["test"]

    elitist_fitnesses = []
    for i in range(len(estimators)):
        test_X = X[test_indices[i]]
        test_y = y[test_indices[i]]
        elitist = estimators[i].solution_composition_.elitist()
        elitist.fit(test_X, test_y, cache=False)
        elitist_fitnesses.append(elitist.fitness_)

    scores["test_elitist_fitness"] = np.array(elitist_fitnesses)
    return scores


def run_evaluation(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    tuned_params: Optional[dict] = None,
    cv: Optional[ShuffleSplit | KFold] = None,
    n_jobs: int = 1,
    random_state: Optional[int] = None,
    scoring: Optional[str | list[str]] = None,
    verbose: int = 0,
) -> tuple[list[BaseEstimator], dict]:
    
    scoring_list = _check_scoring(scoring)
        
    est = clone(estimator)
    if random_state is not None:
        est.set_params(random_state=random_state)
    if tuned_params:
        est.set_params(**tuned_params)


    print(f"[evaluation] Start cross_validate | cv={cv} | scoring={scoring_list} | n_jobs={n_jobs}")

    raw_scores = cross_validate(
        estimator=est,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring_list,
        n_jobs=n_jobs,
        return_estimator=True,
        verbose=verbose,
        error_score="raise",
    )

    estimators: list[BaseEstimator] = raw_scores.pop("estimator") 
    scores: dict = dict(raw_scores) 

    # print summarized scores 
    for key, arr in scores.items():
        if hasattr(arr, "__len__"):
            print(
                f"[evaluation]  {key}: mean={np.mean(arr):.4f}  std={np.std(arr):.4f}"
            )

    return estimators, scores


def run_moo_evaluation(
    estimator, 
    X: np.array, 
    y: np.array, 
    tuned_params=None, 
    cv=None,
    n_jobs=1, 
    random_state=None, 
    verbose=0,
) -> tuple[list[BaseEstimator], dict]:

    est = clone(estimator)
    if random_state is not None:
        est.set_params(random_state=random_state)
    if tuned_params:
        est.set_params(**tuned_params)

    print(f"[evaluation] Start MOO cross_validate | cv={cv} | n_jobs={n_jobs}")

    raw_scores = cross_validate(
        estimator=est, 
        X=X, 
        y=y, 
        cv=cv, 
        n_jobs=n_jobs,
        return_estimator=True, 
        return_indices=True,
        verbose=verbose, 
        error_score="raise",
    )

    raw_scores = _moo_hv_pf_from_scores(raw_scores, X, y)
    """ if include_elitist_fitness:
        raw_scores = _soo_fitness_from_scores(raw_scores, X, y) """

    raw_scores.pop("indices")
    estimators: list[BaseEstimator] = raw_scores.pop("estimator")
    scores: dict = dict(raw_scores)

    print(
        f"[evaluation]  test_hypervolume: mean={np.mean(scores['test_hypervolume']):.4f}  "
        f"std={np.std(scores['test_hypervolume']):.4f}"
    )

    """ estimators = scores["estimator"]
    test_idx = scores["indices"]["test"]
    pf_fitnesses, hvs = [], []

    for i, est_i in enumerate(estimators):
        tX, ty = X[test_idx[i]], y[test_idx[i]]
        pf = deepcopy(est_i.solution_composition_.pareto_front())
        for sol in pf:
            sol.fit(tX, ty, cache=False)
        n = pf[0].fitness.max_genome_length_
        ref = pf[0].fitness.hv_reference if hasattr(pf[0].fitness, "hv_reference") else np.array([1.0, 1.0])
        pf_points = sorted(
            [[1 - c_norm(s.complexity_, n), 1 - pseudo_accuracy(s.error_)] for s in pf],
            key=lambda p: p[0], reverse=True,
        )
        pf_fitnesses.append(pf_points)
        hvs.append(hypervolume(pf_points, ref))

    estimators = scores.pop("estimator")
    scores.pop("indices")
    scores["test_pf_fitness"] = pf_fitnesses
    scores["test_hypervolume"] = np.array(hvs) """
    return estimators, scores