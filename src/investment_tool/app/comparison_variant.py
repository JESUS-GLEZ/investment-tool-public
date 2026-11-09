"""Named configurations of existing comparison implementations."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


def validate_name(name: str) -> str:
    name = name.strip()
    # Result labels are also used in CSV filenames by the existing writers.
    if (
        not name
        or len(name) > 80
        or name in {".", ".."}
        or re.search(r'[\\/:*?"<>|\x00-\x1f]', name)
    ):
        raise ValueError(
            "Use a nonempty configuration name without path characters (up to 80 characters)"
        )
    return name


def validate_names(names: list[str]) -> None:
    normalized = [validate_name(name).casefold() for name in names]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Configuration names must be unique")


@dataclass(frozen=True)
class SolverVariant:
    name: str
    solver: str
    population_size: int | None = None
    n_generations: int | None = None
    n_restarts: int | None = None
    random_seed: int | None = None
    mutation_prob: float | None = None
    divisions_outer: int | None = None
    epsilons: list[float] | None = None

    def __post_init__(self) -> None:
        validate_name(self.name)
        for key, minimum in (
            ("population_size", 10),
            ("n_generations", 1),
            ("n_restarts", 1),
            ("random_seed", 0),
            ("divisions_outer", 1),
        ):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, int) or value < minimum):
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.random_seed is not None and self.random_seed > 4294967295:
            raise ValueError("random_seed must fit an unsigned 32-bit integer")
        if self.mutation_prob is not None and (
            not math.isfinite(self.mutation_prob) or not 0 <= self.mutation_prob <= 1
        ):
            raise ValueError("mutation_prob must lie between zero and one")
        if self.divisions_outer is not None and self.solver.upper() != "NSGA-III":
            raise ValueError("Reference divisions apply only to NSGA-III")
        if self.epsilons is not None:
            if (
                self.solver.upper() != "EPSNSGAII"
                or len(self.epsilons) != 2
                or any(not math.isfinite(x) or x <= 0 for x in self.epsilons)
            ):
                raise ValueError("EpsNSGAII requires two finite positive epsilons")


@dataclass(frozen=True)
class PredictorVariant:
    name: str
    predictor: str
    span: int | None = None
    lag_order: int | None = None
    alpha: float | None = None
    momentum_weight: float | None = None
    min_train_history: int | None = None
    partitions: int | None = None
    lags: int | None = None
    sigma: float | None = None
    beta: float | None = None
    trend: bool | None = None
    grid_points: int | None = None

    def __post_init__(self) -> None:
        validate_name(self.name)
        label = self.predictor.upper()
        fuzzy = label.startswith(("FUZZYTHESIS", "FUZZYRIDGE"))
        pooled = label == "POOLEDRIDGE"
        momentum = label in {"MOMENTUMRIDGE", "MOMENTUM12-2RIDGE"}
        ridge = label.startswith("RIDGEVARX")
        allowed = set()
        if label.startswith(("EWM-", "SENTIMENTEWM-", "EWM12-MEAN-")):
            allowed.add("span")
        if label.startswith("EWM12-MEAN-"):
            allowed.add("momentum_weight")
        if ridge:
            allowed.update(("lag_order", "alpha"))
        if pooled or label.startswith("FUZZYRIDGE") or momentum:
            allowed.update(("alpha", "min_train_history"))
        if fuzzy:
            allowed.update(("partitions", "lags", "sigma", "beta"))
        if label.startswith("FUZZYTHESIS"):
            allowed.update(("trend", "grid_points"))
        if pooled:
            allowed.add("lags")
        for key in self.__dataclass_fields__:
            if (
                key not in {"name", "predictor"}
                and getattr(self, key) is not None
                and key not in allowed
            ):
                raise ValueError(f"{key} is not supported by {self.predictor}")
        for key, minimum in (
            ("span", 1),
            ("lag_order", 1),
            ("min_train_history", 1),
            ("partitions", 2),
            ("lags", 1),
            ("grid_points", 3),
        ):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, int) or value < minimum):
                raise ValueError(f"{key} must be an integer >= {minimum}")
        for key in ("alpha", "sigma", "beta", "momentum_weight"):
            value = getattr(self, key)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if (
            self.alpha is not None
            and self.alpha <= 0
            or self.sigma is not None
            and self.sigma <= 0
            or self.beta is not None
            and self.beta < 0
        ):
            raise ValueError("alpha/sigma must be positive and beta nonnegative")
        if self.momentum_weight is not None and not 0 <= self.momentum_weight <= 1:
            raise ValueError("momentum_weight must lie between zero and one")
        minimum = self.min_train_history if self.min_train_history is not None else 12
        if (
            momentum
            and minimum < 12
            or (pooled or label.startswith("FUZZYRIDGE"))
            and minimum < (self.lags or 3)
        ):
            raise ValueError(
                "Training history is too short for this predictor’s features"
            )
