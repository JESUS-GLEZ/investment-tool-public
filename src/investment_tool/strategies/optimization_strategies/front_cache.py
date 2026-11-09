"""Run-scoped, compact candidate fronts shared by selection-only variants.

Store sparse weights rather than problem matrices or Python pickles. Rebuild
and evaluate candidates against the identical current problem before reuse.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from platypus import Solution, Problem

from investment_tool.output.policy import OutputMode
from investment_tool.output.front_record import front_record, read_record, write_record

from .base_optimization_strategy import ProgressCallback
from .model.markowitz import MarkowitzLongOnly
from .optimization_result import OptimizationResult
from .seeded_optimization import MultiGreedyOptimizationStrategy


class CandidateFrontCache:
    def __init__(
        self,
        directory: Path,
        namespace: str,
        output_mode: OutputMode = OutputMode.DIAGNOSTIC,
    ) -> None:
        self.directory, self.namespace = directory, namespace
        self.output_mode = output_mode

    def _key(self, optimizer: MultiGreedyOptimizationStrategy) -> str:
        config = {
            "search_version": optimizer.search_version,
            "namespace": self.namespace,
            "date": optimizer.date_label,
            "assets": list(map(str, optimizer.expected_returns.index)),
            "covariance_rows": list(map(str, optimizer.covariance_matrix.index)),
            "covariance_columns": list(map(str, optimizer.covariance_matrix.columns)),
            "model": optimizer.model.__name__,
            "constraints": optimizer.model_kwargs,
            "population": optimizer.population_size,
            "evaluations": optimizer.n_generations,
            "restarts": optimizer.n_restarts,
            "seed": optimizer.random_seed,
            "seconds": optimizer.search_seconds,
            "seed_fraction": optimizer.seed_time_fraction,
            "subsets": optimizer.seed_max_subsets,
            "beam": optimizer.seed_beam_width,
            "refinement_supports": optimizer.refinement_supports,
        }
        if optimizer.trading_costs is not None:
            config["trading_costs"] = optimizer.trading_costs.record()
        digest = hashlib.sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode()
        )
        digest.update(np.asarray(optimizer.expected_returns, dtype="<f8").tobytes())
        digest.update(np.asarray(optimizer.covariance_matrix, dtype="<f8").tobytes())
        return digest.hexdigest()

    @staticmethod
    def _unpack(records: list[dict[str, Any]], problem: Problem) -> list[Solution]:
        front = []
        for record in records:
            solution = Solution(problem)
            weights = np.zeros(problem.nvars)
            for index, weight in record["weights"].items():
                weights[int(index)] = weight
            solution.variables[:] = weights.tolist()
            solution.evaluate()
            if not solution.feasible or not np.allclose(
                solution.objectives, record["objectives"], rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    "Shared candidate front does not match the current problem"
                )
            # Evaluation may normalize weights again. Retain the exact stored
            # portfolio and coordinates after validating it against this problem.
            solution.variables[:] = weights.tolist()
            solution.objectives[:] = record["objectives"]
            front.append(solution)
        return front

    def optimize(
        self,
        optimizer: MultiGreedyOptimizationStrategy,
        progress_callback: ProgressCallback | None,
    ) -> OptimizationResult:
        key = self._key(optimizer)
        path = self.directory / f"{key}.json.gz"
        legacy_path = self.directory / f"{key}.json"
        if not path.exists() and legacy_path.exists():
            path = legacy_path
        reused = path.exists()
        if reused:
            payload = read_record(path)
            if payload["key"] != key:
                raise ValueError("Candidate-front cache key mismatch")
            problem = optimizer.model(
                mu=optimizer.expected_returns.to_numpy(),
                cov=optimizer.covariance_matrix.to_numpy(),
                **optimizer.model_kwargs,
                **optimizer.cost_kwargs,
            )
            reference = payload.get("reference")
            result = OptimizationResult(
                front=self._unpack(payload["front"], problem),
                all_archives=[
                    self._unpack(a, problem) for a in payload.get("archives", [])
                ],
                reference_objectives=payload.get("reference_objectives"),
                markowitz_front=self._unpack(
                    reference,
                    MarkowitzLongOnly(
                        optimizer.expected_returns.to_numpy(),
                        optimizer.covariance_matrix.to_numpy(),
                        **optimizer.cost_kwargs,
                    ),
                )
                if reference is not None
                else None,
                search_diagnostics=payload["diagnostics"],
            )
            if progress_callback:
                progress_callback(
                    1, "Reusing identical candidate front for portfolio selection"
                )
        else:
            result = optimizer.optimize(progress_callback=progress_callback)
            self.directory.mkdir(parents=True, exist_ok=True)
            payload = {
                **front_record(result, self.output_mode),
                "key": key,
                "assets": list(map(str, optimizer.expected_returns.index)),
                "date_label": optimizer.date_label,
            }
            write_record(path, payload)

        result.search_diagnostics = {
            **result.search_diagnostics,
            "shared_front_key": key,
            "candidate_front_file": str(path),
            "reused_front": reused,
        }
        # The caller highlights its own selection; this is never cached.
        return result
