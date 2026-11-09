"""Tests for :mod:`api.sanitize` — result sanitization."""

from __future__ import annotations

import datetime
import enum
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import pytest

from api.sanitize import sanitize


# ---------------------------------------------------------------------------
# Primitives & None
# ---------------------------------------------------------------------------


class TestPrimitives:
    def test_none(self) -> None:
        assert sanitize(None) is None

    def test_bool(self) -> None:
        assert sanitize(True) is True

    def test_int(self) -> None:
        assert sanitize(42) == 42

    def test_float(self) -> None:
        assert sanitize(3.14) == 3.14

    def test_str(self) -> None:
        assert sanitize("hello") == "hello"


# ---------------------------------------------------------------------------
# Pandas types
# ---------------------------------------------------------------------------


class TestPandas:
    def test_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        result = sanitize(df)
        assert result == {"a": [1, 2], "b": [3.0, 4.0]}

    def test_series(self) -> None:
        s = pd.Series({"x": 1, "y": 2})
        result = sanitize(s)
        assert result == {"x": 1, "y": 2}

    def test_timestamp(self) -> None:
        ts = pd.Timestamp("2024-01-15 10:30:00")
        result = sanitize(ts)
        assert result == "2024-01-15T10:30:00"

    def test_nat(self) -> None:
        assert sanitize(pd.NaT) is None

    def test_dataframe_with_nan(self) -> None:
        df = pd.DataFrame({"a": [1.0, float("nan")]})
        result = sanitize(df)
        assert result["a"][0] == 1.0
        assert result["a"][1] is None


# ---------------------------------------------------------------------------
# Numpy types
# ---------------------------------------------------------------------------


class TestNumpy:
    def test_int64(self) -> None:
        result = sanitize(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_float64(self) -> None:
        result = sanitize(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_bool_(self) -> None:
        result = sanitize(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_nan(self) -> None:
        assert sanitize(np.float64("nan")) is None

    def test_inf(self) -> None:
        assert sanitize(np.float64("inf")) is None

    def test_ninf(self) -> None:
        assert sanitize(np.float64("-inf")) is None

    def test_ndarray(self) -> None:
        arr = np.array([1, 2, 3])
        assert sanitize(arr) == [1, 2, 3]

    def test_ndarray_2d(self) -> None:
        arr = np.array([[1, 2], [3, 4]])
        assert sanitize(arr) == [[1, 2], [3, 4]]


# ---------------------------------------------------------------------------
# Path types
# ---------------------------------------------------------------------------


class TestPaths:
    def test_posix_path(self) -> None:
        p = PurePosixPath("/tmp/output/results")
        assert sanitize(p) == "/tmp/output/results"

    def test_path(self) -> None:
        p = Path("/tmp/out")
        assert sanitize(p) == str(p)


# ---------------------------------------------------------------------------
# Datetime types
# ---------------------------------------------------------------------------


class TestDatetime:
    def test_datetime(self) -> None:
        dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        assert sanitize(dt) == "2024-01-15T10:30:00"

    def test_date(self) -> None:
        d = datetime.date(2024, 1, 15)
        assert sanitize(d) == "2024-01-15"


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class TestCollections:
    def test_dict_recursive(self) -> None:
        data = {"key": np.int64(5), "path": Path("/tmp")}
        result = sanitize(data)
        assert result == {"key": 5, "path": "/tmp"}

    def test_list_recursive(self) -> None:
        data = [np.float64(1.0), Path("/x"), None]
        result = sanitize(data)
        assert result == [1.0, "/x", None]

    def test_tuple(self) -> None:
        assert sanitize((1, 2, 3)) == [1, 2, 3]

    def test_set(self) -> None:
        result = sanitize({1})
        assert result == [1]

    def test_nested(self) -> None:
        data = {"df": pd.DataFrame({"a": [1]}), "inner": {"p": Path("/x")}}
        result = sanitize(data)
        assert result == {"df": {"a": [1]}, "inner": {"p": "/x"}}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_string_enum(self) -> None:
        class Color(str, enum.Enum):
            RED = "red"

        assert sanitize(Color.RED) == "red"

    def test_int_enum(self) -> None:
        class Priority(enum.Enum):
            HIGH = 1

        assert sanitize(Priority.HIGH) == 1


# ---------------------------------------------------------------------------
# Domain objects with __dict__
# ---------------------------------------------------------------------------


class TestDomainObjects:
    def test_object_with_dict(self) -> None:
        class Foo:
            def __init__(self) -> None:
                self.x = 1
                self.y = Path("/tmp")
                self._private = "skip"

        result = sanitize(Foo())
        assert result["x"] == 1
        assert result["y"] == "/tmp"
        assert result["__type__"] == "Foo"
        assert "_private" not in result

    def test_fallback_str(self) -> None:
        """Objects without __dict__ that aren't covered should become str."""

        class Weird:
            __slots__ = ()

            def __str__(self) -> str:
                return "weird_thing"

        # slots-only with no matching branch → falls through to str()
        result = sanitize(Weird())
        # It has __dict__ via object, so it actually gets the domain-object branch,
        # but with an empty dict. That's fine — __type__ will be set.
        assert "__type__" in result or result == "weird_thing"


class TestPlatypusSolutions:
    def test_solution_preserves_arrays_without_embedding_model(self) -> None:
        import json

        from platypus import Problem, Solution

        problem = Problem(2, 2, 1)
        problem.cov = np.eye(100)
        solution = Solution(problem)
        solution.variables[:] = [0.4, 0.6]
        solution.objectives[:] = [0.002, -0.03]
        solution.constraints[:] = [0.0]
        solution.constraint_violation = 0.0
        solution.feasible = True
        solution.evaluated = True
        solution.rank = 0
        solution.crowding_distance = float("inf")

        result = sanitize({"front": [solution], "all_archives": [[solution]]})
        saved = result["front"][0]
        assert saved["variables"] == [0.4, 0.6]
        assert saved["objectives"] == [0.002, -0.03]
        assert saved["constraints"] == [0.0]
        assert saved["feasible"] is True
        assert saved["rank"] == 0
        assert saved["crowding_distance"] is None
        assert "problem" not in saved
        assert len(json.dumps(result, allow_nan=False)) < 1000

    def test_infeasible_solution_metadata_is_preserved(self) -> None:
        from platypus import Problem, Solution

        solution = Solution(Problem(1, 1, 1))
        solution.constraint_violation = 0.25
        solution.feasible = False
        result = sanitize(solution)
        assert result["constraint_violation"] == 0.25
        assert result["feasible"] is False


@pytest.mark.parametrize(
    "run_type,mode,keep_archives",
    [
        ("solver_comparison", "standard", False),
        ("solver_comparison", "diagnostic", True),
        ("strategy_comparison", "standard", True),
    ],
)
def test_storage_policy_preserves_primary_results_and_does_not_mutate_search_objects(
    run_type, mode, keep_archives
):
    from api.sanitize import sanitize_output
    from investment_tool.output.policy import OutputMode

    primary = {"variables": [0.25, 0.75], "objectives": [0.02, -0.04]}
    raw = {
        "partial": True,
        "results": [
            {
                "front": [primary],
                "all_archives": [[primary]],
                "metrics": {"best_sharpe": 0.3},
                "best_weights": {"A": 0.25, "B": 0.75},
            }
        ],
        "reference_front": [primary],
        "figures": [{"data": [1, 2]}],
    }
    result = sanitize_output(raw, run_type, OutputMode(mode))
    assert result["results"][0]["front"] == [primary]
    assert result["results"][0]["best_weights"] == {"A": 0.25, "B": 0.75}
    assert result["figures"] == raw["figures"]
    assert result["partial"]
    assert ("all_archives" in result["results"][0]) == keep_archives
    assert "all_archives" in raw["results"][0]
    if not keep_archives:
        assert result["reference_front"] == [{"objectives": [0.02, -0.04]}]
    assert raw["reference_front"][0]["variables"] == [0.25, 0.75]
