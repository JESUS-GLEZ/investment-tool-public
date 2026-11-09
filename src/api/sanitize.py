"""Result sanitization: convert coordinator dicts to JSON-safe dicts.

Coordinator ``run()`` methods return dictionaries that may contain
pandas DataFrames, Series, Path objects, numpy scalars, and domain
objects.  This module provides a recursive converter that turns
everything into plain Python types suitable for ``json.dumps()`` /
Pydantic serialisation.
"""

from __future__ import annotations

import datetime
from pathlib import PurePath
from typing import Any
from investment_tool.output.policy import OutputMode

# --------------------------------------------------------------------------
# Optional heavy imports — we use runtime checks so the API layer does not
# hard-depend on numpy/pandas (they'll always be installed in practice, but
# guarding keeps the sanitizer unit-testable in isolation).
# --------------------------------------------------------------------------

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False


try:
    from platypus import Solution

    _HAS_PLATYPUS = True
except ImportError:  # pragma: no cover
    _HAS_PLATYPUS = False


def sanitize(obj: Any) -> Any:  # noqa: PLR0911 – many returns by design
    """Recursively convert *obj* to a JSON-safe Python structure.

    Conversion rules
    ----------------
    * ``pd.DataFrame`` → ``dict`` via :meth:`DataFrame.to_dict("list")`.
    * ``pd.Series`` → ``dict`` via :meth:`Series.to_dict`.
    * ``pd.Timestamp`` / ``pd.NaT`` → ISO-8601 string or ``None``.
    * ``np.integer`` / ``np.floating`` / ``np.bool_`` → Python native.
    * ``np.ndarray`` → nested Python ``list``.
    * ``np.nan`` / ``np.inf`` / ``np.NINF`` → ``None``.
    * ``Path`` / ``PurePath`` → ``str``.
    * ``datetime`` / ``date`` → ISO-8601 string.
    * ``dict`` → recursed.
    * ``list`` / ``tuple`` / ``set`` / ``frozenset`` → recursed list.
    * ``enum.Enum`` → ``.value``.
    * Other objects with a ``__dict__`` → ``{"__type__": cls, ...}``.
    * Everything else → ``str(obj)``.
    """
    # -- None / booleans / strings (fast path) --
    if obj is None or isinstance(obj, (bool, str)):
        return obj

    # -- numpy scalar types (must come before plain int/float checks since
    #    np.float64 is a subclass of float, but np.integer and np.bool_
    #    are NOT subclasses of int/bool) --
    if _HAS_NUMPY:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            if np.isnan(val) or np.isinf(val):
                return None
            return val

    # -- plain int / float --
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        import math

        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # -- pandas types --
    if _HAS_PANDAS:
        if isinstance(obj, pd.DataFrame):
            return {str(k): sanitize(v) for k, v in obj.to_dict("list").items()}
        if isinstance(obj, pd.Series):
            return {str(k): sanitize(v) for k, v in obj.to_dict().items()}
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat() if not pd.isna(obj) else None
        # NaT
        if obj is pd.NaT:
            return None

    # -- numpy ndarray --
    if _HAS_NUMPY:
        if isinstance(obj, np.ndarray):
            return [sanitize(x) for x in obj.tolist()]

    # -- pathlib --
    if isinstance(obj, PurePath):
        return str(obj)

    # -- datetime --
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    # -- collections --
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [sanitize(item) for item in obj]

    # -- enums --
    if hasattr(obj, "value") and hasattr(obj, "name"):
        return obj.value

    # Solutions reference the entire shared model. Serializing __dict__ repeats
    # its covariance matrix per portfolio and loses FixedLengthArray contents.
    if _HAS_PLATYPUS and isinstance(obj, Solution):
        result = {
            "__type__": "Solution",
            "variables": sanitize(list(obj.variables)),
            "objectives": sanitize(list(obj.objectives)),
            "constraints": sanitize(list(obj.constraints)),
            "constraint_violation": sanitize(obj.constraint_violation),
            "feasible": sanitize(getattr(obj, "feasible", False)),
            "evaluated": sanitize(obj.evaluated),
        }
        for key in ("rank", "crowding_distance"):
            if hasattr(obj, key):
                result[key] = sanitize(getattr(obj, key))
        return result

    # -- domain objects with __dict__ (Portfolio, etc.) --
    if hasattr(obj, "__dict__"):
        attrs = {
            str(k): sanitize(v)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
        attrs["__type__"] = type(obj).__name__
        return attrs

    # -- fallback --
    return str(obj)


def sanitize_output(
    raw_result: dict[str, Any], run_type: str, output_mode: "OutputMode"
) -> dict[str, Any]:
    """Project optional solver diagnostics after UI figures have been built.

    The same projection is used for completed, failed and cancelled results
    and checkpoints. Search objects and primary portfolio records are unchanged.
    """
    result: dict[str, Any] = sanitize(raw_result)
    if output_mode == OutputMode.STANDARD and run_type == "solver_comparison":
        for solver in result.get("results", []):
            solver.pop("all_archives", None)
        if result.get("reference_front") is not None:
            result["reference_front"] = [
                {"objectives": point["objectives"]}
                for point in result["reference_front"]
            ]
    return result
