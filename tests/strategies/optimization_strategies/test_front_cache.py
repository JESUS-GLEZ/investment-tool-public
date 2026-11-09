import json
import gzip
from unittest.mock import patch

import numpy as np
import pytest

from investment_tool.strategies.optimization_strategies.front_cache import (
    CandidateFrontCache,
)
from tests.strategies.optimization_strategies.test_seeded_optimization import (
    make_optimizer,
)


def test_reuse_skips_search_but_returns_independent_revalidated_solutions(tmp_path):
    first = make_optimizer()
    first.search_seconds = None
    cache = CandidateFrontCache(tmp_path, "optimization")
    original = cache.optimize(first, None)
    second = make_optimizer()
    second.search_seconds = None
    with patch.object(
        second, "optimize", side_effect=AssertionError("must share front")
    ):
        reused = cache.optimize(second, None)
    assert reused.search_diagnostics["reused_front"]
    assert (
        original.search_diagnostics["shared_front_key"]
        == reused.search_diagnostics["shared_front_key"]
    )
    assert all(s.feasible for s in reused.front)
    np.testing.assert_allclose(
        [s.objectives[:] for s in original.front],
        [s.objectives[:] for s in reused.front],
        rtol=1e-12,
        atol=1e-14,
    )
    old = original.front[0].variables[0]
    reused.front[0].variables[0] = 0.987
    assert original.front[0].variables[0] == old
    assert max(p.stat().st_size for p in tmp_path.iterdir()) < 100_000


def test_holdings_rates_and_forced_liquidations_invalidate_cached_front(tmp_path):
    from investment_tool.domain.trading_costs import TradingCosts

    cache = CandidateFrontCache(tmp_path, "optimization")
    optimizer = make_optimizer()
    keys = [cache._key(optimizer)]
    for rate, weights, outside in [
        (0.001, [0.5, 0.5, 0, 0], 0),
        (0.001, [0.6, 0.4, 0, 0], 0),
        (0.002, [0.5, 0.5, 0, 0], 0),
        (0.001, [0.4, 0.4, 0, 0], 0.2),
    ]:
        optimizer.trading_costs = TradingCosts(rate, np.array(weights), outside)
        keys.append(cache._key(optimizer))
    assert len(set(keys)) == len(keys)
    optimizer.search_seconds = None
    original = cache.optimize(optimizer, None)
    with patch.object(
        optimizer,
        "optimize",
        side_effect=AssertionError("should reuse identical cost state"),
    ):
        reused = cache.optimize(optimizer, None)
    np.testing.assert_allclose(
        [s.objectives[:] for s in original.front],
        [s.objectives[:] for s in reused.front],
    )


@pytest.mark.parametrize(
    "setting,value",
    [
        ("n_restarts", 1),
        ("random_seed", 37),
        ("population_size", 30),
        ("seed_max_subsets", 9),
        ("seed_beam_width", 7),
        ("refinement_supports", 2),
        ("search_version", "next-search-version"),
        ("n_generations", 100),
    ],
)
def test_different_search_configuration_cannot_reuse_front(tmp_path, setting, value):
    cache = CandidateFrontCache(tmp_path, "optimization")
    first = make_optimizer()
    second = make_optimizer()
    setattr(second, setting, value)
    assert cache._key(first) != cache._key(second)


def test_changed_returns_covariance_asset_order_or_period_invalidate_front(tmp_path):
    cache = CandidateFrontCache(tmp_path, "optimization")
    optimizer = make_optimizer()
    key = cache._key(optimizer)
    optimizer.expected_returns = -optimizer.expected_returns
    assert key != cache._key(optimizer)
    optimizer = make_optimizer()
    optimizer.covariance_matrix *= 2
    assert key != cache._key(optimizer)
    optimizer = make_optimizer()
    optimizer.expected_returns.index = list("DCBA")
    assert key != cache._key(optimizer)
    optimizer = make_optimizer()
    optimizer.date_label = "2026-09-01"
    assert key != cache._key(optimizer)


def test_corrupt_front_fails_instead_of_running_an_unpaired_search(tmp_path):
    optimizer = make_optimizer()
    optimizer.search_seconds = None
    cache = CandidateFrontCache(tmp_path, "optimization")
    cache.optimize(optimizer, None)
    path = next(tmp_path.glob("*.json.gz"))
    payload = json.loads(gzip.decompress(path.read_bytes()))
    payload["front"][0]["objectives"][1] = -1000
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="does not match"):
        cache.optimize(optimizer, None)


def test_convex_mv_reference_survives_front_reuse(tmp_path):
    optimizer = make_optimizer()
    optimizer.search_seconds = None
    optimizer.include_markowitz_front = True
    cache = CandidateFrontCache(tmp_path, "optimization")
    original = cache.optimize(optimizer, None)
    assert original.markowitz_front
    with patch.object(optimizer, "optimize", side_effect=AssertionError("no search")):
        reused = cache.optimize(optimizer, None)
    assert reused.markowitz_front
    assert all(s.feasible for s in reused.markowitz_front)
    np.testing.assert_allclose(
        [s.objectives[:] for s in original.markowitz_front],
        [s.objectives[:] for s in reused.markowitz_front],
        rtol=1e-12,
        atol=1e-14,
    )


def test_legacy_uncompressed_cache_is_revalidated_without_rewriting(tmp_path):
    from investment_tool.output.front_record import read_record

    optimizer = make_optimizer()
    optimizer.search_seconds = None
    cache = CandidateFrontCache(tmp_path, "optimization")
    original = cache.optimize(optimizer, None)
    compressed = next(tmp_path.glob("*.json.gz"))
    legacy = compressed.with_suffix("")
    legacy.write_text(json.dumps(read_record(compressed)))
    compressed.unlink()
    previous_bytes = legacy.read_bytes()
    with patch.object(
        optimizer, "optimize", side_effect=AssertionError("no new search")
    ):
        reused = cache.optimize(optimizer, None)
    np.testing.assert_array_equal(
        [list(s.variables) for s in original.front],
        [list(s.variables) for s in reused.front],
    )
    assert legacy.read_bytes() == previous_bytes
