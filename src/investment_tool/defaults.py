"""Canonical default values for all tunable parameters.

This module is the **single source of truth** for every tunable parameter
in the investment tool.  Every other layer — CLI argparse, JSON presets,
coordinator dataclasses, strategy/optimizer constructors — should import
its defaults from here rather than re-declaring its own.

Design
------
Parameters are grouped into frozen dataclasses by functional area.  Each
group is instantiated as a module-level singleton so that consumers can
import either the group object or individual values:

    from investment_tool.defaults import PORTFOLIO, OPTIMIZER
    pop_size = OPTIMIZER.pop_size

    # or flat access:
    from investment_tool.defaults import OPTIMIZER
    print(OPTIMIZER.n_generations)

The frozen-dataclass approach ensures:
- Type annotations and IDE autocompletion.
- Immutability — accidental mutation is caught at runtime.
- Single place to document valid ranges and rationale.

Inconsistency resolutions
-------------------------
Where the same parameter previously had different values in different
layers, the canonical value was chosen as follows:

- ``pop_size = 300``: constants.py value.  CLI/dataclass had 100, which
  was a quick-test shortcut that leaked into production defaults.
- ``n_generations = 500_000``: constants.py value.  CLI=10K and JSON=50K
  were debug shortcuts.
- ``n_restarts = 5``: constants.py value.  The solver comparison's 3 is
  kept as an override in that coordinator (comparison runs are expensive).
- ``risk_aversion = 0.1``: orchestrator default.  Solver comparison's 0.5
  is a deliberate override for that benchmark context.
- ``ewm_span = 6``: used everywhere in practice.  SentimentEWM class
  default of 3 is updated to 6 for consistency; the prediction comparison
  still offers EWM-3 as an explicit variant.
- ``dust_tol = 1e-4``: aligns with ``activity_tol`` in BPRC.  The
  ``repair_portfolio`` default of 1e-3 was too aggressive.
- ``hhi_as_constraint = False``: orchestrator default (3-objective mode).
  Solver comparison's True is a deliberate override.
- ``compute_sentiment = True``: the intended default; JSON had false as a
  shortcut for local testing.
- ``data_source = "alpaca"``: production default.  JSON's "local" was a
  dev convenience.
- ``strategy = "combined"``: the flagship strategy.  JSON's "optimization"
  was a debug shortcut.
- ``mode = "backtesting"``: most common usage.  CLI's "investing" default
  is changed to "backtesting".
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Portfolio & Backtesting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioDefaults:
    """Defaults for portfolio construction and backtesting."""

    initial_cash: float = 1000.0
    """Starting portfolio value in the base currency."""

    transaction_cost_rate: float = 0.001
    """Transaction cost as a fraction of trade value (10 bps)."""

    risk_free_rate: float = 0.0
    """Annual risk-free rate for Sharpe ratio calculation."""

    min_backtest_periods: int = 5
    """Minimum number of historical periods required to run a backtest."""

    start_index_percentage: float = 0.5
    """Fraction of historical data used for the warm-up window."""


PORTFOLIO = PortfolioDefaults()


# ---------------------------------------------------------------------------
# Strategy Selection (enum-string defaults for CLI / coordinator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategySelectionDefaults:
    """Defaults for strategy / mode / source selection."""

    strategy: str = "combined"
    """Default investment strategy type."""

    optimizer_type: str = "evolutionary"
    """Default optimizer backend (evolutionary / decomposition / hybrid)."""

    data_source: str = "alpaca"
    """Default data source for market data."""

    mode: str = "backtesting"
    """Default execution mode."""

    frequency: str = "monthly"
    """Default data frequency."""


STRATEGY_SELECTION = StrategySelectionDefaults()


# ---------------------------------------------------------------------------
# Portfolio Construction (asset constraints & risk preferences)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioConstructionDefaults:
    """Defaults for portfolio structure and risk parameters."""

    min_assets: int = 2
    """Minimum number of active assets in the portfolio."""

    max_assets: int = 5
    """Maximum number of active assets in the portfolio."""

    risk_aversion: float = 0.1
    """Risk aversion coefficient (0 = risk-neutral, 1 = very risk-averse)."""

    hhi_as_constraint: bool = False
    """If True, move HHI from objective to constraint (2-obj mode)."""

    hhi_bound: float = 0.5
    """Upper bound for HHI constraint (only used when hhi_as_constraint=True)."""


CONSTRUCTION = PortfolioConstructionDefaults()


# ---------------------------------------------------------------------------
# Optimizer Core
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizerDefaults:
    """Defaults for the multi-objective evolutionary optimizer."""

    pop_size: int = 300
    """Population size for evolutionary algorithms."""

    n_generations: int = 500_000
    """Maximum function evaluations per optimizer restart."""

    n_restarts: int = 5
    """Number of independent multi-start runs."""

    ea_budget_ratio: float = 0.3
    """Fraction of budget allocated to EA phase in hybrid optimisation."""

    constraint_slack: float = 0.1
    """Relaxation for risk-contribution bounds in BPRC."""

    activity_tol: float = 1e-4
    """Weight threshold below which an asset is considered inactive."""

    dust_tol: float = 1e-4
    """Weight threshold for dust-position cleanup (aligned with activity_tol)."""

    warm_start_generation_ratio: float = 0.7
    """Fraction of n_generations used on warm-started iterations."""


OPTIMIZER = OptimizerDefaults()


# ---------------------------------------------------------------------------
# Genetic Operators (default variator)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneticOperatorDefaults:
    """Defaults for the standard PortfolioVariator."""

    mutation_prob: float = 0.5
    """Probability of applying mutation to an offspring."""

    perturb_prob: float = 0.7
    """Probability that mutation is weight perturbation (vs cardinality change)."""

    cardinality_prob: float = 0.15
    """Probability that mutation is a cardinality change (add/drop asset)."""

    noise_std: float = 0.05
    """Standard deviation of Gaussian perturbation noise."""


GENETIC = GeneticOperatorDefaults()


# ---------------------------------------------------------------------------
# Gentle Variator (less aggressive mutation for exploitation-heavy islands)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GentleVariatorDefaults:
    """Defaults for the 'gentle' variator variant.

    Used on islands that benefit from less aggressive mutation: more
    perturbation, less cardinality change, tighter noise.
    """

    mutation_prob: float = 0.6
    perturb_prob: float = 0.85
    cardinality_prob: float = 0.05
    noise_std: float = 0.03


GENTLE_VARIATOR = GentleVariatorDefaults()


# ---------------------------------------------------------------------------
# Decomposition Optimizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecompositionDefaults:
    """Defaults for the decomposition-based SLSQP optimizer."""

    n_directions: int = 100
    """Number of scalarized subproblems (reference directions)."""

    n_random_starts: int = 3
    """Random restarts per subproblem."""

    maxiter: int = 500
    """Maximum SLSQP iterations per start."""

    n_decomposition_directions: int = 80
    """Reference directions used in hybrid mode's SLSQP phase."""


DECOMPOSITION = DecompositionDefaults()


# ---------------------------------------------------------------------------
# Algorithm Portfolio (island model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgorithmPortfolioDefaults:
    """Defaults for NSGA-III / EpsNSGAII algorithm-specific parameters."""

    divisions_outer: int = 12
    """Number of outer divisions for NSGA-III reference directions."""

    epsilons: tuple[float, ...] = (0.001, 0.001)
    """Epsilon values for EpsNSGAII archive."""


ALGORITHM = AlgorithmPortfolioDefaults()


# ---------------------------------------------------------------------------
# Markowitz Benchmark (lighter settings for the diagnostic frontier)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkowitzDefaults:
    """Defaults for the Markowitz efficient-frontier computation.

    Markowitz is a 2-objective unconstrained problem — it converges much
    faster than the 3-objective constrained BPRC, so it uses its own
    lighter settings.
    """

    n_evaluations: int = 20_000
    """Function evaluations per restart."""

    pop_size: int = 300
    """GA population size."""

    n_restarts: int = 3
    """Independent multi-start runs."""


MARKOWITZ = MarkowitzDefaults()


# ---------------------------------------------------------------------------
# Prediction Strategies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionDefaults:
    """Defaults for return prediction strategies."""

    ewm_span: int = 6
    """EWM span in periods (span 6 gives a half-life of about 2.06 periods)."""

    fallback_tilt: float = 0.0
    """Sentiment tilt when regression cannot be fitted (pure EWM fallback)."""

    lag_order: int = 3
    """Number of lagged return periods for Ridge VARX."""

    ridge_alpha: float = 10.0
    """Ridge regularisation strength for VARX."""

    min_regression_periods: int = 12
    """Minimum aligned rows required to fit the sentiment regression."""


PREDICTION = PredictionDefaults()


# ---------------------------------------------------------------------------
# Momentum / Risk-Parity Strategy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MomentumDefaults:
    """Defaults for the momentum-risk-parity strategy."""

    top_n: int = 10
    """Number of assets to hold after momentum ranking."""

    lookback: int = 6
    """Trailing periods for momentum score."""

    vol_window: int = 6
    """Rolling window for volatility estimation."""

    trend_window: int = 10
    """SMA window for the trend filter."""

    min_weight: float = 0.02
    """Minimum position weight; below this → zeroed and redistributed."""


MOMENTUM = MomentumDefaults()


# ---------------------------------------------------------------------------
# Data Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationDefaults:
    """Defaults for data quality validators."""

    max_missing_pct: float = 0.3
    """Maximum fraction of missing data before a symbol is dropped."""

    max_consecutive_missing: int = 5
    """Maximum consecutive NaN rows before a symbol is flagged."""

    min_periods: int = 30
    """Minimum number of historical rows required per symbol."""


VALIDATION = ValidationDefaults()


# ---------------------------------------------------------------------------
# Infrastructure / API Client
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InfraDefaults:
    """Defaults for API clients and infrastructure."""

    rate_limit_sleep: float = 0.01
    """Seconds to sleep between API calls to avoid rate limiting."""

    news_limit: int = 50
    """Maximum news articles to fetch per symbol per period."""


INFRA = InfraDefaults()


# ---------------------------------------------------------------------------
# Covariance Matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CovarianceDefaults:
    """Defaults for covariance matrix estimation."""

    ewm_span: int = 6
    """EWM span for the exponentially-weighted covariance matrix."""

    shrinkage: float = 0.1
    """Shrink correlations 10% toward independence; preserve asset variances."""


COVARIANCE = CovarianceDefaults()


# ---------------------------------------------------------------------------
# Warm-Start Seed Noise Scales
# ---------------------------------------------------------------------------

# Perturbation noise scales for seeding across restarts — earlier restarts
# exploit the previous front closely, later restarts explore more broadly.
SEED_NOISE_SCALES: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.15)


@dataclass(frozen=True)
class StrategyOptimizerDefaults:
    """Fixed-work search per rebalance; see docs/strategies.md."""

    pop_size: int = 300
    n_generations: int = 6_000
    n_restarts: int = 6
    random_seed: int = 0
    seed_max_subsets: int = 360_000
    seed_beam_width: int = 5
    refinement_supports: int = 80
    search_seconds: float | None = None
    seed_time_fraction: float = 0.4


STRATEGY_OPTIMIZER = StrategyOptimizerDefaults()
