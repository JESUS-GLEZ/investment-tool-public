# Strategies and backtesting

Investment Tool compares portfolio construction methods under the same data, dates and trading assumptions. Its investment mode proposes portfolio weights and trades for review. It does not place orders.

| Strategy | How it builds a portfolio |
| --- | --- |
| Buy and Hold | Holds one asset, such as SPY, as a reference for the comparison. |
| Momentum Risk Parity | Ranks assets by recent performance, screens for trend, and weights the selected assets by inverse historical volatility. This is a simple weighting rule; it does not solve for equal covariance-based risk contributions. |
| Optimization | Forecasts returns using an exponentially weighted mean by default, estimates covariance, and searches for long-only portfolios subject to a chosen range of holdings. It can balance risk contributions, capital weights, or both. |
| Sentiment EWM | Uses news sentiment in its return forecasts, then passes the forecasts to the portfolio optimizer. |
| Combined | Uses a regularized multivariate model with lagged returns and news features to forecast next-period returns, then optimizes the portfolio. |

Optimization searches for trade-offs among estimated return, variance and concentration. A selection rule chooses one of the portfolios found, for example by ranking estimated return relative to volatility or by targeting part of the discovered return range. A return target is a search setting; it does not promise a realized return. When transaction costs are enabled, the optimizer can account for estimated trading costs, and the backtest charges fees on the trades it simulates.

## Backtest design

**Information set and execution.** At each rebalance, the strategy builds its
signals from completed price bars and news available by the recorded decision
cutoff. The simulator uses opening quotes for the new period and marks the
portfolio at period end. Opening quote availability determines whether a
security can be traded. Earlier observations in the input window can be used
for estimation, so the input window and the evaluated period should be reported
separately.

**Historical eligibility.** A point-in-time study needs a dated universe and
coverage for securities that later exit or change ticker. A snapshot assembled
later can restrict the historic candidate set to surviving or readily available
symbols, even if the strategy respects the decision cutoff. Treat the saved
universe, coverage checks and corporate-action assumptions as part of the
experiment. A missing execution quote for an existing holding requires a valid
quote or settlement record.

**Estimating the effect of a change.** To test a news screen, balance rule or
return estimator, vary that component while matching the input snapshot,
realised evaluation periods, execution and cost assumptions, and the search
budget where applicable. For an optimiser study, also fix the forecast,
covariance estimate and constraints, then compare results across search seeds.
Assess configurations selected using historical results on a later period
reserved for evaluation.

## What the results measure

**Funding, costs and benchmarks.** The simulator applies the configured
proportional fee to the absolute value of buys and sells. Interpret net results
under that fee and the simulated fill prices. The Alpaca equity adapter uses
split-adjusted bars, so a buy-and-hold run based on those bars represents price
performance. Check the input snapshot's adjustment basis before comparing it
with an external total-return series. Separate yearly runs start with fresh
capital and positions. Chaining their returns omits the trades required to
transition between their starting and ending portfolios.

**Return and risk.** For runs with recorded calendar periods, CAGR annualises
the compound change in portfolio value over the elapsed calendar time. Sharpe
annualises the mean periodic excess return divided by the sample standard
deviation of periodic returns. Maximum drawdown measures the decline from a
running peak using recorded valuations, so monthly results can miss deeper
losses within a month. Check the recorded evaluation dates and frequency before
comparing these metrics across saved runs.

**The optimisation frontier.** Expected return, model variance and
concentration describe the opportunity set estimated at the rebalance. Capital
weight concentration measures the allocation across assets. Risk-contribution
concentration measures how estimated portfolio variance is attributed using the
covariance matrix. A relative return target interpolates between the returns
of the minimum-variance and maximum-return candidates found in that search.
Matching target fractions can therefore imply different expected returns.
The heuristic search approximates the constrained Pareto front. The convex
mean–variance reference relaxes the balance and holdings-count constraints, so a gap can reflect those constraints
as well as the quality of the search. Within the constrained model, compare
front coverage and the selected portfolio across restarts before attributing
differences to the strategy itself.

The [offline optimisation example](../examples/README.md#offline-optimisation-example)
uses synthetic inputs to illustrate the search and its outputs.
