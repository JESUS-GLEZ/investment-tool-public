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

## What we learned while building it

- **Timing is part of the model.** Price bars and news must be available at the decision time. A backtest that uses a closing price to make a decision cannot also assume it traded at that same close. The execution and valuation timestamps should be auditable.
- **The candidate universe matters.** Reconstructing historical membership or excluding securities based on later coverage can bias results. Small changes to the candidate set can also lead a heuristic optimizer to discover different portfolios. Keep universes fixed when comparing strategies and record their provenance.
- **Balance has a specific meaning.** Balanced estimated risk contributions can coexist with unequal capital weights, shared sector exposures and large losses. Evaluate realized drawdown and trading costs alongside the model's estimated risk.
- **Forecast quality and search quality are separate.** A better portfolio search cannot correct optimistic inputs. Compare against simple baselines, check that selected portfolios are feasible, and evaluate forecasts and portfolio results on periods that were not used to choose settings.

## Reading results

Use identical data windows, execution assumptions and fees when comparing configurations. Report the benchmark's dividend treatment, distinguish period-end drawdowns from intraperiod losses, and distinguish independently restarted backtests from a continuously held portfolio. The optimizer returns a discovered set of portfolios and does not certify a global optimum.

The bundled asset list is for a small software demonstration. The standalone optimizer example uses synthetic inputs. Neither is evidence of investment performance.
