# Examples

## Small monthly backtest in the app

After the [Docker setup](../README.md#local-app-with-docker), sign in and open
**New Run → Backtest investment strategies**. Use the settings below; save them as a preset
if you want to repeat the run. This requires your own Alpaca credentials and data
entitlements. There is no bundled market dataset.

The [universe](../config/example_asset_universe.json) contains six stocks/ETFs
(SPY, QQQ, IEF, GLD, AAPL, MSFT) and BTC/USD. These illustrate equity ETFs, bonds,
gold, individual companies and crypto; overlapping equity exposures are deliberate
for a small software example. Bitcoin and gold are candidates, not required holdings.
`symbol_count` counts the six stock/ETF symbols; crypto is listed separately.

| Shared setting | Value |
| --- | --- |
| Data Source / Frequency | Alpaca / monthly |
| Asset universe | Small example — 7 assets (`example_asset_universe.json`) |
| Historical start / end month | January 2022 / December 2024 |
| Backtest start month | October 2024 |
| Initial cash | 1,000 |
| Transaction cost / risk-free rate | 0.1% per buy or sell / 0% |
| Validation / outputs | warn / standard |
| Include benchmark | SPY |

Remove the prefilled **Combined** configuration and edit the existing
**Optimization** row. Name it **Quickstart optimization**:

| Strategy setting | Value |
| --- | --- |
| Return estimator | EWM-6 |
| Portfolio balance / selection | Balanced risk contributions / Maximum Sharpe ratio |
| Min / max assets | 2 / 3 |
| News shortlist | Off |
| Population / evaluations per restart | 40 / 800 |
| Independent restarts / random seed | 1 / 0 |
| Maximum seed subsets / seed beam width | 64 / 3 |
| Weight refinement subsets | 0 |
| Fixed-work mode | On (no time cutoff) |

Review the configuration and launch. The run fetches 36 monthly periods for
warm-up and valuation but backtests only October–December 2024. Check the
validation report and individual result periods: missing provider coverage can
exclude candidates or prevent a run. Price **and news** retrieval happens first,
even though this configuration does not score sentiment. That initial download
can take minutes; optimiser budgets are deliberately small and are not research
defaults or a guarantee of optimal portfolios. Results, cached data and snapshots
remain in ignored local storage. No orders are placed.

The exact request is [monthly_backtest.json](monthly_backtest.json), validated
against `StrategyComparisonRequest` and its production adapter in the test suite.
It is a body for authenticated `POST /api/comparisons/strategy`, not a legacy
CLI `--use-json` file or a UI-importable preset. API clients can submit that body
using their normal authenticated session. For a first run, use the UI workflow
above; it handles authentication and lets you review every setting.

The fixed dates and assets demonstrate setup, not a point-in-time universe study,
investment recommendation or performance claim. The illustrative cost is not a
broker fee model. These settings are deliberately small so you can inspect a first run quickly.

## Offline optimisation example

Run the repository's actual Multi-Greedy + NSGA-II portfolio optimiser on six
explicitly synthetic assets. This entry point needs Python 3.11 and the small
dependency set below; it needs no API keys, dataset, database, broker or web server.
Package installation needs registry access. Execution after installation is offline.

### Run

From the repository root:

```sh
python3.11 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r examples/requirements.txt
python examples/offline_optimization.py
```

Direct dependency versions match `poetry.lock`; pip resolves their transitive
dependencies. This is not a fully locked standalone environment. For the existing
application lock, use `poetry install --only main,dev --no-root` and
`poetry run python examples/offline_optimization.py` instead.

The command writes these files into `examples/output/` (ignored by Git):

| File | Contents |
| --- | --- |
| `inputs.json` | Synthetic expected returns, covariance and model/search settings |
| `feasible_front.csv` | Candidate weights and independently reconstructed objectives |
| `manifest.json` | Input fingerprint, numerical-library versions, checks and illustrative selection |
| `front.png` | Estimated risk-return projection coloured by risk-contribution HHI |

Use `--output-dir /path/to/output` to write elsewhere; existing files with these
names are overwritten. Use `--seed 17` to set the default seed explicitly.

### What the example verifies

The model minimises variance, negative expected return and risk-contribution HHI,
with 2–3 long-only assets and bounds on estimated risk contributions. It uses
fixed subset/evaluation budgets and two independent NSGA-II restarts. Optional
support refinement and the relaxed mean-variance reference are disabled to keep
the example small. No application optimizer defaults or budgets are changed.

The script independently checks finite fully invested long-only weights,
cardinality, positive variance, risk-contribution bounds, reported objectives and
nondominance in all three objectives. It selects the discovered candidate with
maximum estimated return/volatility using a zero risk-free rate. This is a
model-based selection rule, not a realised Sharpe ratio.

The chart projects three objectives into two dimensions. Visually inferior
risk-return points may remain nondominated because of HHI. These inputs are
synthetic; no future prices, execution, turnover or costs are simulated.
This is not a historical backtest, evidence of alpha or a performance claim.
See the [strategy guide](../docs/strategies.md) for assumptions and limitations.

### Verification

The **Standalone offline example** CI job installs just these direct dependencies
and pinned pytest tools in a fresh environment on Ubuntu x86 with the repository's
pinned Python version. It checks dependency consistency, executes the CLI twice
in-process with Python sockets blocked, verifies the PNG/JSON/CSV exports, checks
the input fingerprint and reconstructs exported return/volatility. JSON and CSV
must agree between runs in that same environment; PNG bytes are not compared.
The test also runs in the regular locked application suite.

Fixed seeds do not guarantee cross-platform bitwise identity or global optimality.
CI's Python socket guard is not a sandbox for native libraries or subprocesses.
The workflow uses ordinary PR/main events, read-only permissions, pinned action
commits and no provider secrets, persisted credentials or dependency cache. It is
unfiltered so documentation changes also receive the check; compare results in the same numerical environment.
