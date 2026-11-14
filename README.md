# Portfolio Research Lab

Research and compare long-only portfolio strategies. The app records data windows,
portfolio weights, simulated trades and performance so runs can be compared and
inspected. Investment mode proposes a rebalance; it does not place orders.

## Watch the demo

This three-minute walkthrough follows a historical portfolio research campaign
from strategy configuration through backtest results, monthly allocations and
optimisation diagnostics. It includes English subtitles.

## Try it without credentials

Run the [synthetic optimizer example](examples/README.md#offline-optimisation-example)
with Python 3.11. It runs the portfolio search without a provider account,
database or Docker. The synthetic outputs illustrate trade-offs and feasibility.

## Local app with Docker

Install Docker Engine 28+ with Docker Compose v2 and Python 3.11+. Run from the
repository root with Docker running:

```sh
python3 tools/setup_local.py
python3 tools/setup_local.py --check
mkdir -p data/input data/output
docker compose -f docker-compose.prod.yml build api worker nginx
docker compose -f docker-compose.prod.yml up -d --wait postgres redis
docker compose -f docker-compose.prod.yml run --rm --no-deps api alembic upgrade head
docker compose -f docker-compose.prod.yml up -d api worker nginx
```

Open [localhost:8080](http://localhost:8080/) and register a local account.
The setup helper generates local database and session secrets. To fetch market
data, use `python3 tools/setup_local.py --alpaca` before starting the services,
or follow the restart instructions printed by the helper when adding credentials
later. The prompt hides the key and secret. Keep the generated `.env` private.
The supplied stack binds published ports to loopback for local use.

Follow the [small monthly backtest](examples/README.md#small-monthly-backtest-in-the-app)
using your own Alpaca credentials. The default universe has seven illustrative
candidates: SPY, QQQ, IEF, GLD, AAPL, MSFT and BTC/USD. Gold and Bitcoin are
optional holdings. To try another universe, copy and edit the
[example JSON](config/example_asset_universe.json) under `config/`.

The app does not bundle market data. It can also read compatible local snapshots;
load only pickle files you created or otherwise trust. Generated data and results
stay local. Stop the stack with `docker compose -f docker-compose.prod.yml down`;
keep the volumes to retain accounts and saved runs. Keep your existing `.env`
when updating an installation with saved data.

## How it works

The [strategy guide](docs/strategies.md) explains the models, trading assumptions
and limitations. The Python engine lives in `src/investment_tool/`, the FastAPI
backend and worker in `src/api/`, and the React app in `frontend/`.
See [development and checks](docs/testing.md) for the host-development path and
verification commands.

## License

Copyright © 2024–2026 Jesus Gonzalez.

The project source is licensed under the [GNU General Public License, version 3](LICENSE)
(GPL-3.0-only). See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream
components used by the app.
