# Development and checks

For a complete local installation, follow the [Docker quickstart](../README.md#local-app-with-docker).
The root `docker-compose.yml` supplies backing services for contributors who run
the API, worker and frontend on the host. `docker-compose.prod.yml` runs the full
stack. Use distinct Compose projects and avoid conflicting loopback ports.

## Development setup

For host development, start the local services from the repository root. Stop
the full-stack project first if it already occupies the loopback ports.

```sh
docker compose -p investment-tool-dev up -d --wait postgres redis
poetry install --only main,dev
export DATABASE_URL=postgresql://investment_tool:investment_tool@localhost:5432/investment_tool
export REDIS_URL=redis://localhost:6379/0
export ENV=development
poetry run alembic upgrade head
```

Run `poetry run uvicorn api.main:app --reload` and
`poetry run celery -A api.worker.celery_app worker --loglevel=info` in separate
terminals with those same development variables. In `frontend/`, use `nvm use`,
`npm ci --ignore-scripts --no-audit --no-fund`, and `npm run dev`, then open
[localhost:5173](http://localhost:5173/). These credentials are local development
defaults for the isolated backing services. They are never production secrets.

## Local verification

Use Python 3.11, Poetry, and the Node version in `frontend/.nvmrc`.
From the repository root:

```sh
poetry check --lock
poetry install --only main,dev --no-root
poetry run python -m pip check
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy
poetry run pytest -q -m 'not slow' --disable-socket --allow-unix-socket
```

For changes to the optimizer or other paths exercised by the slow integration
suite, also run `poetry run pytest -q -m slow --disable-socket --allow-unix-socket`.

```sh
cd frontend
nvm use
npm ci --ignore-scripts --no-audit --no-fund
npm run lint
npm test
npm run build
```

Tests use synthetic inputs and mocks; do not put provider credentials in fixtures.
The socket guard is a Python test check, not a sandbox. Try the
[standalone example](../examples/README.md#offline-optimisation-example) to verify
the small dependency installation separately.
