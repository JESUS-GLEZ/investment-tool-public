"""Migration adds timing without guessing or destroying old cached bars."""

import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


def test_upgrade_and_downgrade_preserve_existing_cache_rows():
    path = Path(__file__).parents[2] / "migrations/versions/0006_bar_time_contract.py"
    spec = importlib.util.spec_from_file_location("bar_time_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE ohlcv_cache (id INTEGER PRIMARY KEY, close FLOAT)")
        )
        connection.execute(text("INSERT INTO ohlcv_cache VALUES (1, 123.5)"))
        with Operations.context(MigrationContext.configure(connection)):
            module.upgrade()
            assert "timing" in {
                c["name"] for c in inspect(connection).get_columns("ohlcv_cache")
            }
            assert connection.execute(
                text("SELECT close, timing FROM ohlcv_cache")
            ).one() == (123.5, None)
            module.downgrade()
            assert "timing" not in {
                c["name"] for c in inspect(connection).get_columns("ohlcv_cache")
            }
            assert (
                connection.execute(text("SELECT close FROM ohlcv_cache")).scalar_one()
                == 123.5
            )
