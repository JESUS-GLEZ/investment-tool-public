"""Fresh local credentials stay private and existing installations stay intact."""

import importlib.util
import os
import getpass
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/setup_local.py"
SPEC = importlib.util.spec_from_file_location("setup_local", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def env_values(path):
    return dict(
        line.split("=", 1)
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    )


def test_fresh_independent_credentials_match_the_database_url(tmp_path):
    first, second = tmp_path / "one.env", tmp_path / "two.env"
    setup.create_local_env(first)
    setup.create_local_env(second)
    one, two = env_values(first), env_values(second)
    url = urlsplit(one["DATABASE_URL"])
    assert url.hostname == "postgres"
    assert url.username == one["POSTGRES_USER"]
    assert url.password == one["POSTGRES_PASSWORD"]
    assert url.path == f"/{one['POSTGRES_DB']}"
    assert len(one["POSTGRES_PASSWORD"]) >= 32
    assert len(one["JWT_SECRET_KEY"]) >= 64
    assert one["JWT_SECRET_KEY"] != one["POSTGRES_PASSWORD"]
    assert one["JWT_SECRET_KEY"] != two["JWT_SECRET_KEY"]
    assert one["POSTGRES_PASSWORD"] != two["POSTGRES_PASSWORD"]
    assert all(
        not one[key]
        for key in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPHA_VANTAGE_API_KEY")
    )
    if os.name == "posix":
        assert stat.S_IMODE(first.stat().st_mode) == 0o600


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_files_and_symlinks_cannot_be_overwritten(tmp_path, symlink):
    target = tmp_path / "existing.env"
    target.write_text("SYNTHETIC_EXISTING_CONFIGURATION=keep\n")
    destination = tmp_path / "link.env" if symlink else target
    if symlink:
        destination.symlink_to(target)
    with pytest.raises(FileExistsError):
        setup.create_local_env(destination)
    assert target.read_text() == "SYNTHETIC_EXISTING_CONFIGURATION=keep\n"


def test_cli_prints_no_credentials_and_refuses_regeneration(tmp_path):
    destination = tmp_path / "new.env"
    command = [sys.executable, str(SCRIPT), "--output", str(destination)]
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    original = destination.read_bytes()
    values = env_values(destination)
    for key in ("JWT_SECRET_KEY", "POSTGRES_PASSWORD", "DATABASE_URL"):
        assert values[key] not in first.stdout + first.stderr
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode == 1
    assert "Refusing to replace" in again.stderr
    assert destination.read_bytes() == original


def test_environment_variants_are_ignored_but_example_remains_shareable():
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=ROOT,
        input=".env\n.env.local\n.env.production\nfrontend/.env.local\n.env.example\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        ".env",
        ".env.local",
        ".env.production",
        "frontend/.env.local",
    ]


def test_local_dataset_files_are_ignored_outside_the_data_directory():
    paths = ["local_snapshot.pkl", "scratch/prices.pickle", "scratch/news.parquet"]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=ROOT,
        input="\n".join(paths) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == paths


def test_hidden_prompt_creates_fresh_config_without_printing_credentials(
    tmp_path, monkeypatch, capsys
):
    destination = tmp_path / "new.env"
    credentials = iter(["SYNTHETIC_KEY_123", "SYNTHETIC_SECRET_456"])
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--output", str(destination), "--alpaca"]
    )
    monkeypatch.setattr(setup.getpass, "getpass", lambda prompt: next(credentials))
    assert setup.main() == 0
    output = capsys.readouterr()
    assert "SYNTHETIC_KEY_123" not in output.out + output.err
    assert "SYNTHETIC_SECRET_456" not in output.out + output.err
    values = env_values(destination)
    assert values["ALPACA_API_KEY"] == "'SYNTHETIC_KEY_123'"
    assert values["ALPACA_API_SECRET"] == "'SYNTHETIC_SECRET_456'"
    assert "CHANGE_ME" not in values["JWT_SECRET_KEY"]


def test_alpaca_update_preserves_unrelated_values_comments_and_secrets(tmp_path):
    destination = tmp_path / "existing.env"
    setup.create_local_env(destination)
    before = env_values(destination)
    with destination.open("a") as file:
        file.write("# Keep this operator setting\nCUSTOM_SETTING=unchanged\n")
    setup.configure_alpaca(
        destination, "synthetic-key", "literal$DOLLAR${NAME}#percent%"
    )
    after = env_values(destination)
    for name in before.keys() - {"ALPACA_API_KEY", "ALPACA_API_SECRET"}:
        assert after[name] == before[name]
    assert after["CUSTOM_SETTING"] == "unchanged"
    assert "# Keep this operator setting" in destination.read_text()
    assert after["ALPACA_API_SECRET"] == "'literal$DOLLAR${NAME}#percent%'"
    if os.name == "posix":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_alpaca_update_refuses_linked_files(tmp_path, link_kind):
    target, destination = tmp_path / "original.env", tmp_path / "linked.env"
    setup.create_local_env(target)
    original = target.read_bytes()
    if link_kind == "symlink":
        destination.symlink_to(target)
    else:
        os.link(target, destination)
    with pytest.raises(ValueError, match="without links"):
        setup.configure_alpaca(destination, "synthetic-key", "synthetic-secret")
    assert target.read_bytes() == original


@pytest.mark.parametrize(
    "credential",
    ["", "has space", "line\nbreak", "back\\slash", "nonascii-é", "has'apostrophe"],
)
def test_invalid_credentials_leave_existing_file_intact(tmp_path, credential):
    destination = tmp_path / "existing.env"
    setup.create_local_env(destination)
    original = destination.read_bytes()
    with pytest.raises(ValueError):
        setup.configure_alpaca(destination, "synthetic-key", credential)
    assert destination.read_bytes() == original


@pytest.mark.parametrize(
    "content",
    [
        "ALPACA_API_KEY=one\nALPACA_API_KEY=two\n",
        "CUSTOM='multiline\nALPACA_API_KEY=inside_value\n'\n",
    ],
)
def test_ambiguous_environment_files_are_not_rewritten(tmp_path, content):
    destination = tmp_path / "existing.env"
    destination.write_text(content)
    with pytest.raises(ValueError, match="manual configuration"):
        setup.configure_alpaca(destination, "synthetic-key", "synthetic-secret")
    assert destination.read_text() == content


@pytest.mark.parametrize(
    "failure", [KeyboardInterrupt, EOFError, getpass.GetPassWarning]
)
def test_failed_or_visible_prompt_saves_nothing(tmp_path, monkeypatch, capsys, failure):
    destination = tmp_path / "new.env"
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--output", str(destination), "--alpaca"]
    )

    def fail(prompt):
        if failure is getpass.GetPassWarning:
            import warnings

            warnings.warn("Hidden input unavailable", getpass.GetPassWarning)
        raise failure()

    monkeypatch.setattr(setup.getpass, "getpass", fail)
    assert setup.main() != 0
    assert not destination.exists()
    assert "No credentials were saved" in capsys.readouterr().err


def test_failed_atomic_update_keeps_original_and_removes_temporary_file(
    tmp_path, monkeypatch
):
    destination = tmp_path / "existing.env"
    setup.create_local_env(destination)
    original = destination.read_bytes()

    def fail(*args):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(setup.os, "replace", fail)
    with pytest.raises(OSError):
        setup.configure_alpaca(destination, "synthetic-key", "synthetic-secret")
    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("existing", [False, True])
def test_cancelled_second_prompt_does_not_save_a_partial_pair(
    tmp_path, monkeypatch, capsys, existing
):
    destination = tmp_path / "config.env"
    if existing:
        setup.create_local_env(destination)
    before = destination.read_bytes() if existing else None
    inputs = iter(["synthetic-key", KeyboardInterrupt()])

    def prompt(label):
        value = next(inputs)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(setup.getpass, "getpass", prompt)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--output", str(destination), "--alpaca"]
    )
    assert setup.main() == 130
    assert (destination.read_bytes() if destination.exists() else None) == before
    assert "synthetic-key" not in capsys.readouterr().err


def test_concurrent_edit_is_preserved_instead_of_replaced(tmp_path, monkeypatch):
    destination = tmp_path / "config.env"
    setup.create_local_env(destination)

    def concurrent_edit(descriptor):
        destination.write_text("CUSTOM=operator-edited\n")

    monkeypatch.setattr(setup.os, "fsync", concurrent_edit)
    with pytest.raises(ValueError, match="changed"):
        setup.configure_alpaca(destination, "synthetic-key", "synthetic-secret")
    assert destination.read_text() == "CUSTOM=operator-edited\n"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize(
    "password", ["synthetic-migration-password", "synthetic%40encoded%25password"]
)
def test_migration_logs_do_not_disclose_database_credentials(password):
    url = f"postgresql://migration_user:{password}@localhost:5432/test"
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=ROOT,
        env={**os.environ, "DATABASE_URL": url, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Using database URL from application config" in result.stderr
    assert url not in result.stdout + result.stderr
    assert password not in result.stdout + result.stderr


def test_migration_stops_on_invalid_application_configuration():
    # Deliberately invalid interpolation must not select the development DB.
    script = """
from unittest.mock import patch
from alembic import command
from alembic.config import Config
with patch('investment_tool.config.get_config', side_effect=ValueError('synthetic invalid config')):
    command.upgrade(Config('alembic.ini'), 'head', sql=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "synthetic invalid config" in result.stderr
    assert "CREATE TABLE" not in result.stdout


def test_repository_root_handles_tools_or_root_placement(tmp_path):
    (tmp_path / ".env.example").touch()
    (tmp_path / "docker-compose.prod.yml").touch()
    assert setup.repository_root(tmp_path / "tools" / "setup_local.py") == tmp_path
    assert setup.repository_root(tmp_path / "setup_local.py") == tmp_path


def test_provider_only_env_cannot_report_success_until_completed(tmp_path):
    destination = tmp_path / ".env"
    original = "# Preserve provider keys\nALPACA_API_KEY='synthetic'\nALPACA_API_SECRET='literal$VALUE#secret'\n"
    destination.write_text(original)
    with pytest.raises(ValueError, match="Incomplete .env settings"):
        setup.configure_alpaca(destination, "replacement", "replacement")
    assert destination.read_text() == original
    setup.complete_local_env(destination)
    setup.validate_local_env(destination.read_text())
    assert original in destination.read_text()
    values = setup._settings(destination.read_text())
    assert urlsplit(values["DATABASE_URL"]).password == values["POSTGRES_PASSWORD"]


def test_complete_template_preserves_custom_database_names_and_other_values(tmp_path):
    destination = tmp_path / ".env"
    original = (ROOT / ".env.example").read_text()
    original = original.replace("POSTGRES_USER=investment_tool", "POSTGRES_USER=custom")
    original = original.replace("investment_tool:CHANGE_ME", "custom:CHANGE_ME")
    original += "\nCUSTOM=CHANGE_ME_strong_random_password\n"
    destination.write_text(original)
    setup.complete_local_env(destination)
    values = setup._settings(destination.read_text())
    assert values["CUSTOM"] == "CHANGE_ME_strong_random_password"
    assert urlsplit(values["DATABASE_URL"]).username == values["POSTGRES_USER"]
    assert urlsplit(values["DATABASE_URL"]).password == values["POSTGRES_PASSWORD"]
    setup.validate_local_env(destination.read_text())


@pytest.mark.parametrize("setting", ["POSTGRES_PASSWORD", "DATABASE_URL"])
def test_complete_refuses_mixed_template_and_real_database_credentials(
    tmp_path, setting
):
    destination = tmp_path / ".env"
    setup.create_local_env(destination)
    values = setup._settings(destination.read_text())
    original = destination.read_text().replace(
        f"{setting}={values[setting]}",
        f"{setting}="
        + (
            "CHANGE_ME_strong_random_password"
            if setting == "POSTGRES_PASSWORD"
            else "postgresql://user:CHANGE_ME_strong_random_password@postgres/db"
        ),
    )
    destination.write_text(original)
    with pytest.raises(ValueError, match="Mixed template"):
        setup.complete_local_env(destination)
    assert destination.read_text() == original


def test_complete_refuses_partial_database_tuple(tmp_path):
    destination = tmp_path / ".env"
    original = "POSTGRES_PASSWORD=synthetic-existing\n"
    destination.write_text(original)
    with pytest.raises(ValueError, match="Partial database"):
        setup.complete_local_env(destination)
    assert destination.read_text() == original


@pytest.mark.parametrize("blank", ["", " # blank"])
def test_complete_preserves_existing_secrets_and_replaces_blank_settings(
    tmp_path, blank
):
    destination = tmp_path / ".env"
    setup.create_local_env(destination)
    before = setup._settings(destination.read_text())
    destination.write_text(
        destination.read_text().replace(
            f"REDIS_URL={before['REDIS_URL']}", f"REDIS_URL={blank}"
        )
    )
    setup.complete_local_env(destination)
    after = setup._settings(destination.read_text())
    assert before == after
    assert destination.read_text().count("REDIS_URL=") == 1


def test_check_reports_missing_keys_without_disclosing_values(
    tmp_path, capsys, monkeypatch
):
    destination = tmp_path / ".env"
    original = "ALPACA_API_KEY=synthetic-never-display\n"
    destination.write_text(original)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--output", str(destination), "--check"]
    )
    assert setup.main() == 1
    output = capsys.readouterr()
    assert "DATABASE_URL" in output.err
    assert "synthetic-never-display" not in output.out + output.err
    assert destination.read_text() == original


def test_custom_output_is_used_in_printed_docker_command(tmp_path, capsys, monkeypatch):
    destination = tmp_path / "custom config.env"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(destination)])
    assert setup.main() == 0
    assert f"--env-file '{destination}'" in capsys.readouterr().out
