"""Configure a local Docker installation without exposing credentials in arguments."""

from __future__ import annotations

import argparse
import getpass
import os
import re
import secrets
import shlex
import stat
import sys
import tempfile
import warnings
from pathlib import Path
from collections.abc import Callable


def repository_root(script: Path) -> Path:
    """Find the project even if the helper was copied from tools/ to its root."""
    for parent in script.resolve().parents:
        if (parent / ".env.example").is_file() and (
            parent / "docker-compose.prod.yml"
        ).is_file():
            return parent
    raise ValueError(
        "Run setup_local.py inside the repository containing .env.example and docker-compose.prod.yml"
    )


ROOT = repository_root(Path(__file__))
DATABASE_SETTINGS = {
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "DATABASE_URL",
}
REQUIRED_SETTINGS = DATABASE_SETTINGS | {"REDIS_URL", "JWT_SECRET_KEY"}


def _settings(content: str) -> dict[str, str]:
    """Read assignment presence without expanding or displaying secret values."""
    result = {}
    for line in content.splitlines():
        match = re.match(
            r"\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*[=:]\s*(.*)", line
        )
        if match is None:
            continue
        name, value = match.groups()
        if name in result:
            raise ValueError(
                "Duplicate environment settings require manual configuration"
            )
        value = value.strip()
        if value.startswith(("'", '"')):
            end = re.search(r"(?<!\\)" + value[0], value[1:])
            if end is None:
                raise ValueError(
                    "Multiline environment values require manual configuration"
                )
            value = value[1 : end.start() + 1]
        else:
            value = value.split(" #", 1)[0].strip()
            if value.startswith("#"):
                value = ""
        result[name] = value
    return result


def validate_local_env(content: str) -> None:
    values = _settings(content)
    missing = sorted(name for name in REQUIRED_SETTINGS if not values.get(name))
    placeholders = sorted(
        name for name in REQUIRED_SETTINGS if "CHANGE_ME" in values.get(name, "")
    )
    if missing or placeholders:
        names = ", ".join(sorted(set(missing + placeholders)))
        raise ValueError(
            f"Incomplete .env settings: {names}. No changes were saved. "
            "For a fresh installation, run python3 tools/setup_local.py --complete. "
            "For an existing database, restore its original configuration; do not regenerate credentials."
        )


def _with_alpaca(content: str, key: str, secret: str) -> str:
    """Replace only single-line Alpaca assignments; preserve other configuration."""
    values = {"ALPACA_API_KEY": key, "ALPACA_API_SECRET": secret}
    for value in values.values():
        if not value or any(
            not 33 <= ord(char) <= 126 or char in "\\'" for char in value
        ):
            raise ValueError(
                "Credentials must be nonempty printable ASCII without spaces, apostrophes or backslashes"
            )
    lines = content.splitlines(keepends=True)
    seen: set[str] = set()
    for index, line in enumerate(lines):
        match = re.match(
            r"\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*[=:]\s*(.*)", line
        )
        if match is None:
            continue
        name, current = match.groups()
        if (
            current.startswith(("'", '"'))
            and re.search(r"(?<!\\)" + current[0], current[1:]) is None
        ):
            raise ValueError(
                "Multiline environment values require manual configuration"
            )
        if name not in values:
            continue
        if name in seen:
            raise ValueError("Duplicate Alpaca settings require manual configuration")
        seen.add(name)
        # Compose single quotes prevent '$' interpolation and preserve '#' literally.
        lines[index] = f"{name}='{values[name]}'\n"
    for name, value in values.items():
        if name not in seen:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f"{name}='{value}'\n")
    return "".join(lines)


def _new_content(template: Path) -> str:
    content = template.read_text(encoding="utf-8")
    replacements = {
        "CHANGE_ME_strong_random_password": secrets.token_urlsafe(32),
        "CHANGE_ME_generate_a_strong_secret": secrets.token_urlsafe(64),
    }
    for placeholder, value in replacements.items():
        if placeholder not in content:
            raise ValueError(
                "The environment template is missing a required placeholder"
            )
        content = content.replace(placeholder, value)
    return content


def create_local_env(
    destination: Path,
    template: Path = ROOT / ".env.example",
    alpaca: tuple[str, str] | None = None,
) -> None:
    """Generate independent secrets; never replace an existing file or symlink."""
    content = _new_content(template)
    if alpaca is not None:
        content = _with_alpaca(content, *alpaca)
    validate_local_env(content)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
        output.write(content)


def _update_env(destination: Path, update: Callable[[str], str]) -> None:
    """Atomically update an owned regular file, refusing concurrent edits/links."""
    before = destination.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("The environment file must be a regular file without links")
    if hasattr(os, "getuid") and before.st_uid != os.getuid():
        raise ValueError("The environment file must belong to the current user")
    content = update(destination.read_bytes().decode("utf-8"))
    validate_local_env(content)
    descriptor, filename = tempfile.mkstemp(prefix=".env.", dir=destination.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        after = destination.lstat()
        if (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_size,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_size,
        ):
            raise ValueError(
                "The environment file changed; retry without concurrent edits"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def configure_alpaca(destination: Path, key: str, secret: str) -> None:
    """Change only the provider pair in an otherwise complete installation."""
    _update_env(destination, lambda content: _with_alpaca(content, key, secret))


def complete_local_env(
    destination: Path, alpaca: tuple[str, str] | None = None
) -> None:
    """Complete a fresh partial/template file; never replace real settings.

    A partially supplied database tuple is ambiguous: refuse rather than invent
    a password or connection URL that could disagree with an existing database.
    """

    def complete(content: str) -> str:
        values = _settings(content)
        defined_database = {name for name in DATABASE_SETTINGS if values.get(name)}
        if defined_database and defined_database != DATABASE_SETTINGS:
            raise ValueError(
                "Partial database configuration requires manual completion using the original database credentials. No changes were saved."
            )
        password_placeholder = "CHANGE_ME_strong_random_password"
        templated_password = values.get("POSTGRES_PASSWORD") == password_placeholder
        templated_url = password_placeholder in values.get("DATABASE_URL", "")
        if templated_password != templated_url:
            raise ValueError(
                "Mixed template and configured database credentials require manual completion. No changes were saved."
            )
        generated = _new_content(ROOT / ".env.example")
        generated_values = _settings(generated)
        for name in sorted(REQUIRED_SETTINGS):
            pattern = rf"(?m)^[ \t]*(?:export[ \t]+)?{name}[ \t]*[=:][^\n]*$"
            placeholder = (
                password_placeholder
                if name in {"POSTGRES_PASSWORD", "DATABASE_URL"} and templated_password
                else "CHANGE_ME_generate_a_strong_secret"
                if name == "JWT_SECRET_KEY"
                and values.get(name) == "CHANGE_ME_generate_a_strong_secret"
                else None
            )
            if placeholder is not None:
                value = generated_values[
                    "POSTGRES_PASSWORD" if name == "DATABASE_URL" else name
                ]
                content = re.sub(
                    pattern, lambda match: match[0].replace(placeholder, value), content
                )
            if not values.get(name):
                # Replace a blank assignment without creating a duplicate.
                replacement = f"{name}={generated_values[name]}"
                if re.search(pattern, content):
                    content = re.sub(pattern, lambda _: replacement, content)
                else:
                    content = content.rstrip() + "\n" + replacement + "\n"
        return _with_alpaca(content, *alpaca) if alpaca is not None else content

    _update_env(destination, complete)


def _prompt_alpaca() -> tuple[str, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        return getpass.getpass("Alpaca API key (hidden): "), getpass.getpass(
            "Alpaca API secret (hidden): "
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check required .env settings without printing values or changing the file",
    )
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Fresh installations only: fill missing/template settings while preserving existing values",
    )
    parser.add_argument(
        "--alpaca",
        action="store_true",
        help="Prompt for Alpaca credentials; create .env or update only that pair",
    )
    args = parser.parse_args()
    if args.check and (args.alpaca or args.complete):
        parser.error("--check cannot be combined with --alpaca or --complete")
    try:
        if args.check:
            validate_local_env(args.output.read_text(encoding="utf-8"))
            print(
                f"Required settings are present in {args.output}. No values were displayed; connectivity was not tested."
            )
            return 0
        pair = _prompt_alpaca() if args.alpaca else None
        exists = args.output.exists() or args.output.is_symlink()
        if args.complete and exists:
            complete_local_env(args.output, pair)
        elif pair is not None:
            if exists:
                configure_alpaca(args.output, *pair)
            else:
                create_local_env(args.output, alpaca=pair)
        else:
            create_local_env(args.output)
    except FileExistsError:
        print(
            "Refusing to replace an existing environment file. Keep it for updates.",
            file=sys.stderr,
        )
        return 1
    except (getpass.GetPassWarning, EOFError):
        print(
            "An interactive terminal with hidden input is required. No credentials were saved.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nCancelled. No credentials were saved.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as error:
        print(f"Could not configure the environment file: {error}", file=sys.stderr)
        return 1
    if args.complete:
        print(
            f"Completed {args.output}; existing non-placeholder settings were preserved."
        )
    elif args.alpaca:
        print(
            f"Saved Alpaca credentials in {args.output}; database and session secrets were preserved for existing installations."
        )
    else:
        print(
            f"Created {args.output}. Provider keys are blank; use --alpaca to configure them interactively."
        )
    print(f"Run Docker commands from {ROOT}, using this environment file.")
    print(
        "After building images and migrating a fresh install (or after active runs finish on an existing install), apply with: "
        f"docker compose --env-file {shlex.quote(str(args.output.resolve()))} "
        f"-f {shlex.quote(str(ROOT / 'docker-compose.prod.yml'))} up -d api worker nginx"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
