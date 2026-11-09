"""Explicit browser origins for the supported local API workflow."""

from ipaddress import IPv4Address, IPv6Address
import os
import re
from urllib.parse import urlsplit

_LOCAL_ORIGINS = (
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://localhost",
    "https://127.0.0.1",
    "https://localhost:8443",
    "https://127.0.0.1:8443",
)


def browser_origins() -> tuple[str, ...]:
    """Read an explicit comma-separated allowlist, defaulting to local origins."""
    configured = os.environ.get("CORS_ORIGINS", "")
    if not configured.strip():
        return _LOCAL_ORIGINS
    origins = []
    try:
        for value in configured.split(","):
            value = value.strip()
            if any(ord(character) < 32 or ord(character) == 127 for character in value):
                raise ValueError
            parsed = urlsplit(value)
            if (
                "*" in value
                or parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError
            port = parsed.port
            if port == 0:
                raise ValueError
            host = parsed.hostname.lower()
            if ":" in host:
                # Browser origins compress IPv6; zone IDs are not URL hosts.
                if "%" in host:
                    raise ValueError
                host = f"[{IPv6Address(host).compressed}]"
            else:
                # Require the browser's ASCII host spelling. Python's legacy
                # IDNA codec can map Unicode to a different domain (ß -> ss).
                if not re.fullmatch(r"[a-z0-9.-]+", host):
                    raise ValueError
                last_label = host.rstrip(".").rsplit(".", 1)[-1]
                if last_label.isdigit() or re.fullmatch(r"0x[0-9a-f]+", last_label):
                    # Browsers expand shorthand, hex and octal IPv4 addresses.
                    # Only accept canonical dotted decimal rather than guess.
                    if str(IPv4Address(host)) != host:
                        raise ValueError
            origin = f"{parsed.scheme}://{host}"
            if port is not None and port != {"http": 80, "https": 443}[parsed.scheme]:
                origin += f":{port}"
            origins.append(origin)
    except ValueError:
        raise ValueError(
            "CORS_ORIGINS requires explicit HTTP(S) origins with ASCII hosts "
            "without paths or credentials"
        ) from None
    return tuple(dict.fromkeys(origins))
