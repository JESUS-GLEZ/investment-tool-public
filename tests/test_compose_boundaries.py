"""Check the supported local network boundary without starting services."""

from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("filename", ["docker-compose.yml", "docker-compose.prod.yml"])
def test_published_ports_bind_only_to_host_loopback(filename: str) -> None:
    root = Path(__file__).resolve().parents[1]
    services = yaml.safe_load((root / filename).read_text())["services"]
    published = [
        port for service in services.values() for port in service.get("ports", [])
    ]
    assert published
    for port in published:
        host, published_port, container_port = port.split(":")
        assert host == "127.0.0.1"
        assert published_port.isdecimal()
        assert 1024 <= int(published_port) <= 65535
        assert container_port.isdecimal()


def test_local_app_and_adminer_use_distinct_unprivileged_host_ports() -> None:
    root = Path(__file__).resolve().parents[1]
    services = yaml.safe_load((root / "docker-compose.prod.yml").read_text())[
        "services"
    ]
    assert services["nginx"]["ports"] == ["127.0.0.1:8080:80"]
    assert services["adminer"]["ports"] == ["127.0.0.1:8081:8080"]


def test_full_stack_api_and_broker_stay_on_internal_service_ports() -> None:
    root = Path(__file__).resolve().parents[1]
    services = yaml.safe_load((root / "docker-compose.prod.yml").read_text())[
        "services"
    ]
    for name, port in [("api", "8000"), ("redis", "6379")]:
        assert not services[name].get("ports")
        assert services[name]["expose"] == [port]
    for name in ["api", "worker"]:
        environment = services[name]["environment"]
        assert any(
            value.startswith("DATABASE_URL=${DATABASE_URL:?") for value in environment
        )
        assert any(value.startswith("REDIS_URL=${REDIS_URL:?") for value in environment)
    assert "CORS_ORIGINS=${CORS_ORIGINS:-}" in services["api"]["environment"]


def test_api_and_worker_share_runtime_image_and_optional_provider_settings() -> None:
    root = Path(__file__).resolve().parents[1]
    services = yaml.safe_load((root / "docker-compose.prod.yml").read_text())[
        "services"
    ]
    assert services["api"]["image"] == services["worker"]["image"]
    assert services["api"]["build"] == services["worker"]["build"]
    for name in ["api", "worker"]:
        environment = services[name]["environment"]
        for key in ["ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPHA_VANTAGE_API_KEY"]:
            assert f"{key}=${{{key}:-}}" in environment
        assert any(
            value.startswith("JWT_SECRET_KEY=${JWT_SECRET_KEY:?")
            for value in environment
        )
