"""Exercise framework upgrades through real static and authenticated file routes."""

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api import auth, main
from api.database import get_db
from api.db_store import DBRunStore
from api.models import Base, UserRow
from api.routers import runs
from api.schemas import RunType


@pytest.fixture()
def files_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, str, str, str]]:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<html>demo</html>")
    (frontend / "asset.txt").write_bytes(b"0123456789")
    (frontend / "many-ranges.txt").write_bytes(b"x" * 1024)
    monkeypatch.setattr(main, "_FRONTEND_DIST", frontend)
    monkeypatch.setattr(runs, "_OUTPUT_BASE", tmp_path / "outputs")
    monkeypatch.setattr(
        auth, "_jwt_config", lambda: ("file-serving-test-secret", "HS256", 20, 60)
    )

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    store = DBRunStore()
    with factory() as db:
        owner = UserRow(email="owner@example.test", password_hash="unused")
        other = UserRow(email="other@example.test", password_hash="unused")
        db.add_all([owner, other])
        db.commit()
        run = store.create(RunType.EXECUTION, {}, db=db, user_id=owner.id)
        owner_token = auth.create_access_token({"sub": str(owner.id)})
        other_token = auth.create_access_token({"sub": str(other.id)})

    output = tmp_path / "outputs" / run.id
    output.mkdir(parents=True)
    (output / "results.txt").write_bytes(b"0123456789")
    (output / "many-ranges.txt").write_bytes(b"x" * 1024)
    (output / ".keep").touch()

    def session() -> Iterator[Session]:
        with factory() as db:
            yield db

    app = main.create_app(store=store)
    app.dependency_overrides[get_db] = session
    try:
        with TestClient(app) as client:
            yield client, run.id, owner_token, other_token
    finally:
        engine.dispose()


def test_static_files_and_api_routes_coexist(files_client) -> None:
    client, _, _, _ = files_client
    assert client.get("/").text == "<html>demo</html>"
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/openapi.json").json()["paths"]["/runs"]
    assert client.get("/asset.txt").content == b"0123456789"
    head = client.head("/asset.txt")
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == "10"


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize(
    ("range_header", "status", "body", "content_range"),
    [
        ("bytes=2-5", 206, b"2345", "bytes 2-5/10"),
        ("bytes=6-9,0-3,2-7", 206, b"0123456789", "bytes 0-9/10"),
        ("bytes=100-200", 416, None, "bytes */10"),
        ("bytes=not-a-range", 400, None, None),
    ],
)
def test_range_requests(
    files_client, static, range_header, status, body, content_range
) -> None:
    client, run_id, token, _ = files_client
    url = "/asset.txt" if static else f"/runs/{run_id}/files/results.txt"
    response = client.get(
        url, headers={"Authorization": f"Bearer {token}", "Range": range_header}
    )
    assert response.status_code == status
    if body is not None:
        assert response.content == body
    if content_range is not None:
        assert response.headers["content-range"] == content_range


@pytest.mark.timeout(5)
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize(
    ("count", "status", "length"),
    [(100, 206, 101), (101, 200, 1024), (1000, 200, 1024)],
)
def test_many_ranges_are_bounded(files_client, static, count, status, length) -> None:
    client, run_id, token, _ = files_client
    ranges = ",".join(f"{i}-{i + 1}" for i in reversed(range(count)))
    url = "/many-ranges.txt" if static else f"/runs/{run_id}/files/many-ranges.txt"
    response = client.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Range": f"bytes={ranges}"},
    )
    assert response.status_code == status
    assert response.content == b"x" * length
    if status == 206:
        assert response.headers["content-range"] == "bytes 0-100/1024"
    else:
        assert "content-range" not in response.headers


def test_download_preserves_attachment_and_zip_contents(files_client) -> None:
    client, run_id, token, _ = files_client
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get(f"/runs/{run_id}/files/results.txt", headers=headers)
    assert response.status_code == 200
    assert response.content == b"0123456789"
    assert (
        response.headers["content-disposition"] == 'attachment; filename="results.txt"'
    )
    archive = client.get(f"/runs/{run_id}/files/download-all", headers=headers)
    assert archive.status_code == 200
    assert archive.headers["content-type"] == "application/zip"
    with ZipFile(BytesIO(archive.content)) as zipped:
        assert set(zipped.namelist()) == {"results.txt", "many-ranges.txt"}
        assert zipped.read("results.txt") == b"0123456789"


@pytest.mark.parametrize("filename", ["", "results.txt", "download-all"])
def test_downloads_require_authentication_and_run_ownership(
    files_client, filename
) -> None:
    client, run_id, _, other_token = files_client
    url = f"/runs/{run_id}/files/{filename}"
    assert client.get(url).status_code == 401
    response = client.get(url, headers={"Authorization": f"Bearer {other_token}"})
    assert response.status_code == 404


@pytest.mark.parametrize("destination", ["external", "internal", "missing"])
def test_file_links_are_not_listed_or_downloaded(files_client, tmp_path, destination):
    client, run_id, token, _ = files_client
    output = tmp_path / "outputs" / run_id
    targets = {
        "external": tmp_path / "other-account.txt",
        "internal": output / "results.txt",
        "missing": tmp_path / "missing.txt",
    }
    if destination == "external":
        targets[destination].write_bytes(b"private-other-account-output")
    (output / "linked.txt").symlink_to(targets[destination])
    headers = {"Authorization": f"Bearer {token}"}
    listing = client.get(f"/runs/{run_id}/files", headers=headers)
    assert listing.status_code == 200
    assert {item["path"] for item in listing.json()} == {
        "results.txt",
        "many-ranges.txt",
    }
    archive = client.get(f"/runs/{run_id}/files/download-all", headers=headers)
    assert archive.status_code == 200
    with ZipFile(BytesIO(archive.content)) as zipped:
        assert set(zipped.namelist()) == {"results.txt", "many-ranges.txt"}
    response = client.get(f"/runs/{run_id}/files/linked.txt", headers=headers)
    assert response.status_code == 400


@pytest.mark.parametrize("filename", ["", "/download-all", "/private.txt"])
def test_linked_run_root_cannot_export_another_directory(
    files_client, tmp_path, filename
):
    client, run_id, token, _ = files_client
    output = tmp_path / "outputs" / run_id
    output.rename(output.with_name("original"))
    other = tmp_path / "other-run"
    other.mkdir()
    (other / "private.txt").write_bytes(b"private-other-account-output")
    output.symlink_to(other, target_is_directory=True)
    response = client.get(
        f"/runs/{run_id}/files{filename}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400


def test_nested_files_work_but_directory_links_do_not(files_client, tmp_path):
    client, run_id, token, _ = files_client
    output = tmp_path / "outputs" / run_id
    nested = output / "plots"
    nested.mkdir()
    (nested / "report.html").write_bytes(b"<html>owned report</html>")
    (output / "linked-plots").symlink_to(nested, target_is_directory=True)
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get(f"/runs/{run_id}/files/plots/report.html", headers=headers)
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert (
        client.get(
            f"/runs/{run_id}/files/linked-plots/report.html", headers=headers
        ).status_code
        == 400
    )
    listing = client.get(f"/runs/{run_id}/files", headers=headers)
    assert "plots/report.html" in {item["path"] for item in listing.json()}
    assert not any("linked-plots" in item["path"] for item in listing.json())
    archive = client.get(f"/runs/{run_id}/files/download-all", headers=headers)
    with ZipFile(BytesIO(archive.content)) as zipped:
        assert "plots/report.html" in zipped.namelist()
        assert not any("linked-plots" in name for name in zipped.namelist())


def test_zip_excludes_external_file_contents(files_client, tmp_path):
    client, run_id, token, _ = files_client
    private = tmp_path / "private.txt"
    private.write_bytes(b"private-other-account-output")
    (tmp_path / "outputs" / run_id / "linked.txt").symlink_to(private)
    response = client.get(
        f"/runs/{run_id}/files/download-all",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    with ZipFile(BytesIO(response.content)) as zipped:
        assert "linked.txt" not in zipped.namelist()
        assert all(
            b"private-other-account-output" not in zipped.read(name)
            for name in zipped.namelist()
        )
