import json
from pathlib import Path

from api.dataset_catalog import read_manifest, source_stamp, write_manifest


def test_bad_sidecars_are_ignored_and_never_override_file_identity(tmp_path: Path):
    source = tmp_path / "data.pkl"
    source.write_bytes(b"trusted")
    sidecar = tmp_path / "data.pkl.dataset.json"
    sidecar.write_text("not json")
    assert read_manifest(source) == {}
    sidecar.write_text(
        json.dumps(
            {
                "format_version": 1,
                "source": source_stamp(source),
                "info": {
                    "title": {},
                    "file": "another.pkl",
                    "stock_count": True,
                    "features": {"news": "true", "prices": True},
                },
            }
        )
    )
    assert read_manifest(source) == {"features": {"prices": True}}
    sidecar.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("private")
    sidecar.symlink_to(outside)
    write_manifest(source, {"title": "Changed"}, source_stamp(source))
    assert read_manifest(source) == {}
    assert outside.read_text() == "private"


def test_source_changed_during_inspection_is_not_cached(tmp_path: Path):
    source = tmp_path / "data.pkl"
    source.write_bytes(b"old")
    stamp = source_stamp(source)
    source.write_bytes(b"new bytes")
    write_manifest(source, {"title": "Stale"}, stamp)
    assert read_manifest(source) == {}
