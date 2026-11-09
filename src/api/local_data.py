"""Resolve operator-trusted API datasets within the mounted input directory."""

from pathlib import Path, PureWindowsPath


def validate_dataset_filename(filename: str) -> None:
    """Require a plain .pkl basename on POSIX and Windows."""
    if (
        not filename.strip()
        or Path(filename).name != filename
        or "\\" in filename
        or PureWindowsPath(filename).drive
        or Path(filename).suffix != ".pkl"
    ):
        raise ValueError(
            "Local dataset must be a .pkl filename within the input directory"
        )


def resolve_dataset_path(data_dir: Path, filename: str) -> Path:
    """Return a regular file whose canonical parent is the trusted input root."""
    validate_dataset_filename(filename)
    try:
        root = data_dir.resolve()
        path = (root / filename).resolve(strict=True)
        if path.parent != root or path.suffix != ".pkl" or not path.is_file():
            raise ValueError
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "Local dataset is unavailable within the input directory"
        ) from exc
    return path


def available_datasets(data_dir: Path) -> list[tuple[str, Path]]:
    """List valid files newest first, retaining their selectable basenames."""
    files = []
    for entry in data_dir.glob("*.pkl"):
        try:
            path = resolve_dataset_path(data_dir, entry.name)
            files.append((entry.name, path, path.stat().st_mtime))
        except (ValueError, OSError):
            continue
    return [
        (name, path)
        for name, path, _ in sorted(files, key=lambda f: f[2], reverse=True)
    ]
