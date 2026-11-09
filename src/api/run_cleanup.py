"""Terminal-run output cleanup; shared input data and caches are retained."""

from pathlib import Path
import re
import shutil


def delete_run_outputs(base: Path, run_id: str) -> None:
    if re.fullmatch(r"[0-9a-f]{12}", run_id) is None:
        raise ValueError("Invalid run ID for output cleanup")
    directory = base / run_id
    if directory.is_symlink() or directory.resolve().parent != base.resolve():
        raise ValueError("Unsafe run output directory")
    if directory.exists():
        if not directory.is_dir():
            raise ValueError("Run output path is not a directory")
        shutil.rmtree(directory)


def ensure_cancelled_task_stopped(task_id: str) -> None:
    """Cancellation dispatch alone does not prove a prefork child has stopped."""
    from api.worker import celery_app

    try:
        inspector = celery_app.control.inspect(timeout=2)
        responses = [inspector.active(), inspector.reserved(), inspector.scheduled()]
    except Exception as exc:
        raise RuntimeError("Could not inspect worker state") from exc
    if any(not response for response in responses):
        raise RuntimeError("Could not verify that the cancelled worker has stopped")
    nodes = [set(response) for response in responses]
    if any(node != nodes[0] for node in nodes):
        raise RuntimeError("Worker availability changed during cancellation check")
    for response in responses:
        for tasks in response.values():
            if any(task.get("request", task).get("id") == task_id for task in tasks):
                raise ValueError(
                    "The cancelled task is still active; retry after it stops"
                )
