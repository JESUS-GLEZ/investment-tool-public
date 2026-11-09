"""Independent liveness/progress persistence, including inside long solver calls."""

import logging
import math
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Any, cast

from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, sessionmaker

from api.models import RunRow

logger = logging.getLogger(__name__)


class RunMonitor:
    def __init__(
        self,
        run_id: str,
        session_factory: sessionmaker[Session],
        interval: float = 10,
    ) -> None:
        self.run_id = run_id
        self.session_factory = session_factory
        self.interval = interval
        self.stop_event = Event()
        self.lock = Lock()
        self.fraction = 0.0
        self.message = "Preparing run"
        self.thread = Thread(target=self._loop, name=f"heartbeat-{run_id}", daemon=True)

    def progress(self, fraction: float, message: str) -> None:
        with self.lock:
            if math.isfinite(fraction):
                self.fraction = min(1.0, max(0.0, fraction))
            self.message = str(message)

    def beat(self) -> bool:
        with self.lock:
            fraction, message = self.fraction, self.message
        with self.session_factory() as db:
            result = cast(
                CursorResult[Any],
                db.execute(
                    update(RunRow)
                    .where(RunRow.id == self.run_id, RunRow.status == "running")
                    .values(
                        heartbeat_at=datetime.now(timezone.utc),
                        progress_fraction=fraction,
                        progress_message=message,
                    )
                ),
            )
            db.commit()
            return result.rowcount > 0

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                if not self.beat():
                    return
            except Exception:
                logger.warning(
                    "Heartbeat failed for run %s", self.run_id, exc_info=True
                )
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
