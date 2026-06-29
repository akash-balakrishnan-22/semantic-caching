import time
from typing import Dict, List, Optional


class PerfEval:
    """Tracks latency per labeled step (e.g. cache_hit, llm_call)."""

    def __init__(self):
        self.durations_by_label: Dict[str, List[float]] = {}
        self.last_time: Optional[float] = None

    def __enter__(self):
        self.last_time = time.time()
        self.durations_by_label = {}
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def start(self):
        self.last_time = time.time()

    def tick(self, label: Optional[str] = None):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
        dt = now - self.last_time
        if label:
            self.durations_by_label.setdefault(label, []).append(dt)
        self.last_time = now
