import json
import time
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    def __init__(self):
        self.count = defaultdict(int)
        self.time = defaultdict(float)
        self.enabled = False

    def enable(self, enabled: bool):
        self.enabled = enabled

    @contextmanager
    def profile(self, name: str, count: int = 1):
        if not self.enabled:
            yield
        else:
            start = time.perf_counter()
            yield
            end = time.perf_counter()
            self.count[name] += count
            self.time[name] += end - start

    def print_stats(self):
        for name in sorted(self.count.keys()):
            print(f"{name}: {self.count[name]} calls, {self.time[name]:.3f} seconds")

    def save(self, path: str):
        merged_results = {
            name: {"count": count, "time": time}
            for name, count, time in zip(self.count.keys(), self.count.values(), self.time.values())
        }
        with open(path, "w") as f:
            json.dump(merged_results, f, indent=4)
        print(f"Saved profiler results to {path}")

    def clear(self):
        self.count.clear()
        self.time.clear()


profiler = Profiler()
