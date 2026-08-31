"""Run seed (full wipe ingest) then inspect_l1; print wall-clock timings."""
from __future__ import annotations

import os
import subprocess
import sys
import time


def main():
    env = {**os.environ, "MARKETS_RESET": "1", "PYTHONUNBUFFERED": "1"}
    py = sys.executable
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    print("pipeline: seed start", flush=True)
    t = time.perf_counter()
    subprocess.run([py, "seed.py"], check=True, env=env)
    timings["seed"] = time.perf_counter() - t

    print("pipeline: inspect start", flush=True)
    t = time.perf_counter()
    subprocess.run([py, "inspect_l1.py"], check=True, env=env)
    timings["inspect"] = time.perf_counter() - t

    total = time.perf_counter() - t0
    print("pipeline_timing:", flush=True)
    for label, secs in timings.items():
        pct = 100 * secs / total if total else 0
        print(f"  {label}: {secs:.1f}s ({pct:.0f}%)", flush=True)
    print(f"  total: {total:.1f}s ({total / 60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
