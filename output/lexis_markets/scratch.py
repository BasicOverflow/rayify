"""Worker-local scratch dirs and orphan cleanup.

Kaggle prepare and seed lake actors use ``/tmp/lexis-*`` trees. A normal
``TemporaryDirectory`` or actor ``shutdown`` deletes them. ``ray.cancel(force=True)``
and worker death leave multi-GB zips behind. A pid marker lets prune skip
in-flight dirs and delete the rest.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

LIVE_MARKER = ".lexis-tmp-live"

ORPHAN_TMP_PREFIXES = (
    "lexis-jw-stage-",
    "lexis-jc-stage-",
    "lexis-kaggle-",
    "lexis-seed-",
    "l1_map_",
)


def mark_live(path: Path | str) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (p / LIVE_MARKER).write_text(str(os.getpid()), encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) on Windows can inject CTRL_C into this console.
        if pid == os.getpid():
            return True
        try:
            from ctypes import windll

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                windll.kernel32.CloseHandle(handle)
                return True
        except OSError:
            return False
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def is_live_scratch(path: Path | str) -> bool:
    marker = Path(path) / LIVE_MARKER
    if not marker.is_file():
        return False
    try:
        pid = int(marker.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    return _pid_alive(pid)


def dir_bytes(path: str) -> int:
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


@contextmanager
def scratch_dir(prefix: str):
    """TemporaryDirectory that prune will not delete while this pid is alive."""
    with tempfile.TemporaryDirectory(prefix=prefix, ignore_cleanup_errors=True) as tmp:
        path = Path(tmp)
        mark_live(path)
        yield path


def prune_orphan_tmp_dirs(tmp_root: str = "/tmp") -> tuple[int, int]:
    """Remove leftover lexis scratch dirs whose owner pid is gone.

    Returns (dirs_removed, bytes_freed). Live seed workers and in-flight
    kaggle prepare keep a pid marker and are skipped.
    """
    removed = 0
    freed = 0
    if not os.path.isdir(tmp_root):
        return 0, 0
    try:
        names = os.listdir(tmp_root)
    except OSError:
        return 0, 0
    for name in names:
        if not any(name.startswith(p) for p in ORPHAN_TMP_PREFIXES):
            continue
        path = os.path.join(tmp_root, name)
        if not os.path.isdir(path):
            continue
        if is_live_scratch(path):
            continue
        try:
            nbytes = dir_bytes(path)
            shutil.rmtree(path, ignore_errors=True)
            if not os.path.exists(path):
                removed += 1
                freed += nbytes
        except OSError:
            pass
    return removed, freed
