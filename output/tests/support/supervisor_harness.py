"""Supervisor subprocess control for chaos tests."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from lexis_markets.config import OUTPUT_DIR

OUTPUT = Path(OUTPUT_DIR)


async def start_supervisor(env: dict[str, str], *, state_path: str) -> asyncio.subprocess.Process:
    run_env = {**os.environ, **env, "PYTHONPATH": str(OUTPUT), "MARKETS_SUPERVISOR_STATE": state_path}
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "lexis_markets.supervisor.main",
        cwd=str(OUTPUT),
        env=run_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )


async def kill_supervisor(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    proc.kill()
    try:
        await asyncio.wait_for(proc.wait(), timeout=30)
    except asyncio.TimeoutError:
        pass


async def stop_supervisor(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=30)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
