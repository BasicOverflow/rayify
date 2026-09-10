"""Named Ray actors holding worker-local lake dirs for seed.

Ingest, compact, align, and materialize write here. Compacted L1 + L3 cache
flush to MinIO once at the end of seed. Daily EOD uses ``LakeStore`` (S3).
"""
from __future__ import annotations

import contextvars
import hashlib
import io
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import ray

from lexis_markets.config import MarketsConfig
from lexis_markets.lake.local import LocalDirLake
from lexis_markets.lake.store import LakeStore
from lexis_markets.logging_setup import get_logger
from lexis_markets.ray.runtime import IO_REMOTE_OPTS
from lexis_markets.scratch import mark_live

logger = get_logger("lake.cluster")

SEED_LAKE_GATE_NAME = "lexis-seed-lake"
SEED_SCRATCH_KEY = "__seed_scratch"


def worker_actor_name(i: int, gen: str = "") -> str:
    if gen:
        return f"lexis-seed-{gen}-w-{i}"
    return f"lexis-seed-w-{i}"

_MONTH_RE = re.compile(r"layer_1/year=(\d+)/month=(\d+)/")
_scratch = contextvars.ContextVar("lexis_seed_scratch", default=False)


def _actor_namespace(cfg: MarketsConfig | None = None) -> str | None:
    if cfg is not None:
        return cfg.ray_namespace or os.environ.get("RAY_NAMESPACE") or None
    return os.environ.get("RAY_NAMESPACE") or None


def _get_actor(name: str, *, namespace: str | None = None):
    ns = namespace or _actor_namespace()
    if ns:
        return ray.get_actor(name, namespace=ns)
    return ray.get_actor(name)


@contextmanager
def seed_scratch_scope(enabled: bool = True):
    token = _scratch.set(bool(enabled))
    try:
        yield
    finally:
        _scratch.reset(token)


def cfg_d_with_scratch(cfg: MarketsConfig) -> dict:
    d = cfg.to_dict()
    d[SEED_SCRATCH_KEY] = bool(_scratch.get())
    return d


def reset_seed_scratch() -> None:
    """Clear the process scratch flag. Call at job dispatch so worker reuse cannot leak seed mode."""
    _scratch.set(False)


def lake_from_cfg_d(cfg_d: dict):
    cfg = MarketsConfig.from_dict(cfg_d)
    scratch = bool(cfg_d.get(SEED_SCRATCH_KEY))
    return open_lake(cfg, scratch=scratch)


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _read_parquet_bytes(
    data: bytes,
    *,
    columns: list[str] | None = None,
    filters: list | None = None,
) -> pd.DataFrame:
    return pd.read_parquet(
        io.BytesIO(data),
        columns=columns,
        filters=filters,
        engine="pyarrow",
    )


def owner_index(key: str, n: int, *, prefix: str = "") -> int:
    """Colocate ``layer_1/year=/month=/`` keys; hash everything else."""
    if n <= 1:
        return 0
    rel = str(key).lstrip("/")
    if prefix and rel.startswith(prefix):
        rel = rel[len(prefix) :]
    m = _MONTH_RE.search(rel)
    token = f"{m.group(1)}-{m.group(2)}" if m else rel
    return int(hashlib.sha256(token.encode()).hexdigest(), 16) % n


def is_flush_key(key: str, *, prefix: str = "") -> bool:
    """Compacted L1 + L3 only. Skip parts, staging, zip cache, markers."""
    rel = str(key).lstrip("/")
    if prefix and rel.startswith(prefix):
        rel = rel[len(prefix) :]
    if "/part-" in rel:
        return False
    return rel.startswith("layer_1/") or rel.startswith("layer_3/")


@ray.remote(**IO_REMOTE_OPTS)
class SeedLakeGate:
    def __init__(self, n: int, gen: str = ""):
        self._n = int(n)
        self._gen = str(gen)

    def worker_count(self) -> int:
        return self._n

    def generation(self) -> str:
        return self._gen


@ray.remote(**IO_REMOTE_OPTS)
class SeedWorker:
    def __init__(self, idx: int, prefix: str = ""):
        self.idx = int(idx)
        self.root = Path(tempfile.mkdtemp(prefix=f"lexis-seed-{self.idx}-"))
        mark_live(self.root)
        self.lake = LocalDirLake(self.root, prefix=prefix)

    def ping(self) -> str:
        return str(self.root)

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        self.lake.put_bytes(key, data, content_type)

    def put_df(self, key: str, data: bytes):
        self.lake.put_bytes(key, data)

    def get_bytes(self, key: str) -> bytes:
        return self.lake.get_bytes(key)

    def get_df(
        self,
        key: str,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> bytes:
        # Parquet bytes, not a DataFrame: worker numpy versions can differ.
        if columns is None and filters is None:
            return self.lake.get_bytes(key)
        return _parquet_bytes(self.lake.get_df_parquet(key, columns=columns, filters=filters))

    def exists(self, key: str) -> bool:
        return self.lake.exists(key)

    def list_keys(self, prefix: str) -> list[str]:
        return self.lake.list_keys(prefix)

    def list_common_prefixes(self, prefix: str) -> list[str]:
        return self.lake.list_common_prefixes(prefix)

    def delete_keys(self, keys: list[str]):
        self.lake.delete_keys(keys)

    def flush_minio(self, cfg_d: dict) -> dict:
        cfg = MarketsConfig.from_dict(cfg_d)
        remote = LakeStore(cfg)
        files = 0
        nbytes = 0
        for key in self.lake.list_keys(""):
            if not is_flush_key(key, prefix=self.lake.prefix):
                continue
            path = self.lake._path(key)
            if not path.is_file():
                continue
            remote.put_file(key, path)
            files += 1
            nbytes += path.stat().st_size
        logger.info("seed flush worker=%s files=%s bytes=%s", self.idx, files, nbytes)
        return {"worker": self.idx, "files": files, "bytes": nbytes}

    def shutdown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class ClusterLake:
    """LakeStore duck-type that RPCs keys to named ``SeedWorker`` actors."""

    def __init__(self, cfg: MarketsConfig, n: int):
        self.cfg = cfg
        self.n = int(n)
        self.prefix = (cfg.lake_prefix or "").strip()
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix = f"{self.prefix}/"
        self.bucket = cfg.s3_bucket
        self.client = None
        ns = _actor_namespace(cfg)
        _n, gen = _seed_lake_info(namespace=ns)
        self._gen = gen
        self._workers = [
            _get_actor(worker_actor_name(i, gen), namespace=ns) for i in range(self.n)
        ]

    def key(self, rel: str) -> str:
        rel = rel.lstrip("/")
        if not self.prefix:
            return rel
        if rel.startswith(self.prefix):
            return rel
        return f"{self.prefix}{rel}"

    def _owner(self, key: str):
        return self._workers[owner_index(key, self.n, prefix=self.prefix)]

    def _owners_for_prefix(self, prefix: str) -> list:
        rel = self.key(prefix)
        if self.prefix and rel.startswith(self.prefix):
            rel = rel[len(self.prefix) :]
        probe = rel if rel.endswith("/") else f"{rel}/"
        if _MONTH_RE.search(probe):
            return [self._workers[owner_index(prefix, self.n, prefix=self.prefix)]]
        return self._workers

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        ray.get(self._owner(key).put_bytes.remote(key, data, content_type))

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream"):
        data = Path(path).read_bytes()
        self.put_bytes(key, data, content_type)

    def put_df_parquet(self, key: str, df: pd.DataFrame):
        ray.get(self._owner(key).put_df.remote(key, _parquet_bytes(df)))

    def get_bytes(self, key: str, *, attempts: int = 1) -> bytes:
        return ray.get(self._owner(key).get_bytes.remote(key))

    def download_file(self, key: str, path: Path, *, attempts: int = 1):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.get_bytes(key))

    def get_df_parquet(
        self,
        key: str,
        *,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame:
        return _read_parquet_bytes(
            ray.get(self._owner(key).get_df.remote(key, columns, filters))
        )

    def get_dfs_parquet_parallel(
        self, keys: list[str], *, columns: list[str] | None = None, max_workers: int = 16
    ) -> list[pd.DataFrame]:
        if not keys:
            return []
        refs = [self._owner(k).get_df.remote(k, columns, None) for k in keys]
        return [_read_parquet_bytes(blob) for blob in ray.get(refs)]

    def exists(self, key: str, *, attempts: int = 1) -> bool:
        return bool(ray.get(self._owner(key).exists.remote(key)))

    def list_keys(self, prefix: str, *, attempts: int = 1) -> list[str]:
        refs = [w.list_keys.remote(prefix) for w in self._owners_for_prefix(prefix)]
        seen: set[str] = set()
        out: list[str] = []
        for chunk in ray.get(refs):
            for k in chunk:
                if k not in seen:
                    seen.add(k)
                    out.append(k)
        return sorted(out)

    def list_common_prefixes(self, prefix: str, *, attempts: int = 1) -> list[str]:
        refs = [w.list_common_prefixes.remote(prefix) for w in self._workers]
        seen: set[str] = set()
        out: list[str] = []
        for chunk in ray.get(refs):
            for p in chunk:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return sorted(out)

    def delete_keys(self, keys):
        keys = list(keys)
        if not keys:
            return
        buckets: list[list[str]] = [[] for _ in range(self.n)]
        for key in keys:
            buckets[owner_index(key, self.n, prefix=self.prefix)].append(key)
        refs = [
            self._workers[i].delete_keys.remote(chunk)
            for i, chunk in enumerate(buckets)
            if chunk
        ]
        if refs:
            ray.get(refs)

    def uri(self, key: str) -> str:
        return f"seed://{self.key(key)}"


def _seed_lake_info(*, namespace: str | None = None) -> tuple[int, str]:
    if not ray.is_initialized():
        return 0, ""
    try:
        gate = _get_actor(SEED_LAKE_GATE_NAME, namespace=namespace)
    except ValueError:
        return 0, ""
    n = int(ray.get(gate.worker_count.remote()))
    try:
        gen = str(ray.get(gate.generation.remote()) or "")
    except Exception:
        gen = ""
    return n, gen


def seed_lake_worker_count(*, namespace: str | None = None) -> int:
    n, _gen = _seed_lake_info(namespace=namespace)
    return n


def open_lake(cfg: MarketsConfig, *, scratch: bool | None = None):
    """MinIO unless this seed task opted into scratch (``scratch=True`` / cfg_d flag)."""
    use = bool(_scratch.get()) if scratch is None else bool(scratch)
    if not use:
        return LakeStore(cfg)
    n = seed_lake_worker_count(namespace=_actor_namespace(cfg))
    if n <= 0:
        raise RuntimeError("seed scratch requested but seed lake actors are down")
    return ClusterLake(cfg, n)


def start_seed_lake(cfg: MarketsConfig, *, workers: int | None = None) -> int:
    """Spawn named scratch actors. Idempotent if a gate already exists."""
    ns = _actor_namespace(cfg)
    existing = seed_lake_worker_count(namespace=ns)
    if existing:
        logger.info("seed lake already up workers=%s", existing)
        return existing
    n = workers
    if n is None or n <= 0:
        cpus = 1
        if ray.is_initialized():
            cpus = int(ray.available_resources().get("CPU", 1) or 1)
        n = max(1, cpus)
    prefix = cfg.lake_prefix or ""
    gen = uuid.uuid4().hex[:8]
    name_opts: dict = {"lifetime": "detached"}
    if ns:
        name_opts["namespace"] = ns
    workers = [
        SeedWorker.options(name=worker_actor_name(i, gen), **name_opts).remote(i, prefix)
        for i in range(n)
    ]
    gate = SeedLakeGate.options(name=SEED_LAKE_GATE_NAME, **name_opts).remote(n, gen)
    ray.get([w.ping.remote() for w in workers])
    ray.get(gate.worker_count.remote())
    listed = seed_lake_worker_count(namespace=ns)
    if listed != n:
        raise RuntimeError(
            f"seed lake named lookup failed after start (expected {n}, got {listed})"
        )
    logger.info("seed lake started workers=%s", n)
    return n


def flush_seed_lake(cfg: MarketsConfig) -> dict:
    ns = _actor_namespace(cfg)
    n = seed_lake_worker_count(namespace=ns)
    if n <= 0:
        return {"workers": 0, "files": 0, "bytes": 0}
    cfg_d = cfg.to_dict()
    _n, gen = _seed_lake_info(namespace=ns)
    refs = [
        _get_actor(worker_actor_name(i, gen), namespace=ns).flush_minio.remote(cfg_d)
        for i in range(n)
    ]
    parts = ray.get(refs)
    out = {
        "workers": n,
        "files": sum(int(p.get("files") or 0) for p in parts),
        "bytes": sum(int(p.get("bytes") or 0) for p in parts),
    }
    logger.info("seed lake flush %s", out)
    return out


def _kill_named_workers(name_for_index, *, namespace: str | None) -> int:
    i = 0
    while True:
        try:
            actor = _get_actor(name_for_index(i), namespace=namespace)
        except ValueError:
            break
        try:
            ray.get(actor.shutdown.remote())
        except Exception as exc:
            logger.warning("seed worker %s shutdown: %s", i, exc)
        ray.kill(actor)
        i += 1
    return i


def stop_seed_lake(*, namespace: str | None = None) -> None:
    if not ray.is_initialized():
        return
    ns = namespace or _actor_namespace()
    _n, gen = _seed_lake_info(namespace=ns)
    killed = 0
    if gen:
        killed += _kill_named_workers(lambda i: worker_actor_name(i, gen), namespace=ns)
    killed += _kill_named_workers(lambda i: worker_actor_name(i, ""), namespace=ns)
    try:
        ray.kill(_get_actor(SEED_LAKE_GATE_NAME, namespace=ns))
    except ValueError:
        pass
    if killed:
        logger.info("seed lake stopped workers=%s gen=%s", killed, gen or "legacy")
