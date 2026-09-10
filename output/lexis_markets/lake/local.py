"""Filesystem lake with the same key layout as MinIO ``LakeStore``.

Used as the per-actor scratch store during seed. Daily EOD keeps using S3.
"""
from __future__ import annotations

import io
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import pandas as pd


class LocalDirLake:
    """Duck-types ``LakeStore`` against a directory tree."""

    def __init__(self, root: Path, *, prefix: str = ""):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.prefix = (prefix or "").strip()
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix = f"{self.prefix}/"
        self.bucket = "local"
        self.client = None

    def key(self, rel: str) -> str:
        rel = rel.lstrip("/")
        if not self.prefix:
            return rel
        if rel.startswith(self.prefix):
            return rel
        return f"{self.prefix}{rel}"

    def _path(self, key: str) -> Path:
        rel = self.key(key)
        parts = [p for p in rel.split("/") if p and p != ".."]
        return self.root.joinpath(*parts)

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream"):
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)

    def put_df_parquet(self, key: str, df: pd.DataFrame):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.put_bytes(key, buf.getvalue())

    def get_bytes(self, key: str, *, attempts: int = 1) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def download_file(self, key: str, path: Path, *, attempts: int = 1):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), path)

    def get_df_parquet(
        self,
        key: str,
        *,
        columns: list[str] | None = None,
        filters: list | None = None,
    ) -> pd.DataFrame:
        return pd.read_parquet(
            io.BytesIO(self.get_bytes(key)),
            columns=columns,
            filters=filters,
            engine="pyarrow",
        )

    def get_dfs_parquet_parallel(
        self, keys: list[str], *, columns: list[str] | None = None, max_workers: int = 16
    ) -> list[pd.DataFrame]:
        if not keys:
            return []
        workers = min(max_workers, len(keys))

        def load(key: str) -> pd.DataFrame:
            return self.get_df_parquet(key, columns=columns)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(load, keys))

    def exists(self, key: str, *, attempts: int = 1) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str, *, attempts: int = 1) -> list[str]:
        full = self.key(prefix)
        root = self.root
        start = root / full if full else root
        if start.is_file():
            return [full]
        if not start.exists():
            parent = start.parent
            name = start.name
            if parent.is_dir():
                return [
                    str(p.relative_to(root)).replace("\\", "/")
                    for p in parent.glob(name + "*")
                    if p.is_file()
                ]
            return []
        out: list[str] = []
        for path in start.rglob("*"):
            if path.is_file():
                out.append(str(path.relative_to(root)).replace("\\", "/"))
        return sorted(out)

    def list_common_prefixes(self, prefix: str, *, attempts: int = 1) -> list[str]:
        rel = prefix if prefix.endswith("/") or prefix == "" else f"{prefix}/"
        full = self.key(rel)
        dirpath = self.root / full if full else self.root
        if not dirpath.is_dir():
            return []
        out: list[str] = []
        for child in sorted(dirpath.iterdir()):
            if child.is_dir():
                out.append(f"{full}{child.name}/")
        return out

    def delete_keys(self, keys: Iterable[str]):
        for key in keys:
            path = self._path(key)
            if path.is_file():
                path.unlink()

    def uri(self, key: str) -> str:
        return f"file://{self._path(key)}"
