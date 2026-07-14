from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "registration" / "source_selection.yml"


@dataclass(frozen=True)
class SourceSelection:
    program_include: tuple[str, ...]
    archive_include: tuple[str, ...]
    source_suffixes: tuple[str, ...]
    archive_suffixes: tuple[str, ...]
    exclude: tuple[str, ...]
    archive_extra_exclude: tuple[str, ...]


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("source selection fields must be lists")
    return tuple(str(item).replace("\\", "/").strip("/") for item in value if str(item).strip())


def load_selection(path: Path = CONFIG) -> SourceSelection:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return SourceSelection(
        program_include=_as_tuple(data.get("program_include")),
        archive_include=_as_tuple(data.get("archive_include")),
        source_suffixes=_as_tuple(data.get("source_suffixes")),
        archive_suffixes=_as_tuple(data.get("archive_suffixes")),
        exclude=_as_tuple(data.get("exclude")),
        archive_extra_exclude=_as_tuple(data.get("archive_extra_exclude")),
    )


def _is_excluded(rel: str, patterns: tuple[str, ...]) -> bool:
    parts = rel.split("/")
    for item in patterns:
        item = item.rstrip("/")
        if rel == item or rel.startswith(item + "/") or item in parts or fnmatch(rel, item):
            return True
    return False


def _iter_path(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (item for item in path.rglob("*") if item.is_file())
    else:
        raise FileNotFoundError(path.relative_to(ROOT).as_posix())


def selected_files(
    *, for_archive: bool = False, selection: SourceSelection | None = None
) -> list[Path]:
    selection = selection or load_selection()
    suffixes = selection.archive_suffixes if for_archive else selection.source_suffixes
    roots = selection.archive_include if for_archive else selection.program_include
    excludes = selection.archive_extra_exclude if for_archive else selection.exclude
    files: set[Path] = set()
    for raw in roots:
        for path in _iter_path(ROOT / raw):
            rel = path.relative_to(ROOT).as_posix()
            if _is_excluded(rel, excludes):
                continue
            if path.suffix.lower() in suffixes or any(rel.endswith(suffix) for suffix in suffixes):
                files.add(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()
