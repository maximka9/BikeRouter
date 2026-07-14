from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "registration" / "source_selection.yml"


@dataclass(frozen=True)
class SourceSelection:
    include: tuple[str, ...]
    include_files: tuple[str, ...]
    source_suffixes: tuple[str, ...]
    archive_suffixes: tuple[str, ...]
    exclude: tuple[str, ...]


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("source selection fields must be lists")
    return tuple(str(item).replace("\\", "/").strip("/") for item in value if str(item).strip())


def load_selection(path: Path = CONFIG) -> SourceSelection:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return SourceSelection(
        include=_as_tuple(data.get("include")),
        include_files=_as_tuple(data.get("include_files")),
        source_suffixes=_as_tuple(data.get("source_suffixes")),
        archive_suffixes=_as_tuple(data.get("archive_suffixes")),
        exclude=_as_tuple(data.get("exclude")),
    )


def _is_excluded(rel: str, selection: SourceSelection) -> bool:
    parts = rel.split("/")
    for item in selection.exclude:
        if rel == item or rel.startswith(item.rstrip("/") + "/") or item in parts:
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
    roots = [*selection.include, *selection.include_files]
    files: set[Path] = set()
    for raw in roots:
        for path in _iter_path(ROOT / raw):
            rel = path.relative_to(ROOT).as_posix()
            if _is_excluded(rel, selection):
                continue
            if path.suffix.lower() in suffixes or any(rel.endswith(suffix) for suffix in suffixes):
                files.add(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()
