from __future__ import annotations

import json
import subprocess
from pathlib import Path

from bike_router.__about__ import __version__

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "registration" / "05_program_size.json"
INCLUDE_SUFFIXES = {".py", ".js", ".html", ".css", ".json"}
INCLUDE_ROOTS = ("bike_router",)
EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "tests",
    "docs",
    "data",
    "experiment_outputs",
    "private",
    "out",
    "vendor",
}


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    return (
        path.is_file()
        and rel.parts[0] in INCLUDE_ROOTS
        and path.suffix.lower() in INCLUDE_SUFFIXES
        and not (parts & EXCLUDE_PARTS)
    )


def build_payload() -> dict[str, object]:
    files = []
    total = 0
    for path in sorted((ROOT / "bike_router").rglob("*"), key=lambda p: p.relative_to(ROOT).as_posix()):
        if not _included(path):
            continue
        size = path.stat().st_size
        total += size
        files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": size})
    return {
        "version": __version__,
        "commit": _commit(),
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
    }


def main() -> int:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{payload['file_count']} files, {payload['total_bytes']} bytes -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
