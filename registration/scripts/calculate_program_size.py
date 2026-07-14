from __future__ import annotations

import json
import subprocess

from bike_router.__about__ import __version__
from registration.scripts.source_selection import ROOT, relative_path, selected_files

OUT = ROOT / "registration" / "private" / "03_program_size.json"


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


def build_payload() -> dict[str, object]:
    files = []
    total = 0
    for path in selected_files(for_archive=False):
        size = path.stat().st_size
        total += size
        files.append({"path": relative_path(path), "bytes": size})
    return {
        "version": __version__,
        "commit": _commit(),
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
    }


def main() -> int:
    payload = build_payload()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{payload['file_count']} files, {payload['total_bytes']} bytes -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
