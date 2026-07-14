from __future__ import annotations

import sys
from pathlib import Path

from bike_router.__about__ import __program_name__

ROOT = Path(__file__).resolve().parents[2]
ABSTRACT = ROOT / "registration" / "abstract.md"
LIMIT = 900


def main() -> int:
    text = ABSTRACT.read_text(encoding="utf-8").strip()
    errors: list[str] = []
    if len(text) > LIMIT:
        errors.append(f"abstract is {len(text)} characters, limit is {LIMIT}")
    if __program_name__ not in text:
        errors.append("exact program name is missing")
    if "Языки программирования: Python, JavaScript" not in text:
        errors.append("programming languages are missing")
    if "Объем программы:" not in text:
        errors.append("program size reference is missing")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"abstract OK: {len(text)} characters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
