from __future__ import annotations

import re
import sys
from pathlib import Path

from bike_router.__about__ import __program_name__
from registration.scripts.calculate_program_size import build_payload

ROOT = Path(__file__).resolve().parents[2]
ABSTRACT = ROOT / "registration" / "abstract.md"
LIMIT = 900
SIZE_RE = re.compile(r"Объем программы:\s*([0-9][0-9 ]*)\s+байт\.", re.IGNORECASE)


def main() -> int:
    text = ABSTRACT.read_text(encoding="utf-8").strip()
    errors: list[str] = []
    if len(text) > LIMIT:
        errors.append(f"abstract is {len(text)} characters, limit is {LIMIT}")
    if __program_name__ not in text:
        errors.append("exact program name is missing")
    if "Языки программирования: Python, JavaScript" not in text:
        errors.append("programming languages are missing")
    size_match = SIZE_RE.search(text)
    if not size_match:
        errors.append("program size must be a numeric byte value in the abstract")
    else:
        actual_size = int(str(build_payload()["total_bytes"]))
        abstract_size = int(size_match.group(1).replace(" ", ""))
        if abstract_size != actual_size:
            errors.append(f"abstract size {abstract_size} does not match calculated {actual_size}")
    lowered = text.lower()
    forbidden = ("см. файл", "указан отдельно", "отдельно")
    for marker in forbidden:
        if marker in lowered:
            errors.append(f"abstract size must not use reference wording: {marker}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"abstract OK: {len(text)} characters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
