from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

from bike_router.__about__ import __registration_tag__, __version__

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "registration" / "manifest.txt"
OUT_DIR = ROOT / "registration" / "private"
ZIP_NAME = f"BikeRouter-{__version__}-registration.zip"


def _run(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def _check_git() -> list[str]:
    errors: list[str] = []
    status = _run(["git", "status", "--porcelain"])
    if status:
        errors.append("git working tree is not clean")
    tags = _run(["git", "tag", "--points-at", "HEAD"]).splitlines()
    if __registration_tag__ not in tags:
        errors.append(f"HEAD is not tagged {__registration_tag__}")
    return errors


def _manifest_files() -> list[Path]:
    files: list[Path] = []
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        path = ROOT / item
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(item)
    return sorted(set(files), key=lambda p: p.relative_to(ROOT).as_posix())


def main() -> int:
    errors = _check_git()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    subprocess.check_call([sys.executable, "-m", "registration.scripts.check_abstract"], cwd=ROOT)
    subprocess.check_call([sys.executable, "-m", "registration.scripts.calculate_program_size"], cwd=ROOT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archive = OUT_DIR / ZIP_NAME
    files = _manifest_files()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            zf.write(path, rel)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (archive.with_suffix(archive.suffix + ".sha256")).write_text(f"{digest}  {ZIP_NAME}\n", encoding="utf-8")
    (OUT_DIR / "manifest.files.txt").write_text(
        "\n".join(p.relative_to(ROOT).as_posix() for p in files) + "\n",
        encoding="utf-8",
    )
    print(f"{archive}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
