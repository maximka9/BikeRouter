from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

from bike_router.__about__ import __registration_tag__, __version__
from registration.scripts.source_selection import ROOT, relative_path, selected_files

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
    files = selected_files(for_archive=True)
    if not files:
        raise RuntimeError("source selection returned no files")
    return files


def _write_manifest_csv(files: list[Path]) -> Path:
    path = OUT_DIR / "04_source_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "bytes", "sha256"])
        for item in files:
            writer.writerow(
                [
                    relative_path(item),
                    item.stat().st_size,
                    hashlib.sha256(item.read_bytes()).hexdigest(),
                ]
            )
    return path


def _write_checksums(files: list[Path], archive: Path | None = None) -> Path:
    path = OUT_DIR / "05_checksums.txt"
    rows = []
    for item in files:
        rows.append(f"{hashlib.sha256(item.read_bytes()).hexdigest()}  {relative_path(item)}")
    if archive is not None:
        rows.append(f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def main() -> int:
    errors = _check_git()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    subprocess.check_call([sys.executable, "-m", "registration.scripts.check_abstract"], cwd=ROOT)
    subprocess.check_call(
        [sys.executable, "-m", "registration.scripts.calculate_program_size"], cwd=ROOT
    )
    subprocess.check_call(
        [sys.executable, "-m", "registration.scripts.generate_deposit_pdf"], cwd=ROOT
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "01_abstract.txt").write_text(
        (ROOT / "registration" / "abstract.md").read_text(encoding="utf-8").strip() + "\n",
        encoding="utf-8",
    )
    archive = OUT_DIR / ZIP_NAME
    files = _manifest_files()
    _write_manifest_csv(files)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, relative_path(path))

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (archive.with_suffix(archive.suffix + ".sha256")).write_text(
        f"{digest}  {ZIP_NAME}\n", encoding="utf-8"
    )
    _write_checksums(files, archive)
    status_after = _run(["git", "status", "--porcelain"])
    if status_after:
        print("ERROR: build changed tracked files", file=sys.stderr)
        print(status_after, file=sys.stderr)
        return 1
    print(f"{archive}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
