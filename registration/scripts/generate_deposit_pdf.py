from __future__ import annotations

import hashlib
from pathlib import Path

from bike_router.__about__ import __author__, __copyright_holder__, __program_name__, __version__
from registration.scripts.source_selection import ROOT, relative_path, selected_files

OUT_DIR = ROOT / "registration" / "private"
PDF = OUT_DIR / "02_deposit.pdf"
BACKUP_PDF = OUT_DIR / "02_deposit_backup.pdf"
CODE_WRAP = 96


def _code_fragments() -> list[tuple[Path, int, int | None]]:
    return [
        (ROOT / "bike_router" / "__main__.py", 1, None),
        (ROOT / "bike_router" / "api.py", 1, None),
        (ROOT / "bike_router" / "models.py", 1, None),
        (ROOT / "bike_router" / "engine.py", 1, None),
        (ROOT / "bike_router" / "services" / "graph.py", 1, None),
        (ROOT / "bike_router" / "services" / "routing.py", 1, None),
        (ROOT / "bike_router" / "services" / "routing_criteria.py", 1, None),
        (ROOT / "bike_router" / "services" / "elevation.py", 1, None),
        (ROOT / "bike_router" / "services" / "green.py", 1, None),
        (ROOT / "bike_router" / "services" / "heat.py", 1, None),
        (ROOT / "bike_router" / "services" / "stress.py", 1, None),
        (ROOT / "bike_router" / "services" / "surface_resolve.py", 1, None),
        (ROOT / "bike_router" / "services" / "surface_ai.py", 1, None),
        (ROOT / "bike_router" / "frontend" / "app.js", 1, 260),
        (ROOT / "bike_router" / "frontend" / "app.js", 780, 1180),
        (ROOT / "bike_router" / "frontend" / "app.js", 1420, 1680),
    ]


def _line_range(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not text:
        return "1"
    return f"1-{len(text)}"


def _wrap_code_line(number: int, line: str) -> list[str]:
    prefix = f"{number:5d}: "
    if not line:
        return [prefix]
    chunks = [line[i : i + CODE_WRAP] for i in range(0, len(line), CODE_WRAP)]
    out = [prefix + chunks[0]]
    out.extend("       " + chunk for chunk in chunks[1:])
    return out


def _code_block(path: Path, start: int, end: int | None) -> list[str]:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if end is None:
        end = len(raw_lines)
    selected = raw_lines[start - 1 : end]
    out: list[str] = ["```text"]
    for offset, line in enumerate(selected, start=start):
        out.extend(_wrap_code_line(offset, line.expandtabs(4)))
    out.append("```")
    return out


def _markdown() -> str:
    files = selected_files(for_archive=False)
    lines = [
        "# ДЕПОНИРУЕМЫЕ МАТЕРИАЛЫ, ИДЕНТИФИЦИРУЮЩИЕ ПРОГРАММУ ДЛЯ ЭВМ",
        "",
        __program_name__,
        "",
        f"Версия: {__version__}",
        "Год завершения: 2026",
        "",
        f"Автор: {__author__}",
        f"Правообладатель: {__copyright_holder__}",
        "",
        "## Содержание",
        "",
        "1. Титульный лист",
        "2. Структура программы",
        "3. Существенные авторские фрагменты Python",
        "4. Существенные авторские фрагменты JavaScript",
        "5. Визуальные отображения, порождаемые программой",
        "",
        "## Структура программы",
        "",
        "Программа состоит из пакета bike_router, HTTP API FastAPI, маршрутизационного ядра, сервисов обработки геоданных, машинного восстановления покрытия, фоновых заданий, конфигурационных политик и веб-интерфейса.",
        "",
        "## Существенные авторские фрагменты",
        "",
    ]
    for path, start, end in _code_fragments():
        if not path.is_file():
            continue
        rel = relative_path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        kind = "JavaScript" if path.suffix == ".js" else "Python"
        actual_end = (
            end
            if end is not None
            else len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        )
        lines.extend(
            [
                f"### {rel}",
                "",
                f"Назначение: существенный авторский фрагмент {kind}.",
                f"Диапазон строк: {start}-{actual_end}.",
                f"Версия: {__version__}.",
                f"SHA-256 исходного файла: {digest}.",
                "",
            ]
        )
        lines.extend(_code_block(path, start, end))
        lines.append("")
    lines.extend(
        [
            "## Визуальные отображения",
            "",
            "Визуальные отображения программы не включены в настоящий комплект; идентификация выполняется по исходному тексту существенных авторских фрагментов.",
            "",
            "## Перечень авторских файлов",
            "",
        ]
    )
    for path in files:
        lines.append(f"- {relative_path(path)}")
    lines.append("")
    return "\n".join(lines)


def _font_name() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for font_path in candidates:
        if font_path.is_file():
            pdfmetrics.registerFont(TTFont("BikeRouterFont", str(font_path)))
            return "BikeRouterFont"
    return "Helvetica"


def _write_pdf(path: Path, text: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    font_name = _font_name()
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    left = 48
    top = height - 48
    line_height = 12
    y = top
    c.setTitle("BikeRouter deposited materials")
    page = 1
    for raw in text.splitlines():
        line = raw.rstrip()
        if y < 48:
            c.setFont(font_name, 8)
            c.drawRightString(width - 48, 24, str(page))
            c.showPage()
            page += 1
            y = top
        if line.startswith("# "):
            c.setFont(font_name, 14)
            c.drawString(left, y, line[2:])
        elif line.startswith("## "):
            c.setFont(font_name, 12)
            c.drawString(left, y, line[3:])
        elif line.startswith("### "):
            c.setFont(font_name, 10)
            c.drawString(left, y, line[4:])
        elif line.startswith("```"):
            c.setFont(font_name, 8)
            c.drawString(left, y, "")
            y -= line_height
            continue
        elif len(line) >= 7 and line[:5].strip().isdigit() and line[5:7] == ": ":
            c.setFont(font_name, 6)
            c.drawString(left, y, line)
        else:
            c.setFont(font_name, 9)
            c.drawString(left, y, line)
        y -= line_height
    c.setFont(font_name, 8)
    c.drawRightString(width - 48, 24, str(page))
    c.save()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "02_deposit.md"
    text = _markdown()
    md_path.write_text(text, encoding="utf-8")
    _write_pdf(PDF, text)
    _write_pdf(BACKUP_PDF, text)
    print(PDF)
    print(BACKUP_PDF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
