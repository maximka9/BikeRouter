from __future__ import annotations

import hashlib
from pathlib import Path

from bike_router.__about__ import __author__, __copyright_holder__, __program_name__, __version__
from registration.scripts.source_selection import ROOT, relative_path, selected_files

OUT_DIR = ROOT / "registration" / "private"
PDF = OUT_DIR / "02_deposit.pdf"
BACKUP_PDF = OUT_DIR / "02_deposit_backup.pdf"


def _line_range(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not text:
        return "1"
    return f"1-{len(text)}"


def _snippets() -> list[Path]:
    preferred = [
        ROOT / "bike_router" / "__main__.py",
        ROOT / "bike_router" / "api.py",
        ROOT / "bike_router" / "models.py",
        ROOT / "bike_router" / "engine.py",
        ROOT / "bike_router" / "services" / "graph.py",
        ROOT / "bike_router" / "services" / "routing.py",
        ROOT / "bike_router" / "services" / "elevation.py",
        ROOT / "bike_router" / "services" / "green.py",
        ROOT / "bike_router" / "services" / "surface_ai.py",
        ROOT / "bike_router" / "services" / "surface_resolve.py",
        ROOT / "bike_router" / "frontend" / "app.js",
    ]
    return [path for path in preferred if path.is_file()]


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
    for path in _snippets():
        rel = relative_path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        kind = "JavaScript" if path.suffix == ".js" else "Python"
        lines.extend(
            [
                f"### {rel}",
                "",
                f"Назначение: существенный авторский фрагмент {kind}.",
                f"Диапазон строк: {_line_range(path)}.",
                f"Версия: {__version__}.",
                f"SHA-256 исходного файла: {digest}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Визуальные отображения",
            "",
            "Веб-интерфейс программы порождает главный экран с интерактивной картой, форму задания двух точек, отображение нескольких альтернативных маршрутов, сравнение метрик, результаты пешеходного и велосипедного профилей, отображение покрытия, озеленения и OpenAPI-документацию. Скриншоты помещаются в этот раздел при финальной сборке из рабочей демонстрационной среды.",
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
    line_height = 13
    y = top
    c.setTitle("BikeRouter deposited materials")
    for raw in text.splitlines():
        line = raw.rstrip()
        if y < 48:
            c.showPage()
            y = top
        if line.startswith("# "):
            c.setFont(font_name, 14)
            c.drawString(left, y, line[2:110])
        elif line.startswith("## "):
            c.setFont(font_name, 12)
            c.drawString(left, y, line[3:120])
        elif line.startswith("### "):
            c.setFont(font_name, 10)
            c.drawString(left, y, line[4:130])
        else:
            c.setFont(font_name, 9)
            c.drawString(left, y, line[:135])
        y -= line_height
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
