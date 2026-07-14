from __future__ import annotations

import tomllib

from bike_router.__about__ import __version__
from registration.scripts.source_selection import relative_path, selected_files


def test_pyproject_version_matches_about() -> None:
    with open("pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["version"] == __version__


def test_registration_source_selection_is_resolved() -> None:
    files = selected_files(for_archive=True)
    assert files
    rels = [relative_path(path) for path in files]
    assert all(path.is_file() for path in files)
    assert not any("frontend/vendor" in rel for rel in rels)
    assert not any(rel.startswith("tests/") for rel in rels)
