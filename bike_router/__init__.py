"""Compatibility shim for running ``python -m bike_router...`` from here.

When the current directory is the package directory itself, Python looks for a
nested ``bike_router`` package. This shim points package imports back to the
real package root one level above. It is ignored when commands are run from the
repository parent, where the real package is found first.
"""

from __future__ import annotations

from pathlib import Path

_real_package_root = Path(__file__).resolve().parents[1]
__path__ = [str(_real_package_root)]
__file__ = str(_real_package_root / "__init__.py")

