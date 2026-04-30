"""Chair falls data-prep package."""

from importlib.metadata import version

try:
    __version__ = version("ld-chair-falls")
except Exception:  # pragma: no cover - editable installs during bootstrap
    __version__ = "0.0.0"
