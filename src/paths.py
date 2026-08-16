"""
Project paths, resolved independently of the current working directory.

The notebook used to rely on `os.getcwd()`, which forced it to run from `Code/`.
This module anchors every path to the repository root (the folder that contains
this `src/` package), so code works no matter where it is launched from.
"""

from pathlib import Path

# src/paths.py -> parents[0] == src/ , parents[1] == repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data"       # raw market-data workbooks
MODELS_DIR = PROJECT_ROOT / "Code"     # serialized .pkl models + the notebook


def find_project_root(markers=(".git", "pyproject.toml", "requirements.txt")):
    """Return the nearest ancestor of the current directory that looks like the repo root.

    Useful for contexts without ``__file__`` (e.g. a Jupyter cell): walk upward
    from the working directory until one of ``markers`` is found.
    """
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return start
