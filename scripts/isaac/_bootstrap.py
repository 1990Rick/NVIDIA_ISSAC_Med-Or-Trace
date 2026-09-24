"""Make ``medortrace`` importable when Isaac scripts are run from a source checkout.

Isaac Sim's ``python.sh`` runs its own interpreter, so the repository root is
put on ``sys.path`` explicitly (no ``pip install -e`` into Isaac's Python needed).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
