"""Package initialiser: makes the project root importable so modules can
`import config` regardless of the working directory."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
