from pathlib import Path

# Expose the existing src/ tree as the bjfg package root.
__path__ = [str(Path(__file__).resolve().parent.parent / "src")]
