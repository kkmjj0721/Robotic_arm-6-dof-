from __future__ import annotations

from config import Config
from sim import run_viewer


def main() -> int:
    run_viewer(Config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
