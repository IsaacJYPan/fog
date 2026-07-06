"""CLI: build all constant tables (spec §1). Thin wrapper over constants/build.py."""
from __future__ import annotations

from fog.constants.build import main

if __name__ == "__main__":
    main()
