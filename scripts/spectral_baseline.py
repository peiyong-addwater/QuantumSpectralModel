#!/usr/bin/env python3
"""Compatibility wrapper for scripts/classical_baseline.py."""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.classical_baseline import main  # noqa: E402


if __name__ == "__main__":
    main()
