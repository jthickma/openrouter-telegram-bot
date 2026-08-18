"""Test configuration."""

from __future__ import annotations

import sys
from pathlib import Path

BOT_DIRECTORY = Path(__file__).resolve().parents[1] / "bot"
sys.path.insert(0, str(BOT_DIRECTORY))
