"""Settings, read from environment variables and from a `.env` file in the repo root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-3.8-flash"

# Variables already set in the real environment win over values in .env.
load_dotenv(BASE_DIR / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    data_dir: Path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "library.db"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"


def load_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "").strip() or "data").expanduser()
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir  # relative paths are relative to the repo root
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
        gemini_model=os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL,
        data_dir=data_dir,
    )


# Other modules read `config.settings` when they need it (instead of copying it at
# import time), so tests can swap it out:
#     monkeypatch.setattr(config, "settings", config.Settings(None, "fake-model", tmp_path))
settings = load_settings()
