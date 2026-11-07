"""config.py: settings from the environment, plus the files that keep secrets out of git."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app import config

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_when_nothing_is_set(clean_env):
    settings = config.load_settings()
    assert settings.gemini_api_key is None
    assert settings.gemini_model == "gemini-3.8-flash"
    assert settings.data_dir == REPO / "data"


def test_values_come_from_the_environment(clean_env, tmp_path):
    clean_env.setenv("GEMINI_API_KEY", "abc123")
    clean_env.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    clean_env.setenv("DATA_DIR", str(tmp_path))
    settings = config.load_settings()
    assert settings.gemini_api_key == "abc123"
    assert settings.gemini_model == "gemini-3.5-flash-lite"
    assert settings.data_dir == tmp_path


def test_stray_spaces_around_values_are_ignored(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "  abc123  ")
    clean_env.setenv("GEMINI_MODEL", " some-model ")
    settings = config.load_settings()
    assert settings.gemini_api_key == "abc123"
    assert settings.gemini_model == "some-model"


def test_blank_values_count_as_not_set(clean_env):
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "DATA_DIR"):
        clean_env.setenv(name, "   ")
    settings = config.load_settings()
    assert settings.gemini_api_key is None
    assert settings.gemini_model == config.DEFAULT_MODEL
    assert settings.data_dir == REPO / "data"


def test_relative_data_dir_is_relative_to_the_repo_not_the_working_directory(clean_env, tmp_path):
    clean_env.chdir(tmp_path)
    clean_env.setenv("DATA_DIR", "some/folder")
    assert config.load_settings().data_dir == REPO / "some" / "folder"


def test_home_folder_shorthand_is_expanded(clean_env):
    clean_env.setenv("DATA_DIR", "~/papers")
    assert config.load_settings().data_dir == Path.home() / "papers"


def test_database_and_pdfs_live_inside_the_data_dir(tmp_path):
    settings = config.Settings(None, "model", tmp_path)
    assert settings.db_path == tmp_path / "library.db"
    assert settings.pdf_dir == tmp_path / "pdfs"


def test_env_template_documents_every_setting_and_holds_no_key():
    lines = (REPO / ".env.example").read_text().splitlines()
    settings = dict(line.split("=", 1) for line in lines if line.strip() and not line.startswith("#"))
    assert set(settings) == {"GEMINI_API_KEY", "GEMINI_MODEL", "DATA_DIR"}
    assert settings["GEMINI_API_KEY"] == ""


def test_secrets_and_the_library_are_gitignored():
    ignored = (REPO / ".gitignore").read_text().splitlines()
    assert ".env" in ignored
    assert "data/" in ignored


# --- Found by mutation testing: nothing checked that .env is read at all ------------------------


def settings_seen_by_a_fresh_copy(tmp_path, dotenv_text, **environment):
    """Start config.py in a new process from a private copy, so it can only find this test's .env."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").touch()
    shutil.copy(config.__file__, tmp_path / "app" / "config.py")
    (tmp_path / ".env").write_text(dotenv_text)
    others = {name: value for name, value in os.environ.items() if name not in ("GEMINI_API_KEY", "GEMINI_MODEL", "DATA_DIR")}
    code = "from app import config; print(config.settings.gemini_api_key, config.settings.gemini_model)"
    finished = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=others | environment, capture_output=True, text=True, timeout=60, check=True
    )
    return finished.stdout.split()


def test_settings_are_read_from_the_dotenv_file(tmp_path):
    assert settings_seen_by_a_fresh_copy(tmp_path, "GEMINI_API_KEY=from-file\nGEMINI_MODEL=model-from-file\n") == ["from-file", "model-from-file"]


def test_a_real_environment_variable_beats_the_dotenv_file(tmp_path):
    seen = settings_seen_by_a_fresh_copy(tmp_path, "GEMINI_API_KEY=from-file\nGEMINI_MODEL=model-from-file\n", GEMINI_MODEL="model-from-environment")
    assert seen == ["from-file", "model-from-environment"]
