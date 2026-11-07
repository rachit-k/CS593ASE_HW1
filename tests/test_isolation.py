"""The safety net itself: tests must not touch the user's key, library, or a real service."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import httpx
import pytest

from app import arxiv_client, citations, config, pdfs

REPO = Path(__file__).resolve().parent.parent


def test_library_lives_in_a_temp_folder(tmp_path):
    assert config.settings.data_dir.is_relative_to(tmp_path)
    assert not config.settings.data_dir.is_relative_to(REPO)


def test_the_users_api_key_is_not_loaded():
    assert os.environ["GEMINI_API_KEY"] == ""
    assert config.load_settings().gemini_api_key is None
    assert config.settings.gemini_api_key is None


def test_real_connections_are_refused():
    with pytest.raises(AssertionError, match="real network connection"):
        socket.create_connection(("127.0.0.1", 9))
    with socket.socket() as sock, pytest.raises(AssertionError, match="real network connection"):
        sock.connect(("127.0.0.1", 9))
    with pytest.raises(AssertionError, match="real network connection"):
        httpx.get("http://127.0.0.1:9/")


@pytest.mark.parametrize(
    "call",
    [
        lambda: arxiv_client.search("testing"),
        lambda: pdfs.download_arxiv_pdf("2401.12345"),
        lambda: citations.citation_counts(["2401.12345"]),
    ],
    ids=["arxiv-search", "arxiv-pdf", "semantic-scholar"],
)
def test_services_must_be_faked_explicitly(call):
    with pytest.raises(AssertionError, match="unexpected"):
        call()

