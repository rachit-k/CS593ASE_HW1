"""Contract tests: the fakes, the docs, and the project's own rules stay true to what they describe.

A suite built on fakes can pass while the real service or the design has moved on. These compare the
fakes with recorded responses and with the real SDK's schema, the code with its docs, and the code
with the rules in CLAUDE.md.

The mutation tool leaves this file out. A test that reads source code would notice any change to it,
which says nothing about whether the suite catches a change in behavior.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import sys
from importlib import metadata
from pathlib import Path
from xml.etree import ElementTree

import httpx
import pydantic
import pytest
from fastapi.routing import APIRoute
from google.genai import errors

from app import arxiv_client
from app.main import app
from helpers import FakeGemini, atom_entry, atom_feed, recorded
from test_citations import counts_reply

ROOT = Path(__file__).parent.parent
APP_MODULES = sorted((ROOT / "app").glob("*.py"))
TEMPLATES = sorted((ROOT / "app" / "templates").rglob("*.html"))
ATOM = "{http://www.w3.org/2005/Atom}"


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def imported_modules(path: Path) -> set[str]:
    """The full dotted name of everything a Python file imports, wherever in the file it does."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module)
    return names


# --- The fakes match what the real services send ------------------------------------------------


def child_tags(element: ElementTree.Element) -> set[str]:
    return {child.tag for child in element}


def test_the_fake_arxiv_feed_only_uses_what_a_real_feed_has():
    fake, real = ElementTree.fromstring(atom_feed()), ElementTree.fromstring(recorded("arxiv_no_results.xml"))
    assert child_tags(fake) <= child_tags(real)


def test_the_fake_arxiv_entry_only_uses_what_a_real_entry_has():
    fake = ElementTree.fromstring(atom_feed(atom_entry())).find(f"{ATOM}entry")
    real = ElementTree.fromstring(recorded("arxiv_three_papers.xml")).find(f"{ATOM}entry")
    assert child_tags(fake) <= child_tags(real)
    assert {f"{ATOM}id", f"{ATOM}title", f"{ATOM}summary", f"{ATOM}published", f"{ATOM}author"} <= child_tags(fake)


def test_a_fake_entry_and_a_real_one_parse_to_the_same_kind_of_result():
    fake = dataclasses.asdict(arxiv_client.parse_feed(atom_feed(atom_entry()))[0])
    real = dataclasses.asdict(arxiv_client.parse_feed(recorded("arxiv_three_papers.xml"))[0])
    assert {name: type(value) for name, value in fake.items()} == {name: type(value) for name, value in real.items()}
    assert all(fake.values()) and all(real.values())


def test_the_fake_citation_reply_has_the_same_shape_as_a_real_one():
    real = json.loads(recorded("semantic_scholar_batch.json"))
    fake = counts_reply(5, None)
    assert None in real and None in fake  # an unknown paper is null in its place, not left out
    assert {key for entry in fake if entry for key in entry} == {key for entry in real if entry for key in entry}


def test_the_fake_gemini_refuses_a_reply_the_real_sdk_would_not_recognize():
    # Every JSON reply a test queues goes through this check, including the hand-written ones.
    with pytest.raises(pydantic.ValidationError):
        FakeGemini().reply(200, json={"candidates": [{"text": "not where the SDK looks for it"}]})


def test_the_fake_gemini_error_reads_the_way_the_real_sdk_expects():
    reply = FakeGemini().reply_error(429, "Quota used up.", "RESOURCE_EXHAUSTED")(httpx.Request("POST", "https://fake.test"))
    error = errors.APIError(reply.status_code, reply.json())
    assert (error.code, error.message, error.status) == (429, "Quota used up.", "RESOURCE_EXHAUSTED")


# --- The docs match the code --------------------------------------------------------------------


def test_the_design_doc_lists_exactly_the_routes_the_app_has():
    documented = {
        (method, re.sub(r"\{\w+\}", "{}", path))
        for method, path in re.findall(r"^\| `(GET|POST|DELETE) (/[^`]*)` \|", read("docs/design.md"), re.MULTILINE)
    }
    actual = {
        (method, re.sub(r"\{\w+\}", "{}", route.path))
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods - {"HEAD"}
    }
    assert documented == actual


def test_the_documented_htmx_version_is_the_one_in_the_repo():
    version = re.search(r'version:"([\d.]+)"', read("app/static/htmx.min.js")).group(1)
    assert f"HTMX {version}" in read("docs/design.md")
    assert f"`htmx.min.js` ({version})" in read("CLAUDE.md")


def calls_to(name: str, path: Path) -> list[ast.Call]:
    """Every call in a Python file to a function with this name, however the file is formatted."""
    calls = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if called == name:
                calls.append(node)
    return calls


def test_the_example_settings_file_lists_every_setting_the_app_reads():
    used = {call.args[0].value for call in calls_to("getenv", ROOT / "app" / "config.py")}
    listed = set(re.findall(r"^(\w+)=", read(".env.example"), re.MULTILINE))
    assert used == listed == {"GEMINI_API_KEY", "GEMINI_MODEL", "DATA_DIR"}


def dependency_closure(requirements_file: str) -> set[str]:
    """Every distribution a clean install of a requirements file brings in, by normalized name."""
    pending = []
    for line in read(requirements_file).splitlines():
        line = line.strip()
        if line.startswith("-r "):
            pending.extend(dependency_closure(line.removeprefix("-r ")))
        elif line and not line.startswith("#"):
            pending.append(re.match(r"[\w.-]+", line).group())
    seen: set[str] = set()
    while pending:
        name = re.sub(r"[-_.]+", "-", pending.pop()).lower()
        if name not in seen:
            seen.add(name)
            try:
                requires = metadata.requires(name) or []
            except metadata.PackageNotFoundError:
                continue
            pending.extend(re.match(r"[\w.-]+", req).group() for req in requires if "extra ==" not in req)
    return seen


@pytest.mark.parametrize(
    ("folder", "requirements_file", "local"),
    [("app", "requirements.txt", {"app"}), ("tests", "requirements-dev.txt", {"app", "conftest", "helpers", "skip_listed"})],
)
def test_a_clean_install_provides_everything_the_code_imports(folder, requirements_file, local):
    installed = dependency_closure(requirements_file)
    providers = metadata.packages_distributions()
    local |= {path.stem for path in (ROOT / folder).glob("*.py")}
    modules = {name.split(".")[0] for path in (ROOT / folder).glob("*.py") for name in imported_modules(path)}
    for module in sorted(modules - local - set(sys.stdlib_module_names)):
        names = {re.sub(r"[-_.]+", "-", name).lower() for name in providers[module]}
        assert names & installed, f"{module} comes from {names}, which {requirements_file} doesn't install"


# --- The rules in CLAUDE.md hold ---------------------------------------------------------------


def test_only_the_network_modules_import_a_network_library():
    network = re.compile(r"^(httpx|requests|aiohttp|urllib3|socket|ssl|http\.client|urllib\.request|google\.genai|google$)")
    using = {path.name for path in APP_MODULES if any(network.match(name) for name in imported_modules(path))}
    assert using == {"arxiv_client.py", "citations.py", "pdfs.py", "llm.py"}


def test_pdfs_only_makes_a_request_in_the_download_function():
    tree = ast.parse(read("app/pdfs.py"))
    download = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "download_arxiv_pdf")
    inside = {id(node) for node in ast.walk(download)}
    outside = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and getattr(node.func.value, "id", None) == "httpx"
        and id(node) not in inside
    ]
    assert outside == []


def test_only_db_opens_the_database():
    assert {path.name for path in APP_MODULES if calls_to("connect", path)} == {"db.py"}


def test_model_output_is_the_only_html_that_skips_escaping():
    unescaped = [(path.name, call) for path in APP_MODULES for call in calls_to("Markup", path)]
    assert [name for name, _ in unescaped] == ["main.py"]
    assert unescaped[0][1].args[0].func.attr == "render"  # the Markdown renderer, which test_security.py attacks
    assert [path.name for path in TEMPLATES if re.search(r"\|\s*safe|autoescape\s+false", path.read_text())] == []


def test_the_pages_load_nothing_from_other_sites():
    asset = re.compile(r"""<(?:script|link|img|iframe)\b[^>]*\b(?:src|href)=["'](?:https?:)?//|url\(\s*["']?(?:https?:)?//|@import""", re.IGNORECASE)
    files = [*TEMPLATES, ROOT / "app" / "static" / "style.css"]
    assert [path.name for path in files if asset.search(path.read_text())] == []


REQUEST_ATTRIBUTE = re.compile(r"""\b(hx-get|hx-post|hx-delete|action)=["']([^"']+)["']""")


def template_requests() -> list[tuple[str, str, str]]:
    found = []
    for path in TEMPLATES:
        for attribute, url in REQUEST_ATTRIBUTE.findall(path.read_text()):
            method = {"hx-get": "GET", "hx-post": "POST", "hx-delete": "DELETE"}.get(attribute)
            found.append((path.name, method, re.sub(r"\{\{.*?\}\}", "1", url)))
    return found


def test_the_templates_make_every_kind_of_request():
    assert {method for _, method, _ in template_requests()} == {"GET", "POST", "DELETE", None}


@pytest.mark.parametrize(("template", "method", "url"), template_requests())
def test_every_request_a_template_makes_goes_to_a_route_that_exists(template, method, url):
    routes = [route for route in app.routes if isinstance(route, APIRoute) and route.path_regex.fullmatch(url)]
    assert routes, f"{template} calls {url}, and no route matches"
    if method:
        assert any(method in route.methods for route in routes), f"{template} sends {method} to {url}"


def test_every_htmx_target_id_is_in_a_template():
    text = "".join(path.read_text() for path in TEMPLATES)
    ids = set(re.findall(r"""\bid=["']([^"'{]+)["']""", text))
    wanted = set(re.findall(r"""\bhx-(?:target|indicator)=["']#([^"']+)["']""", text))
    assert wanted and wanted <= ids


def test_the_files_that_must_stay_private_are_ignored_by_git():
    ignored = read(".gitignore").splitlines()
    assert ".env" in ignored
    assert "data/" in ignored


def test_no_test_folder_is_named_data():
    # .gitignore's data/ matches a folder of that name anywhere, so it would silently never be committed.
    assert [path for path in (ROOT / "tests").rglob("data") if path.is_dir()] == []


def test_no_gemini_key_is_in_the_repo():
    files = [ROOT / name for name in ("README.md", "CLAUDE.md", ".env.example", "requirements.txt", "requirements-dev.txt", "pytest.ini")]
    for folder in ("app", "tests", "tools", "docs"):
        files += [path for path in (ROOT / folder).rglob("*") if path.is_file() and path.suffix in {".py", ".md", ".html", ".json", ".xml", ".txt"}]
    assert [path.name for path in files if path.exists() and re.search(r"AIza[\w-]{30,}", path.read_text())] == []
    assert re.search(r"^GEMINI_API_KEY=\s*$", read(".env.example"), re.MULTILINE)
