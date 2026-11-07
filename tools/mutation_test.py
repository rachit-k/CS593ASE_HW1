"""Mutation testing: make one small change to the code at a time and see whether any test notices.

A change (a "mutant") is killed if a test fails, and survives if the whole suite still passes. A
survivor is behavior no test checks, or a change that makes no difference (an "equivalent" mutant).
Coverage only says a line ran; this says whether anything checks what the line does.

The mutants are generated mechanically, so they aren't chosen with the tests in mind:

- Python: flip comparisons and and/or, negate conditions, change numbers and arithmetic, empty
  strings, return None, delete calls and raises, skip loops, stop catching exceptions, drop keyword
  arguments, and drop the right side of and/or.
- Templates and JavaScript: negate {% if %}, delete {{ output }}, remove filters and behavior-bearing
  attributes, and delete JavaScript statements.

Everything runs in scratch copies. The folder being tested is only read.

Usage, from the repo root:
    python tools/mutation_test.py                      # everything; takes about half an hour
    python tools/mutation_test.py --only app/db.py     # one file
    python tools/mutation_test.py --rerun report.json  # only what survived last time (tests changed, code didn't)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PYTHON_FILES = ["app/arxiv_client.py", "app/citations.py", "app/config.py", "app/db.py", "app/llm.py", "app/main.py", "app/pdfs.py"]
COPIED = ["app", "tests", "docs", "pytest.ini", ".env.example", ".gitignore", "README.md", "CLAUDE.md", "requirements.txt", "requirements-dev.txt"]
# Those tests read source code, so they would notice any change at all, even one in formatting (mutants
# are written with ast.unparse). Only a test that checks behavior says something by killing a mutant.
CHECKS_BEHAVIOR = ["--ignore=tests/test_contracts.py"]
COMPARISONS = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}
ARITHMETIC = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.Pow: ast.Mult}


class Mutator(ast.NodeTransformer):
    """Walks a module, numbering every place it could change, and changes only the target one."""

    def __init__(self, target: int = -1):
        self.target = target
        self.sites: list[tuple[int, str]] = []

    def _site(self, node: ast.AST, description: str) -> bool:
        self.sites.append((getattr(node, "lineno", 0), description))
        return len(self.sites) - 1 == self.target

    # --- Things that aren't behavior: docstrings, annotations, decorators, imports -----------

    def _visit_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        start = 1 if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str) else 0
        new: list[ast.stmt] = body[:start]
        for statement in body[start:]:
            result = self.visit(statement)
            new.extend(result if isinstance(result, list) else [result])
        return new

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = self._visit_body(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.body = self._visit_body(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.args.defaults = [self.visit(default) for default in node.args.defaults]
        node.args.kw_defaults = [self.visit(default) if default is not None else None for default in node.args.kw_defaults]
        node.body = self._visit_body(node.body)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    def visit_Import(self, node: ast.Import) -> ast.Import:
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        return node

    def visit_FormattedValue(self, node: ast.FormattedValue) -> ast.FormattedValue:
        node.value = self.visit(node.value)
        return node

    # --- Operators ---------------------------------------------------------------------------

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        for i, op in enumerate(node.ops):
            new = COMPARISONS.get(type(op))
            if new and self._site(node, f"{type(op).__name__} -> {new.__name__}"):
                node.ops[i] = new()
        return self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        if self._site(node, "and <-> or"):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        if self._site(node, "drop the last operand of and/or"):
            node.values = node.values[:-1]
            if len(node.values) == 1:
                return self.visit(node.values[0])
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        if isinstance(node.op, (ast.Not, ast.USub)) and self._site(node, f"drop {'not' if isinstance(node.op, ast.Not) else 'minus'}"):
            return self.visit(node.operand)
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        new = ARITHMETIC.get(type(node.op))
        if new and self._site(node, f"{type(node.op).__name__} -> {new.__name__}"):
            node.op = new()
        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        new = ARITHMETIC.get(type(node.op))
        if new and self._site(node, f"augmented {type(node.op).__name__} -> {new.__name__}"):
            node.op = new()
        return self.generic_visit(node)

    def _negate(self, node: ast.If | ast.IfExp | ast.While) -> ast.AST:
        if self._site(node, f"negate the {type(node).__name__} condition"):
            node.test = ast.UnaryOp(ast.Not(), node.test)
        return self.generic_visit(node)

    visit_If = visit_IfExp = visit_While = _negate

    def visit_For(self, node: ast.For) -> ast.stmt:
        if self._site(node, "skip the loop"):
            node.iter = ast.List([], ast.Load())
        return self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> ast.comprehension:
        for i, test in enumerate(node.ifs):
            if self._site(test, "negate the comprehension filter"):
                node.ifs[i] = ast.UnaryOp(ast.Not(), test)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.expr:
        value = node.value
        if isinstance(value, bool):
            if self._site(node, f"{value} -> {not value}"):
                return ast.Constant(not value)
        elif isinstance(value, int):
            if self._site(node, f"{value} -> {value + 1}"):
                return ast.Constant(value + 1)
        elif isinstance(value, float):
            if self._site(node, f"{value} -> {value + 1}"):
                return ast.Constant(value + 1)
        elif isinstance(value, str):
            if self._site(node, f"string {value[:40]!r} -> {'empty' if value else 'XX'}"):
                return ast.Constant("" if value else "XX")
        elif isinstance(value, bytes) and value:  # empty bytes can't get any emptier
            if self._site(node, f"bytes {value[:20]!r} -> empty"):
                return ast.Constant(b"")
        return node

    def visit_Return(self, node: ast.Return) -> ast.stmt:
        is_none = node.value is None or (isinstance(node.value, ast.Constant) and node.value.value is None)
        if not is_none and self._site(node, "return None"):
            node.value = ast.Constant(None)
            return node
        return self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> ast.stmt:
        if isinstance(node.value, ast.Call) and self._site(node, f"delete the call {ast.unparse(node.value.func)}(...)"):
            return ast.Pass()
        return self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> ast.stmt:
        if self._site(node, "delete the raise"):
            return ast.Pass()
        return self.generic_visit(node)

    def visit_Continue(self, node: ast.Continue) -> ast.stmt:
        return ast.Pass() if self._site(node, "delete continue") else node

    def visit_Break(self, node: ast.Break) -> ast.stmt:
        return ast.Pass() if self._site(node, "delete break") else node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler:
        if node.type is not None and self._site(node, f"stop catching {ast.unparse(node.type)}"):
            node.type = ast.Tuple([], ast.Load())
            node.body = self._visit_body(node.body)
            return node
        node.body = self._visit_body(node.body)
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        for i, keyword in enumerate(list(node.keywords)):
            if keyword.arg and self._site(node, f"drop the keyword {keyword.arg}="):
                node.keywords = [k for j, k in enumerate(node.keywords) if j != i]
                break
        return self.generic_visit(node)


def python_mutants(path: Path) -> list[dict]:
    mutator = Mutator()
    mutator.visit(ast.parse(path.read_text()))
    return [{"line": line, "change": change, "index": i} for i, (line, change) in enumerate(mutator.sites)]


def apply_python_mutant(source: str, index: int) -> str:
    return ast.unparse(Mutator(index).visit(ast.parse(source))) + "\n"


# Behavior-bearing attributes: what htmx does, what forms submit, what scripts and CSS hook onto.
_ATTRIBUTE_RE = re.compile(r"""\s(hx-[\w:-]+|name|type|href|id|required|selected|accept|action|method|enctype|target|rel|aria-[\w-]+|onclick|maxlength)(=(?:"[^"]*"|'[^']*'))?(?=[\s>/])""")
_IF_RE = re.compile(r"\{%-?\s*(if|elif)\s+(.+?)\s*-?%\}")
_OUTPUT_RE = re.compile(r"\{\{-?\s*(.+?)\s*-?\}\}")
_FILTER_RE = re.compile(r"\|\s*(markdown|author_list|tojson|join\([^)]*\)|length)")
_JS_STATEMENT_RE = re.compile(r"^\s+(?:const |button\.|form\.|for \(|document\.addEventListener).*$", re.M)
# A few style rules that behavior depends on. Other CSS is only how things look.
_CSS_RULES = ["\n.abstract.clamped {", "\n.htmx-indicator {", "\n.empty-note {", "\nol:not(:has(> li)) + .empty-note {", "\n.busy-label,", "\n@media (max-width: 40rem) {", "\n@media (prefers-color-scheme: dark) {"]


def text_mutants(path: Path, name: str) -> list[dict]:
    source = path.read_text()
    found: list[dict] = []

    def add(start: int, end: int, replacement: str, change: str) -> None:
        found.append({"line": source.count("\n", 0, start) + 1, "change": change, "start": start, "end": end, "replacement": replacement, "index": len(found)})

    if name.endswith(".css"):
        for rule in _CSS_RULES:
            at = source.find(rule)
            if at >= 0:
                add(at, at + len(rule), rule.replace(rule.strip(), "#disabled-rule-" + rule.strip()) if not rule.strip().startswith("@") else "\n@media (max-width: 0px) {", f"disable CSS rule {rule.strip()[:40]}")
        return found
    for match in _IF_RE.finditer(source):
        add(match.start(), match.end(), f"{{% {match.group(1)} not ({match.group(2)}) %}}", f"negate {{% {match.group(1)} {match.group(2)[:40]} %}}")
    for match in _OUTPUT_RE.finditer(source):
        add(match.start(), match.end(), "", f"delete output {{{{ {match.group(1)[:40]} }}}}")
    for match in _FILTER_RE.finditer(source):
        add(match.start(), match.end(), "", f"remove the filter |{match.group(1)}")
    # Only inside a tag: the word "type" in a sentence isn't an attribute. Jinja can hold a ">", so it's blanked out first.
    plain = re.sub(r"\{[%{].*?[%}]\}", lambda found: "_" * len(found.group()), source, flags=re.S)
    for match in _ATTRIBUTE_RE.finditer(source):
        if plain.rfind("<", 0, match.start()) > plain.rfind(">", 0, match.start()):
            add(match.start(), match.end(), "", f"remove the attribute {match.group(0).strip()[:50]}")
    if name.endswith("base.html"):
        for match in _JS_STATEMENT_RE.finditer(source):
            text = match.group(0).strip()
            if text.endswith(";"):
                add(match.start(), match.end(), "", f"delete the JavaScript {text[:50]}")
    return found


def apply_text_mutant(source: str, mutant: dict) -> str:
    return source[: mutant["start"]] + mutant["replacement"] + source[mutant["end"] :]


def discover(root: Path, only: list[str] | None) -> list[dict]:
    mutants = []
    for file in PYTHON_FILES:
        if not only or file in only:
            for mutant in python_mutants(root / file):
                mutants.append({"file": file, **mutant})
    text_files = sorted(str(p.relative_to(root)) for p in (root / "app/templates").rglob("*.html")) + ["app/static/style.css"]
    for file in text_files:
        if not only or file in only:
            for mutant in text_mutants(root / file, file):
                mutants.append({"file": file, **mutant})
    return mutants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."), help="the folder holding app/ and tests/")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--only", nargs="*", help="only these files, for example app/db.py")
    parser.add_argument("--rerun", type=Path, help="a previous JSON report: re-test only its survivors")
    parser.add_argument("--report", type=Path, default=Path("mutation-report.json"))
    parser.add_argument("--no-text", action="store_true", help="skip templates, JavaScript, and CSS")
    args = parser.parse_args()

    root = args.root.resolve()
    has_browser_tests = (root / "tests/test_browser.py").exists()
    work = Path(tempfile.mkdtemp(prefix="mutation-"))
    plugin_dir = work / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "skip_listed.py").write_text(
        "import os\n"
        "def pytest_collection_modifyitems(config, items):\n"
        "    skip = set(open(os.environ['SKIP_TESTS'], encoding='utf-8').read().splitlines())\n"
        "    config.hook.pytest_deselected(items=[i for i in items if i.nodeid in skip])\n"
        "    items[:] = [i for i in items if i.nodeid not in skip]\n"
    )
    skip_file = work / "skip.txt"
    skip_file.write_text("")
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(plugin_dir),
        "SKIP_TESTS": str(skip_file),
        "HYPOTHESIS_PROFILE": "mutation",
    }

    def copy_to(destination: Path) -> None:
        destination.mkdir()
        for item in COPIED:
            source = root / item
            if source.is_dir():
                shutil.copytree(source, destination / item, ignore=shutil.ignore_patterns("__pycache__", ".hypothesis"))
            elif source.exists():
                shutil.copy(source, destination / item)

    def run_tests(folder: Path, *options: str) -> int:
        command = [sys.executable, "-m", "pytest", "-x", "-q", "--tb=no", "-p", "no:cacheprovider", "-W", "ignore", "-p", "skip_listed", *CHECKS_BEHAVIOR, *options]
        try:
            return subprocess.run(command, cwd=folder, env=env, capture_output=True, timeout=600).returncode
        except subprocess.TimeoutExpired:
            return 124

    # Tests that already fail on the unmodified code can't catch anything, so they don't count.
    baseline_dir = work / "baseline"
    copy_to(baseline_dir)
    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--tb=no", "-p", "no:cacheprovider", "-W", "ignore", *CHECKS_BEHAVIOR],
        cwd=baseline_dir, env=env, capture_output=True, text=True,
    )
    already_failing = re.findall(r"^(tests/\S.*?) (?:FAILED|ERROR)", baseline.stdout, re.M)
    skip_file.write_text("\n".join(already_failing))
    print(f"unmodified code: {baseline.stdout.strip().splitlines()[-1].strip('= ')}", flush=True)
    if already_failing:
        print(f"ignoring {len(already_failing)} tests that already fail", flush=True)

    mutants = discover(root, args.only)
    if args.no_text:
        mutants = [m for m in mutants if m["file"].endswith(".py")]
    previous: list[dict] = []
    if args.rerun:
        previous = json.loads(args.rerun.read_text())["mutants"]
        survivors = {(m["file"], m["line"], m["change"], m["index"]) for m in previous if m["status"] == "survived"}
        previous = [m for m in previous if m["status"] != "survived"]
        mutants = [m for m in mutants if (m["file"], m["line"], m["change"], m["index"]) in survivors]
    print(f"{len(mutants)} mutants, {args.workers} workers", flush=True)

    folders: list[Path] = []
    for i in range(args.workers):
        folder = work / f"worker-{i}"
        copy_to(folder)
        folders.append(folder)
    pool = list(folders)
    pool_lock = threading.Lock()
    done = [0]

    def test_one(mutant: dict) -> dict:
        with pool_lock:
            folder = pool.pop()
        path = folder / mutant["file"]
        original = path.read_text()
        try:
            changed = apply_python_mutant(original, mutant["index"]) if mutant["file"].endswith(".py") else apply_text_mutant(original, mutant)
            path.write_text(changed)
            code = run_tests(folder, "-m", "not browser")
            if code == 0 and has_browser_tests:
                code = run_tests(folder, "-m", "browser")
                if code == 5:  # no tests collected
                    code = 0
        except Exception as exc:  # a mutant the tool can't apply says nothing about the tests
            code = -1
            mutant["error"] = str(exc)[:200]
        finally:
            path.write_text(original)
            with pool_lock:
                pool.append(folder)
        mutant["status"] = {0: "survived", -1: "not applied"}.get(code, "killed")
        with pool_lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]} of {len(mutants)}", flush=True)
        return mutant

    with ThreadPoolExecutor(args.workers) as executor:
        results = list(executor.map(test_one, mutants))
    shutil.rmtree(work, ignore_errors=True)

    results = previous + results
    results.sort(key=lambda m: (m["file"], m["line"], m["index"]))
    applied = [m for m in results if m["status"] != "not applied"]
    killed = sum(m["status"] == "killed" for m in applied)
    args.report.write_text(json.dumps({"killed": killed, "total": len(applied), "mutants": results}, indent=1))

    print(f"\n{'file':42} {'mutants':>8} {'killed':>7} {'survived':>9} {'score':>7}")
    for file in sorted({m["file"] for m in applied}):
        group = [m for m in applied if m["file"] == file]
        group_killed = sum(m["status"] == "killed" for m in group)
        print(f"{file:42} {len(group):>8} {group_killed:>7} {len(group) - group_killed:>9} {100 * group_killed / len(group):>6.1f}%")
    print(f"{'TOTAL':42} {len(applied):>8} {killed:>7} {len(applied) - killed:>9} {100 * killed / len(applied):>6.1f}%")
    not_applied = len(results) - len(applied)
    if not_applied:
        print(f"{not_applied} mutants could not be applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
