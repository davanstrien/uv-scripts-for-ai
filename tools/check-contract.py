# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Static contract checker for the OCR recipes in `ocr/`.

Parses every `ocr/*.py` recipe with `ast` and reports violations of the invariants
documented in `ocr/CLAUDE.md` ("Conventions & invariants"). Each rule maps to a bug
we have actually hit. This is the machine-checkable half of the review-recipe skill;
the judgment-only conventions (model-card fidelity, context-length, pin rationale) stay
in that skill.

Usage:
    uv run tools/check-contract.py                 # all recipes
    uv run tools/check-contract.py ocr/glm-ocr.py  # one file
    uv run tools/check-contract.py --format github  # CI annotations

Exit code: 1 if any error-severity finding remains, else 0 (warnings do not fail).
"""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

# --- scoping -----------------------------------------------------------------

# Not user recipes: benchmark/eval coordinators and judges. They share code freely and
# do not follow the single-file recipe contract, so the rules below do not apply.
EXCLUDE = {
    "ocr-vllm-judge.py",  # offline vLLM jury judge (internal tooling)
    "ocr-bench-run.py",  # N-model benchmark coordinator (internal tooling)
    "ocr-human-eval.py",  # blind human A/B Gradio app (internal tooling)
    "ocr-jury-bench.py",  # API jury judge (internal tooling)
    "ocr-elo-bench.py",  # ELO benchmark eval (internal tooling)
}

# Per-file, per-rule exemptions for LEGITIMATE divergence only. Real gaps stay in the
# report. Each entry has a one-line reason for why the rule does not apply to that file.
EXEMPT = {
    # Bucket recipes: bucket file I/O (-v hf://…), not a dataset push. No input dataset
    # whose columns could collide (resume-by-skip on the output file instead), and no
    # --split/--image-column dataset surface — they read image files straight from a bucket.
    "falcon-ocr-bucket.py": ["ArgsPresent", "CollisionGuard"],
    "glm-ocr-bucket.py": ["ArgsPresent", "CollisionGuard"],
    "surya-ocr-bucket.py": ["ArgsPresent", "CollisionGuard"],
    # pp-* classical det+rec: inline sink-class collision guard instead of the helper
    # (see ocr/CLAUDE.md "Output-column collision guard" + the pp gotchas).
    "pp-ocrv6.py": ["CollisionGuard"],
    # Layout detection, not text OCR: writes a fixed `layout` column, so there is no
    # --output-column; inline buffer/finalize sink guard, not the helper.
    "pp-doclayout.py": ["CollisionGuard", "ArgsPresent"],
}

# --- rule config -------------------------------------------------------------

# argparse flags every recipe must expose (the ocr-bench launch contract + testing surface).
REQUIRED_ARGS = {
    "--config",
    "--create-pr",
    "--max-samples",
    "--seed",
    "--shuffle",
    "--split",
    "--output-column",
}
# An input column must be configurable; recipes name it by their input modality.
INPUT_COLUMN_ARGS = {
    "--image-column",
    "--text-column",
    "--source-column",
    "--content-column",
}

# inference_info entry schema (frozen — ocr/CLAUDE.md "inference_info schema").
REQUIRED_ENTRY_KEYS = {"model_id", "model_name", "column_name"}
# key actually written -> key it should have been (the two drifts that broke ocr-bench interop)
KEY_ALIASES = {"model": "model_id", "output_column": "column_name"}

CANONICAL_SENTINEL = "[OCR FAILED]"

SECTION = "ocr/CLAUDE.md » Conventions & invariants"
REFS = {
    "ArgsPresent": f"{SECTION} » New-recipe checklist / ocr-bench launch contract",
    "InferenceInfoIsList": f"{SECTION} » inference_info schema (frozen)",
    "InferenceInfoKeys": f"{SECTION} » inference_info schema (frozen)",
    "ErrorSentinel": f"{SECTION} » inference_info schema (one error sentinel)",
    "PushRetryWrapper": f"{SECTION} » GPU + output / push resilience",
    "VllmEnvGuards": f"{SECTION} » Env guards on the bare image",
    "CollisionGuard": f"{SECTION} » Output-column collision guard",
    "CudaCheck": f"{SECTION} » GPU + output",
}


@dataclass
class Finding:
    rule: str
    severity: str  # "error" | "warn"
    line: int
    message: str

    @property
    def ref(self) -> str:
        return REFS.get(self.rule, SECTION)


# --- ast helpers -------------------------------------------------------------


def set_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def ancestors(node: ast.AST):
    cur = getattr(node, "parent", None)
    while cur is not None:
        yield cur
        cur = getattr(cur, "parent", None)


def const_strings(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node


def dict_str_keys(node: ast.Dict) -> dict[str, ast.expr]:
    out = {}
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            out[key.value] = value
    return out


def parse_deps(source: str) -> set[str]:
    """Return normalized PEP 723 dependency names (e.g. {"vllm", "torch", "datasets"})."""
    lines = source.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "# /// script")
        end = next(
            i for i in range(start + 1, len(lines)) if lines[i].strip() == "# ///"
        )
    except StopIteration:
        return set()
    body = "\n".join(
        ln[2:] if ln.startswith("# ") else ln.lstrip("#")
        for ln in lines[start + 1 : end]
    )
    try:
        meta = tomllib.loads(body)
    except tomllib.TOMLDecodeError:
        return set()
    names = set()
    for spec in meta.get("dependencies", []):
        # strip extras and version specifiers: "falcon-perception[ocr]" -> "falcon-perception"
        name = spec.split(";")[0].split("[")[0]
        for sep in ("<", ">", "=", "!", "~", " "):
            name = name.split(sep)[0]
        if name:
            names.add(name.strip().lower())
    return names


def imports_torch(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                a.name == "torch" or a.name.startswith("torch.") for a in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "torch" or node.module.startswith("torch.")
            ):
                return True
    return False


def call_attr_name(call: ast.Call) -> str | None:
    return call.func.attr if isinstance(call.func, ast.Attribute) else None


# --- rules -------------------------------------------------------------------


def rule_args_present(tree: ast.AST, source: str, deps: set[str]) -> list[Finding]:
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and call_attr_name(node) == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                val = node.args[0].value
                if isinstance(val, str) and val.startswith("-"):
                    defined.add(val)
    findings = []
    for arg in sorted(REQUIRED_ARGS - defined):
        findings.append(
            Finding("ArgsPresent", "error", 0, f"missing argparse flag `{arg}`")
        )
    if not (defined & INPUT_COLUMN_ARGS):
        findings.append(
            Finding(
                "ArgsPresent",
                "error",
                0,
                "no input-column flag (need one of --image-column/--text-column/--source-column)",
            )
        )
    return findings


def _inference_entries(tree: ast.AST) -> list[ast.Dict]:
    """Dict literals that look like an inference_info entry: a model key + an output-column key."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = set(dict_str_keys(node))
        has_model = keys & {"model_id", "model", "model_name"}
        has_col = keys & {"column_name", "output_column"}
        if has_model and has_col:
            out.append(node)
    return out


def rule_inference_keys(tree: ast.AST, source: str, deps: set[str]) -> list[Finding]:
    findings = []
    for entry in _inference_entries(tree):
        keys = set(dict_str_keys(entry))
        line = entry.lineno
        # drifted keys: wrong name used *in place of* a required key (extras alongside are fine)
        for bad, good in KEY_ALIASES.items():
            if bad in keys and good not in keys:
                findings.append(
                    Finding(
                        "InferenceInfoKeys",
                        "error",
                        line,
                        f"inference_info entry key `{bad}` should be `{good}`",
                    )
                )
        # genuinely missing required keys (neither the key nor its accepted alias present)
        alias_of = {v: k for k, v in KEY_ALIASES.items()}
        for req in sorted(REQUIRED_ENTRY_KEYS):
            if req in keys:
                continue
            if req in alias_of and alias_of[req] in keys:
                continue  # reported as a drift above
            findings.append(
                Finding(
                    "InferenceInfoKeys",
                    "error",
                    line,
                    f"inference_info entry missing required key `{req}`",
                )
            )
    return findings


def rule_inference_is_list(tree: ast.AST, source: str, deps: set[str]) -> list[Finding]:
    """The entry must be appended into a JSON list, never dumped as a bare dict (olmocr2 bug)."""
    findings = []
    for entry in _inference_entries(tree):
        # entry assigned to a name?
        name = None
        parent = getattr(entry, "parent", None)
        if (
            isinstance(parent, ast.Assign)
            and len(parent.targets) == 1
            and isinstance(parent.targets[0], ast.Name)
        ):
            name = parent.targets[0].id
        # dict literal sitting directly inside a list literal -> fine
        if isinstance(parent, ast.List):
            continue
        if name is None:
            continue
        appended = False
        in_list = False
        bare_dumped = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and call_attr_name(node) == "append":
                if any(isinstance(a, ast.Name) and a.id == name for a in node.args):
                    appended = True
            if isinstance(node, ast.List):
                if any(isinstance(e, ast.Name) and e.id == name for e in node.elts):
                    in_list = True
            if isinstance(node, ast.Call) and _is_json_dumps(node):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == name
                ):
                    bare_dumped = True
        if not (appended or in_list) and bare_dumped:
            findings.append(
                Finding(
                    "InferenceInfoIsList",
                    "error",
                    entry.lineno,
                    "inference_info written as a bare dict; append into a JSON list instead",
                )
            )
    return findings


def _is_json_dumps(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "dumps"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "json"
    )


def rule_error_sentinel(tree: ast.AST, source: str, deps: set[str]) -> list[Finding]:
    findings = []
    seen = set()
    for node in const_strings(tree):
        val = node.value
        if len(val) < 3 or not (val.startswith("[") and val.endswith("]")):
            continue
        inner = val[1:-1]
        if not (inner.endswith("ERROR") or inner.endswith("FAILED")):
            continue
        if not inner.replace(" ", "").isalnum() or not inner[:1].isupper():
            continue
        if val == CANONICAL_SENTINEL or val in seen:
            continue
        seen.add(val)
        findings.append(
            Finding(
                "ErrorSentinel",
                "warn",
                node.lineno,
                f"error sentinel `{val}` — use `{CANONICAL_SENTINEL}`",
            )
        )
    return findings


def rule_push_retry(tree: ast.AST, source: str, deps: set[str]) -> list[Finding]:
    findings = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and call_attr_name(node) == "push_to_hub"):
            continue
        recv = node.func.value  # type: ignore[union-attr]
        # dataset pushes only: skip card pushes and super().push_to_hub() sink overrides
        recv_src = ast.unparse(recv).lower()
        if "card" in recv_src or isinstance(recv, ast.Call):
            continue
        anc = list(ancestors(node))
        has_try = any(isinstance(a, ast.Try) for a in anc)
        has_loop = any(isinstance(a, (ast.For, ast.While)) for a in anc)
        if not (has_try and has_loop):
            findings.append(
                Finding(
                    "PushRetryWrapper",
                    "warn",
                    node.lineno,
                    "push_to_hub not wrapped in a retry loop (bare push loses results on a transient upload error)",
                )
            )
    return findings


def rule_vllm_env_guards(tree: ast.AST, source: str, deps: set[str]) -> list[Finding]:
    if "vllm" not in deps:
        return []
    lines = source.splitlines()
    guard_line = next(
        (i + 1 for i, ln in enumerate(lines) if "VLLM_USE_FLASHINFER_SAMPLER" in ln),
        None,
    )
    import_line = next(
        (
            i + 1
            for i, ln in enumerate(lines)
            if ln.strip().startswith(("from vllm import", "import vllm"))
        ),
        None,
    )
    if import_line is None:
        return []
    if guard_line is None:
        return [
            Finding(
                "VllmEnvGuards",
                "error",
                import_line,
                "VLLM_USE_FLASHINFER_SAMPLER not set before importing vllm",
            )
        ]
    if guard_line > import_line:
        return [
            Finding(
                "VllmEnvGuards",
                "error",
                guard_line,
                f"VLLM_USE_FLASHINFER_SAMPLER set (line {guard_line}) after the vllm import (line {import_line})",
            )
        ]
    return []


def rule_collision_guard(tree: ast.AST, source: str, deps: set[str]) -> list[Finding]:
    defined = any(
        isinstance(n, ast.FunctionDef) and n.name == "ensure_output_columns_free"
        for n in ast.walk(tree)
    )
    called = any(
        isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "ensure_output_columns_free")
            or call_attr_name(n) == "ensure_output_columns_free"
        )
        for n in ast.walk(tree)
    )
    if defined and called:
        return []
    what = (
        "defined but never called"
        if defined
        else ("called but not defined" if called else "absent")
    )
    return [Finding("CollisionGuard", "error", 0, f"ensure_output_columns_free {what}")]


def rule_cuda_check(tree: ast.AST, source: str, deps: set[str]) -> list[Finding]:
    if "torch" not in deps and not imports_torch(tree):
        return []  # CPU recipe (e.g. tesseract) — no GPU guard expected
    # Accept both `torch.cuda.is_available()` and `from torch import cuda; cuda.is_available()`.
    has_check = any(
        isinstance(n, ast.Attribute)
        and n.attr == "is_available"
        and (
            (isinstance(n.value, ast.Attribute) and n.value.attr == "cuda")
            or (isinstance(n.value, ast.Name) and n.value.id == "cuda")
        )
        for n in ast.walk(tree)
    )
    if has_check:
        return []
    return [
        Finding(
            "CudaCheck",
            "error",
            0,
            "no torch.cuda.is_available() guard (recipe imports torch but never checks for a GPU)",
        )
    ]


RULES = [
    rule_args_present,
    rule_inference_is_list,
    rule_inference_keys,
    rule_error_sentinel,
    rule_push_retry,
    rule_vllm_env_guards,
    rule_collision_guard,
    rule_cuda_check,
]


# --- driver ------------------------------------------------------------------


def check_file(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    set_parents(tree)
    deps = parse_deps(source)
    exempt = set(EXEMPT.get(path.name, []))
    findings = []
    for rule in RULES:
        for f in rule(tree, source, deps):
            if f.rule in exempt:
                continue
            findings.append(f)
    findings.sort(key=lambda f: (f.severity != "error", f.line, f.rule))
    return findings


def discover(ocr_dir: Path) -> list[Path]:
    return sorted(p for p in ocr_dir.glob("*.py") if p.name not in EXCLUDE)


def print_human(results: dict[Path, list[Finding]]) -> None:
    for path, findings in results.items():
        rel = f"ocr/{path.name}"
        if not findings:
            print(f"\033[32m✓\033[0m {rel}")
            continue
        errs = sum(f.severity == "error" for f in findings)
        warns = len(findings) - errs
        tag = "\033[31m✗\033[0m" if errs else "\033[33m!\033[0m"
        print(
            f"{tag} {rel}  ({errs} error{'s' * (errs != 1)}, {warns} warning{'s' * (warns != 1)})"
        )
        for f in findings:
            colour = "\033[31m" if f.severity == "error" else "\033[33m"
            loc = f":{f.line}" if f.line else ""
            print(f"    {colour}{f.severity:<5}\033[0m {f.rule:<20} {rel}{loc}")
            print(f"          {f.message}")
            print(f"          → {f.ref}")


def print_github(results: dict[Path, list[Finding]]) -> None:
    for path, findings in results.items():
        for f in findings:
            level = "error" if f.severity == "error" else "warning"
            line = f.line or 1
            title = f"{f.rule} ({f.ref})"
            print(
                f"::{level} file=ocr/{path.name},line={line},title={title}::{f.message}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "files", nargs="*", help="specific ocr/*.py recipes (default: all)"
    )
    parser.add_argument("--format", choices=["human", "github"], default="human")
    args = parser.parse_args()

    ocr_dir = Path(__file__).resolve().parent.parent / "ocr"
    if args.files:
        paths = [Path(f).resolve() for f in args.files]
    else:
        paths = discover(ocr_dir)

    results: dict[Path, list[Finding]] = {}
    for path in paths:
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 2
        results[path] = check_file(path)

    if args.format == "github":
        print_github(results)
    else:
        print_human(results)
        total_err = sum(f.severity == "error" for fs in results.values() for f in fs)
        total_warn = sum(f.severity == "warn" for fs in results.values() for f in fs)
        clean = sum(not fs for fs in results.values())
        print()
        print(
            f"{len(results)} recipe(s): {clean} clean, {total_err} error(s), {total_warn} warning(s)"
        )

    has_error = any(f.severity == "error" for fs in results.values() for f in fs)
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
