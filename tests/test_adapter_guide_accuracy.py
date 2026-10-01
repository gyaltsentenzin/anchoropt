"""THE COLLABORATOR GUIDE MUST NOT DRIFT FROM THE CODE.

A porting guide that names a hook the contract does not require, or an import path that moved, costs a
collaborator a day and teaches them to distrust the docs. These tests pin the guide's EXECUTABLE claims
-- the hook tables, the import paths, the dataclass field names, the outcome-class strings -- against
the code they describe.

Prose is not checked. Structure is.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GUIDE = REPO / "docs" / "ADAPTER_GUIDE.md"
for p in (str(REPO), str(REPO / "examples" / "toy_host")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def text():
    return GUIDE.read_text()


def test_the_guide_exists_and_names_both_target_benchmarks(text):
    assert "TauBench" in text and "AppWorld" in text


def test_the_MANDATORY_hook_table_matches_the_contract_exactly(text):
    """A guide listing a hook core does not require sends a porter to write dead code."""
    from anchoropt.testing import MANDATORY_HOOKS

    code = {h for h, _ in MANDATORY_HOOKS}
    # The mandatory table is the block between the two headings; read hooks out of its first column.
    block = text.split("### Mandatory")[1].split("### Optional")[0]
    listed = set(re.findall(r"^\| `([A-Za-z_]+)", block, re.M))
    listed |= set(re.findall(r"^\| `([A-Za-z_]+)\(", block, re.M))
    listed |= {"HOST"} if "`HOST`" in block else set()
    assert listed == code, f"guide lists {sorted(listed)}, contract requires {sorted(code)}"


def test_every_OPTIONAL_hook_the_guide_names_is_really_optional(text):
    from anchoropt.testing import MANDATORY_HOOKS, OPTIONAL_HOOKS

    optional = {h for h, _ in OPTIONAL_HOOKS}
    mandatory = {h for h, _ in MANDATORY_HOOKS}
    block = text.split("### Optional")[1].split("## 2.")[0]
    for hook in set(re.findall(r"`([a-z_]+)\(", block)) | set(re.findall(r"`([a-z_]+)`", block)):
        if hook in mandatory:
            pytest.fail(f"the guide lists {hook!r} as optional but the contract requires it")
        if hook in optional:
            continue    # named and genuinely optional


def test_every_import_path_the_guide_shows_actually_imports(text):
    """The paths a collaborator will copy-paste."""
    import importlib

    for mod, names in re.findall(r"from ([\w.]+) import \(?\s*([^)\n]+)", text):
        if not mod.startswith("anchoropt"):
            continue
        m = importlib.import_module(mod)
        for name in [n.strip().rstrip(",") for n in names.split(",") if n.strip()]:
            if not name or name.startswith("#"):
                continue
            assert hasattr(m, name), f"{mod} has no {name!r}, but the guide imports it"


def test_the_ExecutorCapability_fields_the_guide_shows_all_exist(text):
    from anchoropt.learning.executor_capability import ExecutorCapability

    fields = set(ExecutorCapability.__dataclass_fields__)
    block = text.split("ExecutorCapability(")[1].split(")")[0]
    for kw in re.findall(r"^\s*([a-z_]+)=", block, re.M):
        assert kw in fields, f"the guide passes {kw!r} to ExecutorCapability, which has no such field"


def test_the_ThetaResult_fields_the_guide_shows_all_exist(text):
    from anchoropt.learning.policy_class import ThetaResult

    fields = set(ThetaResult.__dataclass_fields__)
    block = text.split("return ThetaResult(")[1].split("\n```")[0]
    for kw in re.findall(r"([a-z_]+)=", block):
        if kw in ("dict", "tuple", "c", "got", "baseline"):
            continue
        assert kw in fields, f"the guide passes {kw!r} to ThetaResult, which has no such field"


def test_the_four_outcome_class_names_match_the_code(text):
    from anchoropt.learning.external_evaluation import (
        BUDGET_EXHAUSTED, EVALUATED_NO_BENEFIT, IMPROVED, UNEVALUATED,
    )

    # Stop at the next HEADING, not at the first "---": a markdown table's own separator row is `|---|`
    # and splitting on "---" truncated the table to its header.
    tail = text.split("### The four outcome classes")[1]
    table = tail.split("\n## ")[0]
    for const in (IMPROVED, UNEVALUATED, EVALUATED_NO_BENEFIT, BUDGET_EXHAUSTED):
        assert f"`{const}`" in table, f"{const} is missing from the guide's outcome table"


def test_the_SUPPRESS_variant_table_matches_the_contract(text):
    """The A3/E1 split is the guide's most load-bearing structural claim."""
    from anchoropt.learning.action_contract import CONTRACTS, Operator

    c = CONTRACTS[Operator.SUPPRESS]
    assert "preservation" not in c.required_for("cancel_proposed"), \
        "remove-outright must NOT require preservation; the guide says so"
    assert "preservation" in c.required_for("withhold_and_replay"), \
        "withhold-and-replay MUST require preservation; the guide says so"
    assert "preservation" in c.enforced, "and it must be executor-backed; the guide says so"
    assert "variant_required" in text, "the guide must name the mechanism, not just the behaviour"


def test_the_action_families_the_guide_calls_closed_are_closed(text):
    from anchoropt.anchor import Action

    assert {a.value for a in Action} == {"noop", "reprompt", "suppress", "reroute"}
    assert "Do not add a fifth" in text


def test_every_repo_path_the_guide_links_exists(text):
    """A broken path in a porting guide is a collaborator's first impression."""
    for link in re.findall(r"\]\(((?:\.\./|examples/|tests/|docs/|anchoropt/)[^)#]+)\)", text):
        target = (GUIDE.parent / link).resolve()
        assert target.exists(), f"the guide links {link!r}, which does not exist"


def test_the_commands_in_the_checklist_reference_real_files(text):
    block = text.split("## 8.")[1]
    for path in re.findall(r"(tests/[\w./]+\.py|examples/[\w./]+\.py)", block):
        assert (REPO / path).exists(), f"the checklist runs {path!r}, which does not exist"


def test_the_reference_implementation_passes_its_own_contract():
    """The guide calls examples/toy_host the reference implementation. It must satisfy the contract."""
    from anchoropt.testing import check_adapter_contract

    import toy_adapter
    rep = check_adapter_contract(toy_adapter.ADAPTER)
    assert rep.ok, rep.summary()
