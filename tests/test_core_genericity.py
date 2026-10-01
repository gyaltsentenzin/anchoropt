"""Core must not name a benchmark. One guard over the whole package, checked by IDENTIFIER.

WHY THIS FILE EXISTS. The rule was real but unenforced in aggregate: three separate tests each
hard-coded one file path with its own word list and its own matching strategy, so
`signal_grammar.py`, `candidate_search.py`, `realization.py`, `anchor_policy_opt.py` and
`residual_problem.py` were covered by nothing -- and `prestate_eligibility.py` had drifted, encoding
one benchmark's terminal status string and its trace field names directly in core.

WHY IDENTIFIERS AND LITERALS, NOT A VOCABULARY GREP. The older guards banned words like "read",
"write", "memory", "container", "similarity" and even "threshold" as bare substrings of the whole
file, comments included. That is too blunt in both directions: it fails on a docstring that says "read
the trajectory" while passing anything that spells a benchmark concept in a synonym. What actually
breaks portability is core hard-coding a SPECIFIC runtime's identifiers -- its trace status strings,
field names, tool names, cell names, env switches -- so those are what is forbidden, and they are
matched as identifiers/literals rather than as substrings of prose.

The portability claim is tested positively as well: `tests/test_structured_search.py` drives the
entire WHERE->WHAT->HOW schedule against a toy runtime with its own domain vocabulary. A core that
needed a benchmark concept could not pass that file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parent.parent / "anchoropt"

# Identifiers and literals belonging to ONE runtime. Every entry is a concrete spelling that only
# makes sense for the BFCL v4 memory benchmark -- not a general concept that core may legitimately
# discuss. Matched with word boundaries, so prose is unaffected.
BENCHMARK_IDENTIFIERS = (
    # trace status / event vocabulary
    "answer_end_turn",
    # trace field names
    "decoded", "tool_results", "error_signal", "cell_key", "gates_fired", "test_entry_id",
    # store / tool names
    "core_memory", "archival_memory", "core_memory_retrieve", "archival_memory_search",
    "memory_retrieve", "list_keys", "retrieve_all",
    # backend / cell names
    "rec_sum", "memory_kv", "memory_vector", "notetaker",
    # benchmark and gate identifiers
    "bfcl", "bfcl_eval", "on_low_similarity_cross_container", "on_domain_error_core_full",
    "enable_zero_call_reprompt", "enable_archival_evict_duplicate", "memory_gates",
    # env switches
    "ANCHOROPT_XCM", "ANCHOROPT_ELEN", "ANCHOROPT_PHI_THETA", "ANCHOROPT_SNAPSHOT_CACHE",
)

# ------------------------------------------------------------------------------------------------
# TWO TIERS, deliberately.
#
# STRICT: the modules carrying the WHERE->WHAT->HOW algorithm. These must name no benchmark
# identifier at all, with no exemption, because they are what a second benchmark would reuse
# unchanged.
#
# RATCHET: the rest of core carries PRE-EXISTING debt -- 24 files, catalogued in
# docs/core_genericity_debt.json with the exact identifiers each one names. That debt is real and
# should shrink, but paying it off is a separate piece of work from building the optimizer, and doing
# it here would mean editing attribution, telemetry and six mechanism modules in the same change.
# So the baseline is FROZEN instead: existing violations are tolerated by file, and any NEW one --
# or any new identifier in an already-listed file -- fails. The list may only get shorter.
# ------------------------------------------------------------------------------------------------

STRICT = ("learning/structured_search.py", "learning/search_state.py",
          "learning/learned_signal.py", "learning/boundary_search.py",
          "learning/controller.py", "learning/signal_expansion.py")

DEBT_BASELINE = Path(__file__).resolve().parent.parent / "docs" / "core_genericity_debt.json"


def _baseline() -> dict[str, list[str]]:
    with open(DEBT_BASELINE) as fh:
        return json.load(fh)


def _hits(src: str) -> list[str]:
    return sorted({tok for tok in BENCHMARK_IDENTIFIERS
                   if re.search(rf"(?<![A-Za-z0-9_]){re.escape(tok)}(?![A-Za-z0-9_])", src)})


def _core_files() -> list[Path]:
    return sorted(p for p in CORE.rglob("*.py") if "__pycache__" not in str(p))


def test_there_are_core_files_to_check():
    """A guard that silently matches nothing is worse than no guard."""
    assert len(_core_files()) > 20


@pytest.mark.parametrize("rel", STRICT)
def test_the_algorithm_modules_name_no_benchmark_identifier(rel):
    """No exemption: these are exactly what a second benchmark reuses unchanged."""
    hits = _hits((CORE / rel).read_text())
    assert not hits, (f"{rel} names benchmark identifier(s) {hits}. Core must ask the runtime "
                      f"adapter -- see structured_search.optimize_residual for the contract.")


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: str(p.relative_to(CORE)))
def test_no_core_module_acquires_a_NEW_benchmark_identifier(path):
    """The ratchet. Pre-existing debt is frozen per file; anything new fails."""
    rel = str(path.relative_to(CORE))
    allowed = set(_baseline().get(rel, ()))
    new = [h for h in _hits(path.read_text()) if h not in allowed]
    assert not new, (f"{rel} names NEW benchmark identifier(s) {new}. Ask the runtime adapter "
                     f"instead; do not add to docs/core_genericity_debt.json.")


def test_the_recorded_debt_is_accurate_and_only_shrinks():
    """A baseline that drifts from reality stops being a ratchet.

    An entry that is no longer violated must be REMOVED from the file -- that is how paying the debt
    down gets locked in rather than silently re-openable.
    """
    stale = {}
    for rel, toks in _baseline().items():
        path = CORE / rel
        if not path.exists():
            stale[rel] = "file gone"
            continue
        actual = set(_hits(path.read_text()))
        fixed = sorted(set(toks) - actual)
        if fixed:
            stale[rel] = f"no longer names {fixed}"
    assert not stale, (f"docs/core_genericity_debt.json is stale -- remove the fixed entries so the "
                       f"ratchet tightens: {stale}")


def test_the_new_search_modules_are_clean_without_any_allowance():
    """The three modules carrying the algorithm must need no exemption at all."""
    for rel in ("learning/structured_search.py", "learning/search_state.py",
                "learning/learned_signal.py"):
        assert rel in STRICT, f"{rel} carries the algorithm and must be in the STRICT tier"
        src = (CORE / rel).read_text()
        for tok in BENCHMARK_IDENTIFIERS:
            assert not re.search(rf"(?<![A-Za-z0-9_]){re.escape(tok)}(?![A-Za-z0-9_])", src), \
                f"{rel} names {tok!r}"


def test_core_declares_no_semantic_stage_ordering():
    """The deleted hand-engineering: a declared stage order is an answer, not an input.

    Boundaries come from the trajectory. A core module that ships an ordering of named stages has
    reintroduced the map whose removal is the reason boundary derivation exists.
    """
    for path in _core_files():
        src = path.read_text()
        assert "STAGE_ORDER" not in src, f"{path.relative_to(CORE)} declares a stage order"


def test_structured_search_reaches_the_runtime_only_through_named_hooks():
    """Every adapter call in the orchestrator is a documented contract method.

    If a new `runtime.<something>` appears here, it is a new thing every adapter must implement, so
    it belongs in the module docstring's contract list -- this keeps that list honest.
    """
    src = (CORE / "learning/structured_search.py").read_text()
    called = set(re.findall(r"runtime\.([a-z_][a-z0-9_]*)", src))
    documented = {"declared_signals", "synthesis_fields", "install_signal", "signal_boundaries",
                  "boundary_from_key", "is_decision", "boundary_key", "evaluate_signal",
                  "probe_params"}
    assert called <= documented, (f"undocumented runtime hook(s) {sorted(called - documented)} -- "
                                 f"add them to the contract in the module docstring")
