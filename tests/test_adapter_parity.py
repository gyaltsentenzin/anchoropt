"""The benchmark adapter must be a MOVE, not a rewrite.

Five copies of the case-ID grammar were replaced by one adapter, and three inline tool-name tests in
`trace_backward` were replaced by adapter predicates. The whole value of that change depends on the
behaviour being identical, so these tests re-derive the OLD implementations inline and assert the
adapter agrees with them case for case.

The old implementations are duplicated here on purpose. A parity test that imports the new code twice
proves nothing; this one keeps the thing it is comparing against.

One deliberate difference, asserted rather than tolerated: an unrecognised case id used to be filed
under `rec_sum` by two of the five copies. It now raises.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "bfcl_v4"))

from adapter import (
    UnknownCaseId,
    backend_of,
    cell_of,
    chain_of,
    container_of,
    is_clear,
    is_list_keys,
    is_prereq,
    is_read_all,
    parse_case_id,
    phase_of,
    scenario_of,
    target_id,
)

# ------------------------------------------------------------------------------------------------
# The implementations being replaced, verbatim
# ------------------------------------------------------------------------------------------------

_OLD_CHAIN_RE = re.compile(r"^(memory_(?:kv|vector|rec_sum))_(?:prereq_)?(\d+)-([a-z_]+)")
_OLD_BACKEND_RE = re.compile(r"^memory_(kv|vector|rec_sum)_")
_OLD_VECID_RE = re.compile(r"vec_id\s*=\s*(\d+)")


def _old_chain_of(case_id: str) -> tuple[str, int]:
    """`trace_backward._chain_of`, as it was."""
    m = _OLD_CHAIN_RE.match(case_id)
    if not m:
        return ("unknown", 0)
    backend, num, scenario = m.groups()
    return (f"{backend}-{scenario}", int(num))


def _old_backend_of_strict(case_id: str):
    """`check_attribution.backend_of`, as it was -- None on unknown."""
    m = _OLD_BACKEND_RE.match(case_id or "")
    return m.group(1) if m else None


def _old_backend_of_loose(case_id: str) -> str:
    """`run.py` / `derive_per_backend.py`, as they were -- the silent rec_sum fallback."""
    if "memory_kv" in case_id:
        return "kv"
    return "vector" if "vector" in case_id else "rec_sum"


def _corpus_ids() -> list[str]:
    data = REPO / "benchmarks" / "bfcl_v4" / "data"
    ids: set[str] = set()
    for name in ("g8_balanced_train_cases.json", "g8_balanced_test_cases.json"):
        for case in json.loads((data / name).read_text()):
            ids.add(case["id"])
    return sorted(ids)


# ------------------------------------------------------------------------------------------------
# Case-ID grammar parity, over the WHOLE corpus
# ------------------------------------------------------------------------------------------------

def test_the_corpus_is_big_enough_to_be_worth_checking():
    ids = _corpus_ids()
    assert len(ids) >= 490, f"expected ~498 ids across both splits, got {len(ids)}"


def test_backend_matches_the_strict_old_implementation_on_every_id():
    for case_id in _corpus_ids():
        assert backend_of(case_id) == _old_backend_of_strict(case_id), case_id


def test_backend_matches_the_LOOSE_old_implementations_on_every_REAL_id():
    """The loose copies were only wrong on ids they could not parse. On the corpus they agree."""
    for case_id in _corpus_ids():
        assert backend_of(case_id) == _old_backend_of_loose(case_id), case_id


def test_chain_of_matches_the_old_implementation_on_every_id():
    for case_id in _corpus_ids():
        assert list(chain_of(case_id)) == list(_old_chain_of(case_id)), case_id


def test_chain_of_keeps_the_tolerant_contract_for_stray_files():
    """`trace_backward` sorts trajectory files that may include non-corpus artifacts, so this one
    returns ("unknown", 0) rather than raising. That was its contract and it is preserved."""
    assert chain_of("not_a_case_id") == ("unknown", 0) == _old_chain_of("not_a_case_id")


def test_the_derived_fields_are_consistent_with_the_raw_id():
    """scenario/phase/prereq/cell are new conveniences; they must not disagree with the id text."""
    for case_id in _corpus_ids():
        parsed = parse_case_id(case_id)
        assert parsed.backend in ("kv", "vector", "rec_sum")
        assert f"memory_{parsed.backend}_" in case_id
        assert f"-{parsed.scenario}-" in case_id or case_id.endswith(f"-{parsed.scenario}")
        assert is_prereq(case_id) == ("_prereq_" in case_id)
        assert phase_of(case_id) == ("prereq" if "_prereq_" in case_id else "query")
        assert cell_of(case_id) == (parsed.backend, parsed.scenario)
        assert scenario_of(case_id) == parsed.scenario


# ------------------------------------------------------------------------------------------------
# THE deliberate behaviour change
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["weird_id_no_backend", "", "memory_graph_1-x-1", "kv_1-x-1"])
def test_an_unrecognised_id_raises_instead_of_being_filed_under_rec_sum(bad):
    """The one intended difference. Two of the five old copies returned `rec_sum` for anything they
    could not parse -- a chain of `if/elif` with no failure branch -- so an unknown id was filed into
    a real backend. That is worse than a loud failure, and it is now a loud failure."""
    assert _old_backend_of_loose(bad) == "rec_sum", "the old behaviour, for the record"
    with pytest.raises(UnknownCaseId):
        backend_of(bad)


def test_the_five_old_copies_really_did_disagree():
    """Documents WHY this adapter exists, so nobody re-inlines a local copy later."""
    bad = "weird_id_no_backend"
    assert _old_backend_of_strict(bad) is None
    assert _old_backend_of_loose(bad) == "rec_sum"


# ------------------------------------------------------------------------------------------------
# Tool-vocabulary parity
# ------------------------------------------------------------------------------------------------

CALLS = [
    "core_memory_list_keys()",
    "core_memory_retrieve_all()",
    "archival_memory_list_keys()",
    "archival_memory_retrieve_all()",
    'core_memory_add(key="k", value="v")',
    'archival_memory_add(text="t")',
    'core_memory_update(vec_id=42, new_text="x")',
    "archival_memory_remove(vec_id=7)",
    "core_memory_clear()",
    "archival_memory_clear()",
    'memory_append(content="c")',
    'core_memory_retrieve(query="q")',
    'archival_memory_key_search(query="q")',
    "nonsense()",
    "",
    "core_memory_remove(key='a')",
]


@pytest.mark.parametrize("call", CALLS, ids=lambda c: (c[:34] or "empty"))
def test_read_all_predicate_matches_the_old_inline_test(call):
    """`trace_backward` asked this as two substring tests OR'd together."""
    old = "core_memory_list_keys" in call or "core_memory_retrieve_all" in call
    assert is_read_all(call, "core") is old


@pytest.mark.parametrize("call", CALLS, ids=lambda c: (c[:34] or "empty"))
def test_list_keys_predicate_matches_the_old_inline_test(call):
    old = "archival_memory_list_keys" in call
    assert is_list_keys(call, "archival") is old


@pytest.mark.parametrize("call", CALLS, ids=lambda c: (c[:34] or "empty"))
def test_target_id_matches_the_old_vecid_regex(call):
    match = _OLD_VECID_RE.search(call)
    assert target_id(call) == (match.group(1) if match else None)


def test_container_of_returns_none_for_the_blob_backend():
    """rec_sum's verbs name no container. None is a FACT about that backend, not a parse failure --
    treating it as one is why it logged zero admission events for a whole line of work."""
    assert container_of('memory_append(content="c")') is None
    assert container_of("core_memory_clear()") == "core"
    assert container_of("archival_memory_add(text='t')") == "archival"


def test_clear_and_remove_are_distinguished():
    """A remove forced by a capacity limit is legitimate recovery; a clear destroys state
    gratuitously. Gating them together breaks the repair A1 depends on."""
    from adapter import is_destructive, is_remove

    assert is_clear("archival_memory_clear()") and not is_remove("archival_memory_clear()")
    assert is_remove("archival_memory_remove(vec_id=7)")
    assert not is_clear("archival_memory_remove(vec_id=7)")
    assert is_destructive("core_memory_clear()") and is_destructive("core_memory_remove(key='a')")


# ------------------------------------------------------------------------------------------------
# The core must not import the benchmark at module scope
# ------------------------------------------------------------------------------------------------

def test_no_module_under_anchoropt_imports_benchmarks_at_module_scope():
    """A top-level `import benchmarks...` would make the core unloadable without one."""
    offenders = []
    for path in sorted((REPO / "anchoropt").rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("import benchmarks", "from benchmarks")):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}")
    assert not offenders, f"core imports a benchmark at module scope: {offenders}"


def test_the_core_imports_without_a_benchmark_and_then_refuses_to_guess():
    """Two properties, and the second is the one that matters.

    IMPORTABLE: `anchoropt/` must load with no benchmark present, or the adapter has become a
    dependency rather than a seam.

    AND THEN IT REFUSES: with nothing registered, core code that needs benchmark vocabulary raises
    `NoAdapterRegistered` naming the fix. It does NOT fall back to a default benchmark. An earlier
    version defaulted to `name="bfcl_v4"` and located the file itself, which put knowledge of a
    specific benchmark back into the core one level up from where it was removed.
    """
    import subprocess
    import textwrap

    code = textwrap.dedent(
        """
        import anchoropt.attribution.trace_backward as tb
        import anchoropt.mechanisms.redundant_write as rw
        from anchoropt.attribution import NoAdapterRegistered
        print("IMPORTED")
        print(rw._state_attrs())
        try:
            tb._chain_of("memory_kv_30-healthcare-0")
            print("GUESSED")
        except NoAdapterRegistered as exc:
            print("REFUSED", "register_adapter" in str(exc))
        """
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                       text=True, check=False)
    assert r.returncode == 0, f"core failed to import without a benchmark:\n{r.stderr[-700:]}"
    assert "IMPORTED" in r.stdout
    assert "core_memory" in r.stdout, "the standalone fallback must still work"
    assert "REFUSED True" in r.stdout, (
        "the core must RAISE with the fix named, not silently locate a benchmark"
    )
    assert "GUESSED" not in r.stdout


# `anchoropt/__init__.py` names the vendored evaluator's directory to extend its own package path.
# That is PACKAGING plumbing for vendored third-party code, not attribution logic: the vendored
# evaluator does `from anchoropt.eval_common import ...` because in its source repo those modules sit
# in a package of this name, and rewriting those imports would touch ~14 files whose measured
# behaviour every frozen result depends on. Different kind of coupling, out of scope here, exempted
# by name rather than by a broad pattern so a NEW instance elsewhere still fails.
_PACKAGING_EXEMPT = {"anchoropt/__init__.py"}


def test_the_core_does_not_name_or_locate_a_specific_benchmark():
    """Grep the core for the coupling this refactor removed: a benchmark NAME used as a value, or a
    path to one, in executable code. Docstrings are exempt -- they explain the inversion."""
    import tokenize

    offenders = []
    for path in sorted((REPO / "anchoropt").rglob("*.py")):
        if str(path.relative_to(REPO)) in _PACKAGING_EXEMPT:
            continue
        with path.open() as fh:
            for tok in tokenize.generate_tokens(fh.readline):
                if tok.type != tokenize.STRING:
                    continue
                if tok.line.strip().startswith(('"""', "'" * 3)):
                    continue
                if "bfcl" in tok.string.lower() or "benchmarks/" in tok.string:
                    offenders.append(f"{path.relative_to(REPO)}:{tok.start[0]}")
    assert not offenders, f"core names/locates a specific benchmark in code: {offenders}"


def test_an_unparseable_id_cannot_enter_a_chain_statistic():
    """The tolerant `chain_of` contract is bounded to ORDERING, never to counting.

    `_chain_of` returns ("unknown", 0) so trajectory files sort even when a directory holds foreign
    artifacts. But grouping refuses to report an unknown chain: before this guard, a stray file was
    grouped, traced, and folded into the per-chain survival table with nothing saying so.
    """
    import adapter as _bfcl

    from anchoropt.attribution.trace_backward import UNKNOWN_CHAIN, _group_into_chains

    _bfcl.register()
    episodes = [
        {"case_id": "memory_kv_30-healthcare-0"},
        {"case_id": "memory_kv_31-healthcare-1"},
        {"case_id": "stray_harness_file"},
        {"case_id": ""},
    ]
    chains, skipped = _group_into_chains(episodes)
    assert UNKNOWN_CHAIN not in chains, "an unknown chain must never be reportable"
    assert set(chains) == {"memory_kv-healthcare"}
    assert skipped == ["stray_harness_file", ""], "and the excluded ids must be surfaced"


def test_all_case_id_parsers_route_through_the_adapter():
    """There were SEVEN copies of this grammar, not five, with THREE unknown-id conventions:
    `None`, a silent `rec_sum`, and `"?"`. No module under anchoropt/ may hold an eighth."""
    offenders = []
    for path in sorted((REPO / "anchoropt").rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if "re.compile" in line and "memory_" in line and "kv" in line:
                offenders.append(f"{path.relative_to(REPO)}:{lineno}")
    assert not offenders, (
        f"a local copy of the case-id grammar is back: {offenders}. Use the active adapter."
    )


def test_the_three_unknown_id_conventions_are_all_preserved_deliberately():
    """Each call site had its own sentinel, and each is kept because each is USED differently:

        backend_of  -> raises   : a caller that cannot name the backend has a bug
        chain_of    -> unknown  : sorts trajectory files, which may include foreign artifacts
        chain_of()  -> "?"      : a grouping LABEL in the miner's own report

    They are documented rather than unified because unifying them now would change behaviour in
    three modules at once, which is not what a parity-preserving refactor does.
    """
    import adapter as _bfcl

    from anchoropt.attribution.attribution_miner import chain_of as miner_chain
    from anchoropt.attribution.trace_backward import _chain_of as trace_chain
    from anchoropt.learning.check_attribution import backend_of as tolerant_backend

    _bfcl.register()
    with pytest.raises(UnknownCaseId):
        _bfcl.backend_of("stray")
    assert tolerant_backend("stray") is None
    assert trace_chain("stray") == ("unknown", 0)
    assert miner_chain("stray") == "?"
