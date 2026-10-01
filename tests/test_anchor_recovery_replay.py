"""The recovery benchmark's own measurement channel, which is easy to get silently wrong.

Each test pins a defect that made the benchmark report something other than what it measured:

  * mechanism terms matched as bare SUBSTRINGS, so ordinary English scored as anchor agreement
  * tightening must be prophylactic -- it may remove a hit, never add one, and must not move a
    verdict already on the record
  * the scored projection is `mechanism + consequential_decision` ONLY; a term that appears in the
    `evidence` field is not agreement, and a test that reads the raw JSON would not notice
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "benchmarks" / "bfcl_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load():
    """Import the replay script by path -- it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location(
        "anchor_recovery_replay", REPO / "scripts" / "anchor_recovery_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REPLAY = _load()
matches = REPLAY.mechanism_term_matches


# ---- the substring trap ------------------------------------------------------------------------
#
# Bare `in` matching reported one anchor's condition as present in 37 of 47 episodes (capstone,
# cappuccinos, capabilities) when the true count was ZERO. Same class as the bare "similarity" and
# bare "archival" hits that made two earlier recoveries false positives.

@pytest.mark.parametrize("term,text", [
    ("cap", "a recap of the episode"),
    ("cap", "its capabilities"),
    ("cap", "the capstone project"),
    ("cap", "escape the loop"),
    ("clear", "clearly topic-bearing keys"),
    ("clear", "it was unclear"),
    ("full", "handled carefully"),
    ("duplicate", "duplicated"),
    ("blob", "blobs"),
])
def test_whole_word_terms_do_not_match_unrelated_english(term, text):
    assert not matches(term, text), f"{term!r} spuriously matched {text!r}"


# ---- stems must still grow ---------------------------------------------------------------------
#
# The opposite failure: a naive \b rule on both edges breaks the DELIBERATE stems, which exist so
# one term covers a family of inflections. Only the left edge may be anchored for these.

@pytest.mark.parametrize("term,text", [
    ("without retriev*", "answered without retrieving"),
    ("never consult*", "never consulting the archival store"),
    ("lookup fail*", "the lookup failed"),
    ("truncat*", "truncating the blob"),
    ("collid*", "colliding with an existing key"),
    ("evict*", "evicting a redundant copy"),
    ("exceed*", "would exceed the cap"),
    ("compact*", "compacting the entry"),
    ("rewrit*", "rewrites the value"),
])
def test_stem_terms_match_their_inflections(term, text):
    assert matches(term, text), f"stem {term!r} failed to match {text!r}"


@pytest.mark.parametrize("term,text", [
    ("cap", "the cap was exceeded"),
    ("clear", "the model proposes a clear"),
    ("full", "the container is full"),
    ("capacity", "an unresolved capacity condition"),
])
def test_whole_word_terms_still_match_genuine_uses(term, text):
    assert matches(term, text), f"{term!r} should match {text!r}"


def test_a_stem_is_still_left_anchored():
    """A stem relaxes the RIGHT edge only; it must not match mid-word."""
    assert not matches("cap*", "a recap")
    assert matches("cap*", "the capacity limit")


# ---- tightening is prophylactic, not retroactive -----------------------------------------------

def _scored_text(anchor: str) -> str:
    """EXACTLY the projection the scorer builds: mechanism + consequential_decision, lowered.

    Deliberately not `json.dumps(raw)`: the `evidence` field carries verbatim tool output (including
    error strings that contain anchor condition words), and scoring against it would turn mechanism
    agreement into a search of the runtime facts the learner was shown.
    """
    path = REPO / f"rounds/REC_{anchor}/diagnoses.json"
    rows = json.load(open(path))
    return " ".join(f"{r.get('failure_mechanism', '')} {r.get('consequential_decision', '')}"
                    for r in rows).lower()


# The term lists as the scorer holds them, so this test moves when the scorer does.
TERMS = {
    "A2": ("not found", "not_found", "missing key", "lookup fail*", "wrong key", "guessed"),
    "A3": ("duplicate", "collid*", "same key", "re-writ*", "rewrit*", "overwrite",
           "already stored"),
    "A4": ("no tool call", "without retriev*", "without querying", "no retrieval",
           "never queried", "without performing any retrieval", "without any retrieval"),
    "A8": ("clear", "destructive", "duplicate", "evict*", "remove one", "at capacity"),
    "A9": ("other container", "second store", "never searched", "never consult*",
           "not consulted", "unsearched", "only searched one", "both containers",
           "cross-container", "no archival lookup", "never touched"),
}


@pytest.mark.parametrize("anchor,expect_agreement", [
    ("A2", True), ("A3", False), ("A4", True), ("A8", False), ("A9", True),
])
def test_tightening_does_not_move_a_verdict_already_on_the_record(anchor, expect_agreement):
    """Whatever mech_ok was under bare matching, it is the same now.

    This is the argument that tightening is not "changing the criteria after a result": it cannot
    manufacture agreement, and on every corpus already scored it removes only spurious hits.
    """
    if not (REPO / f"rounds/REC_{anchor}/diagnoses.json").exists():
        pytest.skip(f"no REC_{anchor} corpus")
    txt = _scored_text(anchor)
    hits = [t for t in TERMS[anchor] if matches(t, txt)]
    assert bool(hits) is expect_agreement, f"{anchor}: mech_ok moved (hits={hits})"


def test_a8_has_no_mechanism_agreement_at_all():
    """Both of A8's old "hits" were artefacts, for two DIFFERENT reasons.

    `clear` matched inside "clearly topic-bearing keys" -- a substring artefact, killed by the
    whole-word rule. `capacity` was word-bounded and genuine as a WORD, but it appeared in a
    diagnosis whose mechanism is "a write acknowledgement was accepted as sufficient grounding": a
    read-side premature-answer failure that mentions capacity only as ambient context. A term
    satisfiable by a generic diagnosis of this surface is not mechanism agreement, which is the same
    rule that removed bare "similarity" and bare "archival" from A9.

    A8's condition is a destructive CLEAR proposed while a removable DUPLICATE exists. Neither idea
    appears anywhere in its 24 diagnoses.
    """
    if not (REPO / "rounds/REC_A8/diagnoses.json").exists():
        pytest.skip("no REC_A8 corpus")
    txt = _scored_text("A8")
    assert "clear" in txt, "fixture changed: the spurious substring is gone"
    assert not matches("clear", txt), "the spurious 'clear' still matches"
    for gone in ("destructive", "duplicate", "evict*"):
        assert not matches(gone, txt), f"{gone} unexpectedly present -- re-check the verdict"
    assert not [t_ for t_ in TERMS["A8"] if matches(t_, txt)], "A8 must have NO agreement"


def test_matching_can_only_remove_hits_never_add_them():
    """Across every corpus and every term list: tightened hits are a SUBSET of bare-substring hits."""
    for anchor, terms in TERMS.items():
        if not (REPO / f"rounds/REC_{anchor}/diagnoses.json").exists():
            continue
        txt = _scored_text(anchor)
        bare = {t for t in terms if t.rstrip("*") in txt}
        tight = {t for t in terms if matches(t, txt)}
        assert tight <= bare, f"{anchor}: tightening ADDED {tight - bare}"


# ---- the table's denominator -------------------------------------------------------------------
#
# `scored` used to mean "everything except NO_RESIDUAL", so a channel fault (NO_DIAGNOSES) or an
# absent trigger counted as an algorithm failure and silently depressed the headline. Three of the
# eight anchors are untestable, so this is the difference between 2/4 and 2/7.

def test_not_testable_verdicts_are_outside_the_denominator(tmp_path, monkeypatch, capsys):
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("recovery_table", REPO / "scripts" / "recovery_table.py")
    table = _ilu.module_from_spec(spec)
    spec.loader.exec_module(table)

    rounds = tmp_path / "rounds"
    fixture = {
        "A2": {"verdict": "EQUIVALENT", "failure_support": 24},
        "A3": {"verdict": "COINCIDENTAL", "failure_support": 24},
        "A1": {"verdict": "INSUFFICIENT_SUPPORT", "failure_support": 1},
        "A5": {"verdict": "NO_RESIDUAL", "failure_support": 0},
        "A7": {"verdict": "NO_RESIDUAL_IN_FAILURES", "failure_support": 0},
    }
    for name, rec in fixture.items():
        d = rounds / f"REC_{name}"
        d.mkdir(parents=True)
        json.dump(rec, open(d / "recovery.json", "w"))
    monkeypatch.setattr(table, "REPO", tmp_path)
    table.main()
    out = capsys.readouterr().out

    # 1 rediscovered (A2) of 2 TESTABLE (A2, A3) -- the three untestable ones are excluded.
    assert "1 / 2 TESTABLE" in out, out
    assert "NOT TESTABLE (3)" in out, out
    for name in ("A1", "A5", "A7"):
        assert name in out
    assert "3 untestable" in out


def test_every_not_testable_verdict_has_a_stated_reason():
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("recovery_table", REPO / "scripts" / "recovery_table.py")
    table = _ilu.module_from_spec(spec)
    spec.loader.exec_module(table)
    assert table.NOT_TESTABLE <= set(table.WHY_UNTESTABLE), "an untestable verdict with no reason"
    assert not (table.NOT_TESTABLE & table.COUNTS), "a verdict cannot both count and be untestable"


# ---- the replay script must not decide WHERE, WHAT or HOW ---------------------------------------
#
# The script used to inline the whole search (193 lines: boundary derivation, clause pooling,
# synthesis, expansion, arm building, per-locus loops). It now supplies evidence and scores historical
# agreement. These tests keep it that way, because the recovery result only means something if the
# search that produced it is the generic one.

def test_the_script_holds_nothing_that_could_express_a_locus_signal_or_action():
    """Structural, not stylistic: it cannot choose a coordinate it cannot even name."""
    src = (REPO / "scripts" / "anchor_recovery_replay.py").read_text()
    # CODE only: the script carries a comment naming these precisely to record that they are ABSENT,
    # and a test that counted that comment as a violation would forbid documenting the property.
    code = "\n".join(line.split("#", 1)[0] for line in src.split('"""', 2)[-1].splitlines())
    for forbidden in ("IncisionPoint", "AnchorPolicyOpt", "SearchSpaceProposal",
                      "expand_and_resume", "group_blocked_by_clause", "build_arms"):
        assert forbidden not in code, (f"the replay script references {forbidden!r} in CODE -- the "
                                       f"search belongs to core, so the script cannot steer it")


def test_synthesize_is_not_called_by_the_script():
    src = (REPO / "scripts" / "anchor_recovery_replay.py").read_text()
    assert "synthesize(" not in src, "signal synthesis is core's; the script must not call it"


def test_the_historical_anchor_is_loaded_only_AFTER_the_search_has_run():
    """The oracle boundary, checked by position rather than by reading the comments."""
    src = (REPO / "scripts" / "anchor_recovery_replay.py").read_text()
    i_search = src.index("optimize_residual(_Residual()")
    assert src.index("from rounds.anchors import ANCHORS") > i_search
    # the inventory READ (not the docstring mention) must also follow the search
    i_read = src.index('json.load(open(REPO / "docs/anchor_inventory.json"))')
    assert i_read > i_search, "the equivalence criterion was available before the search ran"


def test_the_script_passes_no_boundary_or_signal_into_the_core_call():
    """optimize_residual receives evidence only: events, states, and the residual's case ids."""
    src = (REPO / "scripts" / "anchor_recovery_replay.py").read_text()
    call = src[src.index("optimize_residual(_Residual()"):]
    call = call[:call.index(")\n")]
    for leak in ("signals=", "boundaries=", "locus", "action="):
        assert leak not in call, f"the core call passes {leak!r}, which would steer the search"
