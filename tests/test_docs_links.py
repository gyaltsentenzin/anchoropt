"""Relative markdown links must resolve.

The progression is navigated by link -- the top README points into `rounds/`, each round points at
its neighbours and its frozen spec. A broken link in that chain is a broken argument, so this is
checked rather than trusted.

Files not yet written are listed in PLANNED. A planned file that has since been created must be
removed from the list; the test enforces that too, so the exemption list cannot rot into a
permanent excuse.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Still to be written. Every entry is a commitment, not a waiver.
#
# NOTE: an adapter framework (`TrajectoryAdapter` / `ActionAdapter` / a toy `minimal_adapter`) was
# planned and then deliberately dropped -- see docs/GENERALIZABILITY.md. Abstracting an interface from
# ONE benchmark encodes that benchmark's accidents into a base class, so the seams are documented
# instead. Do not re-add those paths here without a second benchmark to generalise from.
PLANNED: set[str] = set()

_LINK = re.compile(r"\[([^\]]+)\]\(([^)#]+?)\)")


# Directories that contain generated or vendored markdown we do not author. `.pytest_cache` is the
# one that actually bit: pytest writes a README.md there on its FIRST run, so a second run collected
# one extra parametrised case and the suite total moved 161 -> 162. A test count that depends on
# whether the suite has run before is a test bug, not a curiosity.
_SKIP_DIRS = {".venv", ".git", ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules",
              "build", "dist",
              # Vendored third party (Apache-2.0 BFCL fork). Its docs are upstream's, not ours: we
              # neither author them nor control their links, and rewriting them would break the
              # diff against upstream.
              "harness"}


def _markdown_files() -> list[Path]:
    """Markdown this repo AUTHORS -- never generated caches or vendored trees."""
    return sorted(
        p for p in REPO.glob("**/*.md")
        if not _SKIP_DIRS & set(p.parts)
        # `.venv` above is an exact name match, so a venv created in-tree under any other name --
        # `.venv-harness`, `.venv-appworld-anchoropt` -- was still walked, and the suite then failed
        # on a dependency's own docs (dashscope ships a `[source](url)` placeholder). Which
        # third-party markdown a checkout happens to contain is not this repo's correctness.
        and not any(part.startswith(".venv") for part in p.parts)
    )


def test_markdown_files_exist():
    assert _markdown_files(), "no markdown found -- the test is looking in the wrong place"


@pytest.mark.parametrize("md", _markdown_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_relative_links_resolve(md: Path):
    broken = []
    for label, target in _LINK.findall(md.read_text()):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        rel = (md.parent / target).resolve()
        try:
            planned = str(rel.relative_to(REPO)) in PLANNED
        except ValueError:
            planned = False
        if not rel.exists() and not planned:
            broken.append(f"[{label}]({target})")
    assert not broken, f"{md.relative_to(REPO)} has unresolvable links: {broken}"


def test_planned_files_are_still_missing():
    """Once a planned file lands, drop it from PLANNED so real breakage is caught again."""
    landed = [p for p in sorted(PLANNED) if (REPO / p).exists()]
    assert not landed, f"these now exist and must be removed from PLANNED: {landed}"


def test_every_round_has_a_readme_and_states_how_it_was_accepted():
    """Every round ships a README and an acceptance document.

    The acceptance document's NAME is a claim, so any of three names satisfies this and the specific
    mapping is pinned in tests/test_progression_integrity.py:
        FROZEN.md    criteria pre-registered before launch
        CRITERIA.md  criteria transcribed, or variants swept with results visible
        OVERRIDE.md  a frozen clause was failed and overridden
    """
    rounds = sorted(d for d in (REPO / "rounds").glob("T*") if d.is_dir())
    assert len(rounds) == 9, f"expected 9 rounds, found {[d.name for d in rounds]}"
    for d in rounds:
        assert (d / "README.md").exists(), f"{d.name} has no README"
        accepted = (
            list(d.glob("*FROZEN*.md")) + list(d.glob("CRITERIA.md")) + list(d.glob("OVERRIDE.md"))
        )
        assert accepted, f"{d.name} does not state how it was accepted"


def test_rounds_index_links_to_every_round():
    index = (REPO / "rounds" / "README.md").read_text()
    for d in sorted(x for x in (REPO / "rounds").glob("T*") if x.is_dir()):
        assert f"{d.name}/" in index, f"rounds/README.md does not link to {d.name}"


def test_collection_ignores_generated_markdown():
    """The suite total must not depend on whether the suite has run before.

    pytest writes `.pytest_cache/README.md` on its first run. Globbing it made the collected test
    count move 161 -> 162 between the first and second invocation in a fresh clone -- which looks
    exactly like a flaky suite. Guard the exclusion directly.
    """
    for md in _markdown_files():
        assert not _SKIP_DIRS & set(md.parts), f"{md} is generated/vendored and must not be checked"


def test_authored_markdown_is_all_tracked_by_git():
    """Every doc we lint links for should be a file we actually ship."""
    import subprocess
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO, capture_output=True, text=True, check=False,
    ).stdout.split()
    if not tracked:              # not a git checkout (e.g. an export); nothing to compare against
        pytest.skip("not a git checkout")
    tracked_set = {str(REPO / t) for t in tracked}
    for md in _markdown_files():
        assert str(md) in tracked_set, f"{md.relative_to(REPO)} is untracked -- generated?"


def test_expansion_ladder_is_documented_in_order():
    """When signals run out there is a PROCEDURE, not a hunch.

    Two orthogonal escalations, both needed by anyone who hits exhaustion:
      richer evidence   lexical -> semantic -> structural
      earlier in flow   query phase -> write phase
    Scoped to the ladder section, since these words also occur elsewhere in the doc.
    """
    doc = (REPO / "docs" / "THE_LOOP.md").read_text()
    start = doc.index("three rungs, cheapest first")
    end = doc.index("The bar a new detector must clear")
    ladder = doc[start:end].lower()

    positions = [ladder.index(rung) for rung in ("lexical", "semantic", "structural")]
    assert positions == sorted(positions), "the three rungs must appear cheapest-first"

    # The orthogonal move, and the evidence for it: a read-side action cannot conjure an unwritten
    # fact, which is why one iteration's rank 1 was rejected without spending an arm.
    assert "query" in ladder and "write" in ladder
    assert "0 of 50" in ladder
    assert "220/220" in ladder, "write-phase concentration is the measured justification"


def test_collaborator_guide_names_code_at_every_step():
    """Each of the six flow steps must point at real code a collaborator can open.

    A guide that describes a procedure without saying which module implements it makes the reader
    reverse-engineer the mapping -- which is the work the guide exists to save.
    """
    doc = (REPO / "COLLABORATORS.md").read_text()
    # One "> **Code:**" block per step.
    assert doc.count("> **Code:**") >= 6, "every step needs a code pointer"

    # And every module path mentioned in those blocks must exist.
    import re
    missing = [
        m for m in set(re.findall(r"\[`((?:anchoropt|scripts|benchmarks)/[\w/.]+\.py)`\]", doc))
        if not (REPO / m).exists()
    ]
    assert not missing, f"COLLABORATORS.md points at files that do not exist: {missing}"


def test_backward_induction_explains_the_ordering_by_branching():
    """Downstream-first is not a fallback for when cheap signals run dry -- it is the procedure.

    The reason is branching: a write-side anchor changes what is IN the store, so every later read runs
    against a different world and the scenarios needing validation multiply. Reaching upstream before
    the downstream is settled means validating against a moving target.
    """
    loop = (REPO / "docs" / "THE_LOOP.md").read_text()
    assert "Backward induction" in loop
    assert "general procedure" in loop, "the ordering must not read as a last resort"
    assert "moving target" in loop
    assert "multipl" in loop.lower(), "the branching argument must be stated"

    readme = (REPO / "README.md").read_text().lower()
    # The CLAIM, not one phrasing of it: downstream is settled before moving upstream.
    assert "upstream" in readme and "downstream" in readme, (
        "the README must state the downstream-first ordering somewhere"
    )


def test_lookahead_section_distinguishes_necessary_from_pathological():
    """A1 CAUSED the failures A2/A3 fix, so the method cannot simply forbid induced errors.

    It classifies them instead: an induced signal whose follow-on is forced is the cost of a real
    repair; one that merely invites a new failure mode is harm. Without that distinction the README
    would be self-contradicting -- two of its four anchors propagate downstream.
    """
    # Section numbering is a style choice; slice on the re-mining discussion that owns this rule.
    doc = (REPO / "docs" / "THE_LOOP.md").read_text()
    start = doc.lower().index("induced errors: necessary cost of a repair")
    sec = doc[start:start + 4000]
    assert "necessary" in sec.lower() and "pathological" in sec.lower()
    # "80 %" or "0.8" -- either states the threshold; what matters is that one is stated.
    assert "80" in sec or "0.8" in sec, (
        "the forced-rate criterion makes the necessary/pathological split measurable, not rhetorical"
    )
    # The honest limit: it is ONE step, not a multi-round search.
    assert "one*-step" in sec or "one-step" in sec.lower()
    assert "25 of those 29" in sec, "the rejected contrast case is what makes the rule credible"


def test_cited_line_numbers_in_the_flow_comparison_are_current():
    """COLLABORATORS.md cites firing-site line numbers. Line numbers drift on every edit -- they had
    already shifted by 3-12 lines before this test existed -- so pin them to the real code."""
    import re
    doc = (REPO / "COLLABORATORS.md").read_text()
    src = (REPO / "benchmarks/bfcl_v4/evaluator/memory_evaluator.py").read_text().splitlines()

    # anchor -> a substring that uniquely identifies its firing site
    sites = {
        "A3": 'enable_redundant_write_suppress")',
        "A4": 'enable_zero_call_reprompt")',
        "A1": 'enable_capacity_repair")',
        "A2": 'REGISTRY_BY_KEY["on_domain_error_key_not_found"], test_category)',
    }
    actual = {}
    for name, needle in sites.items():
        hits = [i + 1 for i, ln in enumerate(src) if needle in ln]
        assert hits, f"{name}'s firing site not found -- did the guard change?"
        actual[name] = hits[0]

    cited = dict(re.findall(r"lines (\d+) \(A3\), (\d+) \(A4\), (\d+) \(A1\), (\d+) \(A2\)", doc)
                 and zip(("A3", "A4", "A1", "A2"),
                         (int(x) for x in re.search(
                             r"lines (\d+) \(A3\), (\d+) \(A4\), (\d+) \(A1\), (\d+) \(A2\)",
                             doc).groups())))
    assert cited, "the firing-site line table is missing from COLLABORATORS.md"
    for name, n in cited.items():
        assert n == actual[name], (
            f"{name}: doc says line {n}, code has {actual[name]} -- update COLLABORATORS.md"
        )


def test_the_five_layer_model_names_all_five_layers_in_order():
    """OBSERVABILITY_VS_DECISION_CONTEXT.md is a conceptual doc, so pin its spine.

    The claim that makes the doc worth having is that these are DISTINCT layers and that a failure in
    any of them looks like a policy problem from outside. If a later edit drops one, the argument
    silently becomes the three-layer one it was written to replace.
    """
    body = (REPO / "docs" / "OBSERVABILITY_VS_DECISION_CONTEXT.md").read_text()
    for layer in ("observability", "error attribution", "decision context", "local policy", "action"):
        assert layer in body, f"the {layer!r} layer is missing"
    # The distinction itself, not just the words.
    assert "information for UNDERSTANDING the decision" in body
    assert "information USED TO MAKE the decision" in body


def test_decision_context_claim_matches_the_anchor_set():
    """The doc claims all but two anchors change what EXECUTES without informing the model.

    That is a factual claim about the anchor set, so derive it rather than trusting the prose: only
    the reprompt family can supply decision context. The ratio moved when A5-A7 were added -- all
    three are reroutes -- so the prose states the RULE and the counts are derived here.
    """
    from anchoropt.anchor import Action
    from rounds.anchors import ANCHORS, E1

    all_anchors = (*ANCHORS, E1)
    informing = [a.name for a in all_anchors if a.action is Action.REPROMPT]
    non_informing = [a.name for a in all_anchors if a.action is not Action.REPROMPT]

    assert len(all_anchors) == 9, "A1-A5 + A7 + A8 + A9, plus E1"
    assert sorted(informing) == ["A2", "A4"], (
        "A2/A4 remain the only anchors that could supply decision context"
    )
    # A9 is REROUTE by action, so it counts as non-informing under the action-family rule -- but it is
    # the one case where that rule is genuinely inadequate: it reroutes a READ, so it changes what the
    # model SEES without instructing it. The doc must say so rather than filing it under "executes".
    assert len(non_informing) == 7, (
        "A1, A3, A5, A7, A8, A9, E1 are not reprompts -- but see the A9 exception below"
    )
    doc = (REPO / "docs" / "OBSERVABILITY_VS_DECISION_CONTEXT.md").read_text().lower()
    assert "data channel" in doc and "instruction channel" in doc, (
        "A9 enriches decision context through the DATA channel; the doc must record that the "
        "reprompt-only rule does not cleanly cover it"
    )

    body = (REPO / "docs" / "OBSERVABILITY_VS_DECISION_CONTEXT.md").read_text()
    low = body.lower()
    # The claim is a RULE about the reprompt family, not a frozen headcount -- a headcount in prose
    # goes stale the moment an anchor is added, which is exactly what happened here.
    assert "only the reprompt family" in low or "only reprompt" in low, (
        "the doc must state the RULE (only reprompt can inform) rather than a headcount that "
        "goes stale whenever an anchor is added"
    )
    for name in ("A5", "A7", "A8"):
        assert name in body, f"{name} must appear in the observability doc's accounting"


def test_readme_prompt_comparison_arithmetic_is_derived_not_asserted():
    """docs/GENERALIZABILITY.md claims 7 of 9 admissible cells are unreachable by any prompt. Derive it.

    A prompt acts only at pre_generation, where feasible_actions() leaves {noop, reprompt}. If the
    grid or its exclusions ever change, this claim must change with them.
    """
    from anchoropt.anchor import Action, IncisionPoint, feasible_actions

    admissible = [(p, a) for p in IncisionPoint for a in feasible_actions(p)]
    pre_gen = [c for c in admissible if c[0] is IncisionPoint.PRE_GENERATION]

    assert len(admissible) == 9
    assert {a for _, a in pre_gen} == {Action.NOOP, Action.REPROMPT}
    unreachable = len(admissible) - len(pre_gen)
    assert unreachable == 7

    body = (REPO / "docs" / "GENERALIZABILITY.md").read_text().lower()
    # Accept either the numeral or the word, in either order -- the CLAIM is what is pinned.
    stated = (f"{unreachable} of the {len(admissible)}" in body
              or f"{unreachable} of {len(admissible)}" in body
              or "seven of the nine" in body or "seven of nine" in body)
    assert stated, (
        f"docs/GENERALIZABILITY.md must state that {unreachable} of {len(admissible)} admissible "
        "cells are unreachable by a prompt -- this is the structural limit of prompting"
    )


def test_readme_rl_comparison_states_the_operational_claim_and_its_boundary():
    """The RL comparison is only useful if it says where RL WINS too."""
    body = (REPO / "docs" / "GENERALIZABILITY.md").read_text()
    # Heading text is a style choice; the section must exist and compare against BOTH alternatives.
    assert "prompting" in body.lower() and ("RL" in body or "reinforcement" in body.lower())

    # The distribution-shift argument the section exists to make.
    assert "re-selection problem, not a" in body
    assert "retraining problem" in body

    # ...and the honest boundary. A comparison that only flatters the method is marketing.
    # The boundary claim in any phrasing: RL wins when the MODEL must change.
    low = body.lower()
    assert "better tool" in low and ("never make it smarter" in low or "around a fixed model" in low), (
        "the RL comparison must say where RL is the better choice, not only where anchors win"
    )
    assert "never make it smarter" in body
    # Interpretability is claimed for the RULES, not for the inference.
    assert "assumption-free" in body.lower(), (
        "the interpretability claim must be scoped: readable rules != assumption-free inference"
    )


def test_collaborator_guide_emphasises_sharding_and_says_why():
    """Sharding is the highest-leverage thing to copy from the run protocol, so it must be prominent
    AND justified. "It is faster" is the wrong reason and would invite skipping it."""
    body = (REPO / "COLLABORATORS.md").read_text()
    low = body.lower()

    assert "--shard" in body, "the flag a collaborator would actually use must be named"
    assert "six independent jobs" in low
    assert "not byte-reproducible" in low, (
        "the REASON is determinism, not wall-clock -- stating only the speedup invites skipping it"
    )
    # The three per-shard requirements, each learned from a broken fleet.
    assert "arm × backend" in body or "arm x backend" in body
    assert "never on cache keys" in low
    assert "same** store fingerprint" in body or "same store fingerprint" in low
    # The soundness precondition, and the instruction to check it on YOUR corpus.
    assert "no case dependency crosses" in low
    assert "check the same property on your corpus" in low
    # And the axis confusion that sharding invites.
    assert "not permission to raise" in low


def test_no_authored_doc_still_calls_a_later_anchor_unshipped():
    """"A5 is in flight and not in this repo" was true once and shipped stale for several commits.

    A collaborator reading it would conclude the write-side anchors do not exist here. Anything that
    IS in ANCHORS must not be described as absent, in any authored markdown.
    """
    import sys

    sys.path.insert(0, str(REPO))
    from rounds.anchors import ANCHORS

    shipped = {a.name for a in ANCHORS}
    offenders = []
    for md in _markdown_files():
        text = md.read_text()
        for name in shipped:
            for claim in (f"{name} is in flight", f"{name} is in flight and not in this repo",
                          f"{name} mining, still in flight"):
                if claim in text:
                    offenders.append(f"{md.relative_to(REPO)}: {claim!r}")
    assert not offenders, f"accepted anchors described as unshipped: {offenders}"


def test_narrative_docs_do_not_fix_the_anchor_count_in_prose():
    """"all four anchors" ages badly, and did. Prose should describe the RULE, not the headcount.

    FROZEN.md files are exempt: they are pre-registered records of a moment and must never be edited.
    """
    import sys

    sys.path.insert(0, str(REPO))
    from rounds.anchors import ANCHORS

    stale = ("all four anchors", "four accepted anchors", "all four accepted anchors")
    offenders = []
    for md in _markdown_files():
        if "FROZEN" in md.name or md.name == "OVERRIDE.md":
            continue          # pre-registered; frozen by design
        low = md.read_text().lower()
        for phrase in stale:
            if phrase in low:
                offenders.append(f"{md.relative_to(REPO)}: {phrase!r}")
    assert not offenders, (
        f"the stack has {len(ANCHORS)} anchors; these hardcode a stale count: {offenders}"
    )


def test_every_referenced_image_exists_and_carries_alt_text():
    """A broken image link ships silently -- it renders as a small grey box nobody reports.

    Also requires non-empty alt text: these docs are referenced by people checking numbers, and
    `![](fig.png)` is unreadable with a screen reader and unsearchable for everyone else.
    """
    import re as _re

    pattern = _re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    missing, unlabelled = [], []
    for md in sorted(REPO.rglob("*.md")):
        rel = md.relative_to(REPO)
        if any(part in {".venv", "benchmarks", "node_modules"} for part in rel.parts):
            continue
        body = md.read_text()
        for match in pattern.finditer(body):
            alt, target = match.group(1), match.group(2)
            if target.startswith(("http://", "https://", "data:")):
                continue
            # Skip syntax shown as an EXAMPLE rather than used as a reference -- docs/img/README.md
            # quotes `![](...)` to say don't do that, and a checker that cannot tell the difference
            # forces the guidance to be deleted to keep itself green.
            line_start = body.rfind("\n", 0, match.start()) + 1
            if "`" in body[line_start:match.start()]:
                continue
            resolved = (md.parent / target.split("#")[0]).resolve()
            if not resolved.exists():
                missing.append(f"{rel} -> {target}")
            if not alt.strip():
                unlabelled.append(f"{rel} -> {target}")

    assert not missing, f"referenced image(s) do not exist: {missing}"
    assert not unlabelled, f"image(s) with empty alt text: {unlabelled}"


def test_the_loop_figure_is_referenced_from_both_places_that_explain_the_loop():
    """The figure is the fastest way to see where anchors attach, so the README's intervention-points
    section and THE_LOOP must both show it rather than one linking to the other."""
    figure = "anchoropt_agent_loop.png"
    assert figure in (REPO / "README.md").read_text()
    assert figure in (REPO / "docs" / "THE_LOOP.md").read_text()
    assert (REPO / "docs" / "img" / figure).exists(), (
        f"docs/img/{figure} is missing -- save the diagram there; both docs reference it"
    )
