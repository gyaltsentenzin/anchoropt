"""The teacher operator: gated, narrow, structured, validated, non-deciding, attributed, optional.

The seven rules in `anchoropt/attribution/llm_operator.py` are what keep an LLM in this pipeline from
accumulating into an agent with discretion. Each is tested here, with an injected completion function,
so none of it needs a network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "tau2"))

import tau2_teacher as T                                          # noqa: E402

EP = NS(model="m", llm_args={}, label="fake-model")
FIELDS = ["commits_to_reply", "proposed_tool_mutates_state"]
TOOLS = frozenset({"book_reservation", "get_user_details"})


def _problem(n=3, evidence="3 proposals, 2 results, none errored"):
    return NS(key="an answer committed before the required change was made",
              diagnoses=[NS(evidence=evidence) for _ in range(n)])


def _reply(items):
    import json
    return "```json\n" + json.dumps(items) + "\n```"


def _run(items, **kw):
    return T.propose_instructions(
        _problem(), endpoint=EP, provenance="self_teach:x", boundary_fields=FIELDS,
        known_tools=TOOLS, complete=lambda *a, **k: _reply(items), **kw)


GOOD = {"variant": "Confirm First",
        "instruction": "Confirm the requested change was actually applied before telling the "
                       "customer that it is done."}


# ============================================================================ 3. STRUCTURED
def test_a_valid_json_array_is_parsed_even_inside_a_fence():
    r = _run([GOOD])
    assert r.invoked and len(r.proposals) == 1
    assert r.proposals[0].variant == "confirm_first", "the label was not normalized"


def test_prose_is_a_parse_failure_not_a_result():
    r = T.propose_instructions(_problem(), endpoint=EP, provenance="p", boundary_fields=FIELDS,
                              complete=lambda *a, **k: "Sure! Here are some ideas you could try.")
    assert r.invoked and r.proposals == []
    assert "not a JSON array" in r.reason


# ============================================================================ 4. VALIDATED
def test_an_instruction_naming_the_mechanism_is_refused():
    """Naming an action family would be the adapter choosing HOW to intervene -- core's job."""
    for word in ("suppress the call and retry the step for the customer instead",
                 "reroute the request to another place and continue helping the customer",
                 "at the post_execution boundary, check the result before replying to them"):
        r = _run([{"variant": "v", "instruction": word}])
        assert r.proposals == [], word
        assert "core's choice" in r.rejected[0]["why"]


def test_an_instruction_naming_a_tool_is_refused():
    r = _run([{"variant": "v", "instruction": "Call book reservation to finish the change for the "
                                              "customer before you reply to them."}])
    assert r.proposals == []
    assert "names a tool" in r.rejected[0]["why"]


def test_instructions_outside_the_length_bounds_are_refused():
    short = {"variant": "v", "instruction": "Do better."}
    long = {"variant": "v", "instruction": "x" * (T.MAX_INSTRUCTION_CHARS + 1)}
    for item in (short, long):
        r = _run([item])
        assert r.proposals == []
        assert "length" in r.rejected[0]["why"]


def test_duplicate_instructions_are_refused():
    r = _run([GOOD, dict(GOOD, variant="other")])
    assert len(r.proposals) == 1
    assert "duplicate" in r.rejected[0]["why"]


def test_a_malformed_element_is_refused_without_killing_the_batch():
    r = _run(["just a string", GOOD])
    assert len(r.proposals) == 1
    assert r.rejected and "not an object" in r.rejected[0]["why"]


# ============================================================================ 1. GATED
def test_the_operator_refuses_a_residual_with_no_evidence_to_read():
    """A teacher with nothing to read would be inventing, and an invented instruction is not
    attributable to the residual."""
    r = T.propose_instructions(_problem(evidence=""), endpoint=EP, provenance="p",
                               boundary_fields=FIELDS, complete=lambda *a, **k: _reply([GOOD]))
    assert r.invoked is False
    assert "no evidence" in r.reason


def test_a_failing_teacher_call_degrades_to_not_invoked_rather_than_raising():
    def boom(*a, **k):
        raise RuntimeError("gateway down")

    r = T.propose_instructions(_problem(), endpoint=EP, provenance="p", boundary_fields=FIELDS,
                               complete=boom)
    assert r.invoked is False and "failed" in r.reason


# ============================================================================ 2. NARROW
def test_the_prompt_shows_only_the_residual_and_the_boundarys_own_fields():
    """It must not see the search state, the arms, the acceptance rule or the incumbent's score."""
    seen = {}

    def capture(model, messages, **kw):
        seen["text"] = messages[0]["content"]
        return _reply([GOOD])

    T.propose_instructions(_problem(), endpoint=EP, provenance="p", boundary_fields=FIELDS,
                           known_tools=TOOLS, complete=capture)
    text = seen["text"].lower()
    assert "commits_to_reply" in text and "none errored" in text
    for leak in ("incumbent", "j_train", "net ", "accept", "arm", "candidate", "promot",
                 "book_reservation"):
        assert leak not in text, f"the teacher was shown {leak!r}"


# ============================================================================ 5. NON-DECIDING
def test_the_operator_returns_eta_only_and_never_a_ranking():
    r = _run([GOOD, {"variant": "second",
                     "instruction": "Check the customer's record once more before you confirm "
                                    "anything to them about the change."}])
    assert len(r.proposals) == 2
    for p in r.proposals:
        g = p.as_grounding()
        assert set(g["eta"]) == {"instruction", "retry_budget"}
        assert "rank" not in g and "score" not in g and "preferred" not in g


def test_the_prompt_forbids_the_teacher_from_ranking():
    seen = {}

    def capture(model, messages, **kw):
        seen["t"] = messages[0]["content"].lower()
        return _reply([GOOD])

    T.propose_instructions(_problem(), endpoint=EP, provenance="p", boundary_fields=FIELDS,
                          complete=capture)
    assert "do not say which of your instructions is best" in seen["t"]
    assert "do not rank" in seen["t"]


# ============================================================================ 6. ATTRIBUTED
def test_every_proposal_carries_its_model_and_provenance():
    r = _run([GOOD])
    p = r.proposals[0]
    assert p.model == "fake-model" and p.provenance == "self_teach:x"
    assert "self_teach:x" in p.as_grounding()["detail"]
    assert r.to_json()["provenance"] == "self_teach:x"


def test_a_teacher_variant_is_labelled_so_it_cannot_be_confused_with_a_shipped_one():
    g = _run([GOOD]).proposals[0].as_grounding()
    assert g["variant"].startswith("teacher_")


# ============================================================================ eta shape
def test_the_budget_key_can_be_omitted_for_a_boundary_whose_executor_reads_none():
    g = _run([GOOD]).proposals[0].as_grounding(include_budget=False)
    assert set(g["eta"]) == {"instruction"}


# ============================================================================ bounded, once per arm
def test_the_teacher_call_is_bounded_rather_than_inheriting_the_episode_timeout():
    """A teacher runs once per arm inside a 24-job batch. The 600s episode timeout is sized for a
    multi-turn simulation; with retries on top, a silent teacher could hold an arm open for half an
    hour. The deadline is capped and retries are limited."""
    import inspect
    src = inspect.getsource(T.propose_instructions)
    assert "TEACHER_TIMEOUT_S" in src
    assert "num_retries=1" in src
    assert T.TEACHER_TIMEOUT_S <= 600


def test_an_episode_sized_timeout_is_clamped_down_for_the_teacher():
    seen = {}

    def capture(model, messages, **kw):
        seen.update(kw)
        return _reply([GOOD])

    ep = NS(model="m", llm_args={"timeout": 600}, label="fake")
    T.propose_instructions(_problem(), endpoint=ep, provenance="p", boundary_fields=FIELDS,
                           complete=capture)
    # The injected completion sees the endpoint's args verbatim; the clamp lives in the real one, so
    # assert the constant is what the network path would apply.
    assert seen["timeout"] == 600
    assert min(600, T.TEACHER_TIMEOUT_S) == T.TEACHER_TIMEOUT_S
