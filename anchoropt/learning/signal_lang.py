"""A declarative signal language: proposed phi as DATA, compiled by us, never executed as code.

THE PROBLEM THIS SOLVES
-----------------------
Autonomous discovery needs to propose signals nobody pre-registered. The two obvious ways to allow
that are both unacceptable:

    a fixed list of signal names   -> the proposer can only re-select what is already known, so the
                                      accepted anchor library becomes the design of the search
    LLM-authored predicate code    -> an arbitrary patch. The closed-space guarantee is gone, and
                                      `docs/THE_LOOP.md`'s claim that no candidate can name anything
                                      outside the declared vocabulary becomes false

So a signal is an EXPRESSION TREE over fields the runtime declares, built from a closed operator
set, compiled by this module into a callable. The proposer emits JSON. No `eval`, no `exec`, no code
strings, no imports at proposal time.

WHAT THE COMPILER GUARANTEES, SO THAT REVIEW DOES NOT HAVE TO
------------------------------------------------------------
  * every referenced field is DECLARED by the runtime, with a matching type
  * every operator is in the closed set, applied to a type it accepts
  * the signal's boundary set is DERIVED as the INTERSECTION of its fields' declared boundaries

The third is the important one. A4 v1 evaluated "about to answer without looking" at
PRE_GENERATION, where the generated decision does not exist; it fired 303/303 and scored -4.95 pp
against +3.63 pp for byte-identical text one boundary later. The first version of the BFCL runtime
guarded that with a hand-written SIGNAL_BOUNDARIES table -- which promptly disagreed with its own
predicate for `duplicate_identifier`. Deriving the boundary set from the fields makes the whole
error class unrepresentable rather than merely discouraged.

HISTORY IS NOT IN THE LANGUAGE
------------------------------
There is deliberately no way to reference another step or another boundary. A signal that straddles
two decision points is observable at neither, and an anchor built on one claims a locus it does not
have. Conditions that genuinely span time -- "a write was refused earlier and a clear is proposed
now" -- read CARRIED SUMMARY fields the wrapper maintains (`last_error_kind`,
`pending_blocked_write`). Those are ordinary single-boundary fields, so the anchor stays single-locus
and `U_H(l)` keeps meaning what it says.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# ------------------------------------------------------------------------------------------------
# The closed operator set
# ------------------------------------------------------------------------------------------------
#
# `accepts` is the field types an operator may be applied to; `arity` says whether it takes a value.
# Adding an operator is a deliberate, reviewable act -- which is the point of enumerating them.
_LEAF_OPS: Mapping[str, Mapping[str, Any]] = {
    "is_true":      {"accepts": (bool,),              "needs_value": False},
    "is_false":     {"accepts": (bool,),              "needs_value": False},
    "is_none":      {"accepts": (bool, int, float, str), "needs_value": False},
    "is_not_none":  {"accepts": (bool, int, float, str), "needs_value": False},
    "eq":           {"accepts": (bool, int, float, str), "needs_value": True},
    "ne":           {"accepts": (bool, int, float, str), "needs_value": True},
    "lt":           {"accepts": (int, float),         "needs_value": True},
    "lte":          {"accepts": (int, float),         "needs_value": True},
    "gt":           {"accepts": (int, float),         "needs_value": True},
    "gte":          {"accepts": (int, float),         "needs_value": True},
    "in":           {"accepts": (str, int),           "needs_value": True},
    # Delegates to policy_tree.vacuous_result_kind -- the EXISTING detector. A second definition of
    # "vacuous" could disagree with the first, and this project has already paid for a forked
    # predicate once. `{"ranked_results": []}` is a successful call carrying no information.
    "is_vacuous":   {"accepts": (str,),               "needs_value": False},
    "vacuous_kind": {"accepts": (str,),               "needs_value": True},
}

_NODE_OPS = ("all", "any", "not")

MAX_DEPTH = 4
MAX_LEAVES = 8


class SignalSpecError(ValueError):
    """A proposed signal that is not a legal expression. Carries WHY, for the decline log.

    Distinct from a signal that compiles and does not fire: this one never becomes a controller, and
    the reason is recorded so a rejected proposal is attributable rather than merely absent.
    """


@dataclass(frozen=True)
class CompiledSignal:
    """A validated signal: its predicate, and the boundaries at which it is observable."""

    name: str
    expr: Mapping[str, Any]
    predicate: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    boundaries: frozenset[str]
    fields_used: frozenset[str]
    params_used: frozenset[str]

    def observable_at(self, boundary: str) -> bool:
        return boundary in self.boundaries


# ------------------------------------------------------------------------------------------------
# validation + compilation
# ------------------------------------------------------------------------------------------------

def _leaves(expr: Mapping[str, Any], depth: int = 0) -> list[Mapping[str, Any]]:
    if depth > MAX_DEPTH:
        raise SignalSpecError(f"expression nested deeper than {MAX_DEPTH}")
    if not isinstance(expr, Mapping):
        raise SignalSpecError(f"expression node must be an object, got {type(expr).__name__}")
    node_key = [k for k in _NODE_OPS if k in expr]
    if len(node_key) > 1:
        raise SignalSpecError(f"node declares more than one combinator: {sorted(node_key)}")
    if node_key:
        key = node_key[0]
        children = expr[key]
        if key == "not":
            children = [children]
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
            raise SignalSpecError(f"{key!r} takes a list of sub-expressions")
        if not children:
            raise SignalSpecError(f"{key!r} takes at least one sub-expression")
        out: list[Mapping[str, Any]] = []
        for child in children:
            out.extend(_leaves(child, depth + 1))
        return out
    if "field" not in expr:
        raise SignalSpecError(f"leaf must name a 'field'; got keys {sorted(expr)}")
    return [expr]


def _check_leaf(leaf: Mapping[str, Any], fields: Mapping[str, Any]) -> None:
    name = leaf["field"]
    op = leaf.get("op")
    if name not in fields:
        raise SignalSpecError(
            f"field {name!r} is not declared by this runtime; declared: {sorted(fields)}")
    if op not in _LEAF_OPS:
        raise SignalSpecError(f"operator {op!r} is not in the closed set {sorted(_LEAF_OPS)}")
    spec = _LEAF_OPS[op]
    ftype = fields[name].type
    if ftype not in spec["accepts"]:
        raise SignalSpecError(
            f"operator {op!r} cannot be applied to field {name!r} of type {ftype.__name__}; "
            f"it accepts {[t.__name__ for t in spec['accepts']]}")
    has_value = "value" in leaf or "param" in leaf
    if spec["needs_value"] and not has_value:
        raise SignalSpecError(f"operator {op!r} on {name!r} requires 'value' or 'param'")
    if not spec["needs_value"] and has_value:
        raise SignalSpecError(f"operator {op!r} on {name!r} takes no value")
    if "value" in leaf and "param" in leaf:
        raise SignalSpecError(f"leaf on {name!r} declares both 'value' and 'param'")
    enum = getattr(fields[name], "enum", ())
    if enum and op in ("eq", "ne") and "value" in leaf and leaf["value"] not in enum:
        raise SignalSpecError(
            f"field {name!r} takes one of {list(enum)}; got {leaf['value']!r}")
    if op == "in" and "value" in leaf:
        if not isinstance(leaf["value"], Sequence) or isinstance(leaf["value"], (str, bytes)):
            raise SignalSpecError(f"operator 'in' on {name!r} requires a list value")


def _eval_leaf(leaf: Mapping[str, Any], state: Mapping[str, Any],
               params: Mapping[str, Any]) -> bool:
    from anchoropt.learning.policy_tree import vacuous_result_kind

    op = leaf["op"]
    actual = state.get(leaf["field"])
    if "param" in leaf:
        pname = leaf["param"]
        if pname not in params:
            # Same discipline as the runtime's typed schema: a missing parameter PRUNES the
            # candidate rather than being silently defaulted. A signal run with a guessed threshold
            # is not the signal that was proposed.
            raise KeyError(f"signal parameter {pname!r} not supplied")
        expected = params[pname]
    else:
        expected = leaf.get("value")

    if op == "is_true":
        return bool(actual) is True
    if op == "is_false":
        return bool(actual) is False
    if op == "is_none":
        return actual is None
    if op == "is_not_none":
        return actual is not None
    if op == "is_vacuous":
        return actual is not None and vacuous_result_kind(actual) is not None
    if op == "vacuous_kind":
        return actual is not None and vacuous_result_kind(actual) == expected
    if actual is None:
        # A comparison against a field that does not exist here is FALSE, never an exception: the
        # boundary check already guarantees the field is declared at this locus, so None means the
        # host genuinely had no value (a backend exposing no similarity scores, say).
        return False
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "lt":
        return float(actual) < float(expected)
    if op == "lte":
        return float(actual) <= float(expected)
    if op == "gt":
        return float(actual) > float(expected)
    if op == "gte":
        return float(actual) >= float(expected)
    raise SignalSpecError(f"unhandled operator {op!r}")


def _eval(expr: Mapping[str, Any], state: Mapping[str, Any],
          params: Mapping[str, Any]) -> bool:
    if "all" in expr:
        return all(_eval(c, state, params) for c in expr["all"])
    if "any" in expr:
        return any(_eval(c, state, params) for c in expr["any"])
    if "not" in expr:
        return not _eval(expr["not"], state, params)
    return _eval_leaf(expr, state, params)


def compile_signal(name: str, expr: Mapping[str, Any], *, fields: Mapping[str, Any]
                   ) -> CompiledSignal:
    """Validate a proposed expression and compile it. Raises `SignalSpecError` if it is not legal.

    `fields` is the runtime's declared alphabet (`bfcl_capabilities.all_fields()`), so this function
    knows no benchmark vocabulary -- it is handed one.

    The returned `boundaries` is the INTERSECTION of the referenced fields' declared boundaries.
    Empty means the expression combines facts that never coexist at any single decision point; that
    is a rejection, not a signal, because a condition observable nowhere cannot anchor anything.
    """
    if not str(name).strip():
        raise SignalSpecError("signal needs a non-empty name")
    leaves = _leaves(expr)
    if not leaves:
        raise SignalSpecError("expression has no leaves")
    if len(leaves) > MAX_LEAVES:
        raise SignalSpecError(f"expression has {len(leaves)} leaves, more than {MAX_LEAVES}")
    for leaf in leaves:
        _check_leaf(leaf, fields)

    used = {leaf["field"] for leaf in leaves}
    params_used = {leaf["param"] for leaf in leaves if "param" in leaf}
    boundaries: set[str] | None = None
    for fname in used:
        declared = set(fields[fname].boundaries)
        boundaries = declared if boundaries is None else (boundaries & declared)
    boundaries = boundaries or set()
    if not boundaries:
        detail = ", ".join(f"{f}@{sorted(fields[f].boundaries)}" for f in sorted(used))
        raise SignalSpecError(
            f"signal {name!r} references fields that never coexist at one decision point "
            f"({detail}); a condition observable at no boundary cannot anchor an intervention")

    def predicate(state: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
        return bool(_eval(expr, state, params or {}))

    return CompiledSignal(name=str(name), expr=dict(expr), predicate=predicate,
                          boundaries=frozenset(boundaries), fields_used=frozenset(used),
                          params_used=frozenset(params_used))


def describe(expr: Mapping[str, Any]) -> str:
    """Render an expression as readable prose, for decline logs and audit trails."""
    if "all" in expr:
        return "(" + " AND ".join(describe(c) for c in expr["all"]) + ")"
    if "any" in expr:
        return "(" + " OR ".join(describe(c) for c in expr["any"]) + ")"
    if "not" in expr:
        return f"NOT {describe(expr['not'])}"
    op = expr.get("op")
    target = expr.get("param") and f"${expr['param']}" or repr(expr.get("value"))
    if op in ("is_true", "is_false", "is_none", "is_not_none", "is_vacuous"):
        return f"{expr['field']} {op}"
    return f"{expr['field']} {op} {target}"
