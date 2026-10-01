"""Install a core-synthesized Controller from a JSON spec. Imported for its side effect.

    ANCHOROPT_CONTROLLER_SPEC=/path/to/controller.json  python -c "import install_controller"

WHY THIS REPLACES A HAND-WRITTEN PREDICATE. Its predecessor (`install_phi.py`) hard-coded two shapes
and read one threshold from the environment, which was fine while every arm tested the SAME
conjunction at a different theta. It cannot carry cycle #2's controller: that phi is whatever
`optimize_residual` synthesized over the runtime's declared fields, so its shape is not known when the
runner is written. Hand-transcribing it would put a human back inside the loop at exactly the point
the loop is supposed to own -- and a transcription error would run a different trigger under the
loop's name.

So the spec is DATA, emitted by core, and this file only evaluates it. The grammar is deliberately
tiny and total: an atom over one declared field, or a conjunction of atoms. Anything else is refused
loudly rather than approximated.

    {"name": "...", "locus": "post_execution",
     "eta": {"primitive": "...", ...},
     "predicate": {"all": [{"field": "best_similarity", "op": "lt", "value": 0.294},
                           {"field": "searched_other_container", "op": "falsy"}]}}

A MISSING FIELD IS NOT A FIRING. If a state lacks a field the predicate reads, the atom is False --
the same rule the runtime hook applies to a raising phi. That keeps "the condition did not hold"
distinguishable from "the state never carried what phi reads", which is a defect this project has paid
for more than once.
"""

import json
import os

_OPS = {
    "lt": lambda v, x: v is not None and float(v) < float(x),
    "lte": lambda v, x: v is not None and float(v) <= float(x),
    "gt": lambda v, x: v is not None and float(v) > float(x),
    "gte": lambda v, x: v is not None and float(v) >= float(x),
    "eq": lambda v, x: v == x,
    # `signal_grammar.py` (the structured-search path) spells equality "equals" while
    # `signal_lang.py` spells it "eq". Both are core's own vocabulary, and this file evaluates
    # whatever core emits -- 6 of the 219 arms in the first live round carried "equals" and
    # raised `unknown op` at install time. Aliased, not renamed: rewriting the emitted spec
    # would edit the arm rather than run it.
    "equals": lambda v, x: v == x,
    "ne": lambda v, x: v != x,
    "truthy": lambda v, x: bool(v),
    "falsy": lambda v, x: v is False,          # deliberately NOT `not v`: see below
    "present": lambda v, x: v is not None,
    "absent": lambda v, x: v is None,
}
# `falsy` means the field was OBSERVED False, not merely absent-or-empty. A synthesized conjunct like
# "the other store was not consulted" must not be satisfied by a state that never recorded whether it
# was -- that is how an arm fires on episodes its residual does not contain.


class SpecPredicate:
    """Duck-typed to the hook contract: `.fires_on(state)`, `.eta`, `.name`."""

    def __init__(self, spec):
        self.spec = dict(spec)
        self.name = str(self.spec.get("name") or "phi")
        self.eta = dict(self.spec.get("eta") or {})
        # PHASE ELIGIBILITY travels with the controller. Default "query": an undeclared controller
        # must not become active during storage construction just because the spec omitted the field.
        self.phase = str(self.spec.get("phase") or "query")
        self._pred = self.spec.get("predicate") or {}
        self._check(self._pred)

    @staticmethod
    def _check(node):
        # A DECLARED SIGNAL: the host ships the predicate, so the spec names it instead of describing
        # an atom. This is what a Phi seeded from the residual family's grounded observables emits --
        # those signals are the host's own, not grammar atoms synthesized by expansion.
        if "declared_signal" in node:
            if not str(node["declared_signal"] or "").strip():
                raise ValueError("`declared_signal` must name a signal")
            return
        if "all" in node:
            terms = node["all"]
            if not isinstance(terms, list) or not terms:
                raise ValueError("`all` must be a non-empty list of atoms")
            for t in terms:
                SpecPredicate._check(t)
            return
        if "field" not in node or "op" not in node:
            raise ValueError(f"atom needs `field` and `op`: {node!r}")
        if node["op"] not in _OPS:
            raise ValueError(f"unknown op {node['op']!r}; known: {sorted(_OPS)}")

    def _eval(self, node, state):
        if "declared_signal" in node:
            # Evaluated BY THE HOST, by name. Imported lazily and from the same module the run
            # imports, so the controller and the evaluator cannot disagree about what the signal
            # means -- a second implementation here is exactly how an offline-validated predicate
            # stops matching the one that runs.
            from anchoropt.memory_gates import evaluate_signal as _es
            # PARAMS TRAVEL WITH THE PREDICATE. Evaluating a parameterized signal with `{}` raises,
            # and "a raising phi is not a firing" then makes the controller silently never fire --
            # an installed arm that runs as the control and reports a measured zero.
            return bool(_es(str(node["declared_signal"]), dict(state or {}),
                            dict(node.get("params") or {})))
        if "all" in node:
            return all(self._eval(t, state) for t in node["all"])
        return bool(_OPS[node["op"]](state.get(node["field"]), node.get("value")))

    def fires_on(self, state):
        try:
            return self._eval(self._pred, state or {})
        except Exception:
            return False          # a raising phi is NOT a trigger

    # BOTH NAMES, deliberately. The runtime hook calls `.fires_on(state)`; every core comparison
    # utility -- firing_vector, fingerprint, verify_projection -- calls `.evaluate(state)`. Exposing
    # only one made a round-trip check report ZERO firings against a predicate that fires 14 times:
    # `firing_vector` called the missing method, the call raised, and the "a raising phi is not a
    # trigger" rule turned the AttributeError into a plausible all-False vector. That rule is right at
    # runtime and dangerous in a verification path, so the fix belongs here rather than there.
    evaluate = fires_on


def _install_one(spec, *, source=""):
    """Install one spec at its declared locus and report it. Returns (name, locus, identity_key)."""
    from anchoropt.runtime_hook import install
    locus = str(spec.get("locus") or "post_execution")
    p = SpecPredicate(spec)
    install(locus, p)
    key = "/".join(str(spec.get(k) or "") for k in
                   ("locus", "signal", "action", "operator", "capability_id", "phase"))
    print("[controller] installed %s at %s  eta=%s%s"
          % (p.name, locus, sorted(p.eta), ("  <- " + source) if source else ""), flush=True)
    return p.name, locus, key


# ---------------------------------------------------------------------------------------------------
# THE STACK. `ANCHOROPT_CONTROLLER_SPEC` installs ONE controller, which made the moving incumbent
# inexpressible from the runner: every arm was structurally `H0 + exactly one controller`, so an
# accepted discovery could never be carried into the next round's baseline and each round re-searched
# from H0. Measured instance: rounds/AUTORUN/R1_RESULT.json reports controllers_installed=0 and
# incumbent/correct=18 on kv -- 18 being exactly the control score of an already-accepted +4 controller.
#
# `anchoropt.runtime_hook` ALREADY supports a stack (`_INSTALLED` is dict[locus, list], `install()`
# appends, `decide()` iterates first-fire-wins), so nothing in core changes here. Only the loader was
# single-controller.
#
# ORDER IS DATA. `decide()` is first-fire-wins, so at a shared locus the order of the list is part of
# the arm. It is taken verbatim from the stack file and never sorted or inferred.
#
# The two variables COMPOSE rather than override: the stack is the incumbent being carried forward and
# SPEC is the candidate under test on top of it, which is exactly the paired shape a round needs. The
# candidate is installed LAST so an incumbent controller at the same locus keeps its precedence, and a
# candidate that only fires where the incumbent does not is measurable rather than shadowed.
# ---------------------------------------------------------------------------------------------------
# WHICH COPY OF THIS FILE IS RUNNING? Asserted, because getting it wrong voided a 3-arm GPU round.
#
# `run_memory_eval.py` is invoked BY PATH from a shared tree, and Python puts a script's own directory
# at sys.path[0] AHEAD of PYTHONPATH. So `import_module("install_controller")` resolved to the SHARED
# copy, which had no stack support: the arms installed only their candidate and the control installed
# nothing, while a PROBE process (which did import the isolated copy) printed "STACK of 1 installed"
# into the same log. The log showed a 2-controller stack; zero relocate_* gates fired.
#
# ANCHOROPT_EXPECT_LOADER_UNDER lets the caller state where this module MUST have come from, and a
# mismatch is fatal here rather than a silent wrong-arm measurement. Unset = no assertion, so nothing
# changes for callers that do not opt in.
_expect = os.environ.get("ANCHOROPT_EXPECT_LOADER_UNDER")
if _expect and _expect not in os.path.abspath(__file__):
    raise SystemExit(
        "LOADER IDENTITY FAILED: %s resolved to %s, which is not under %r. A shared namesake shadowed "
        "the isolated copy, so the controller stack would NOT have installed and the arm would have "
        "measured as the control. Deliver the loader under a DISTINCT module name rather than by "
        "shadowing, and set ANCHOROPT_PHI_MODULE to that name."
        % (__name__, os.path.abspath(__file__), _expect))

_STACK_PATH = os.environ.get("ANCHOROPT_CONTROLLER_STACK")
_PATH = os.environ.get("ANCHOROPT_CONTROLLER_SPEC")
_installed = []

if _STACK_PATH:
    with open(_STACK_PATH) as _fh:
        _stack = json.load(_fh)
    # Accept either a bare list of paths or {"controllers": [...]} carrying provenance alongside.
    if isinstance(_stack, dict):
        _entries = _stack.get("controllers") or _stack.get("specs") or []
    else:
        _entries = _stack
    if not isinstance(_entries, list):
        raise ValueError("ANCHOROPT_CONTROLLER_STACK must be a JSON list of spec paths, or an "
                         "object with a `controllers` list; got %s" % type(_entries).__name__)
    for _e in _entries:
        # An entry is a path, or an inline spec. A path is resolved and READ here rather than
        # transcribed, so the installed controller is byte-for-byte the one that was accepted.
        if isinstance(_e, str):
            with open(_e) as _efh:
                _espec = json.load(_efh)
            _src = _e
        elif isinstance(_e, dict) and "spec" in _e:
            _espec, _src = _e["spec"], str(_e.get("name") or "inline")
        elif isinstance(_e, dict):
            _espec, _src = _e, str(_e.get("name") or "inline")
        else:
            raise ValueError("stack entry must be a spec path or a spec object; got %r" % (_e,))
        _installed.append(_install_one(_espec, source="stack:" + _src))
    print("[controller] STACK of %d installed from %s" % (len(_installed), _STACK_PATH), flush=True)

if _PATH:
    with open(_PATH) as _fh:
        _spec = json.load(_fh)
    _installed.append(_install_one(_spec, source="candidate" if _STACK_PATH else ""))

if not _installed:
    print("[controller] ANCHOROPT_CONTROLLER_SPEC/_STACK unset -- nothing installed (control arm)",
          flush=True)
else:
    # One line a log grep can find, naming the composition that actually ran. An arm whose record
    # cannot say what was installed is the defect `stack_fingerprint` exists to close.
    print("[controller] COMPOSITION n=%d: %s"
          % (len(_installed), " | ".join("%s@%s" % (n, l) for n, l, _ in _installed)), flush=True)
