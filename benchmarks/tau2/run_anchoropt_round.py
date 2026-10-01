#!/usr/bin/env python3
"""One AnchorOpt round on tau-bench, as explicit phases that each read and write files in one arm dir.

  The round (core's search):
    baseline   run the FROZEN INCUMBENT (P0) on a split and record its trajectories  -> baseline.json
    propose    mine the residual, localize, expand Phi, ground actions (+ teacher)     -> arm_manifest.json
               NAMES NO WINNER. The outcome class is UNEVALUATED, which is not a negative result.
    evaluate   run arms PAIRED against the same frozen P0                              -> results.json
               An arm not run is OMITTED, never written as zero. Void arms are omitted too.
    select     core's argmax over J_train on what was actually measured                -> selection.json

  Checking the round's answer:
    replicate  re-run P0 K times on the same cases and portal       -> incumbent_rep_<k>.json
    rescore    score every measured arm against the P0 DISTRIBUTION -> rescore.json
    validate   P0-test and P1-test on the held-out split, K runs each -> validation_<split>.json
    refill     re-run only the void episodes of saved held-out runs, then `validate` again

WHY PHASES AND NOT ONE CALL. Measuring one tau-bench arm means hundreds of model calls, so evaluation is
a job rather than a function. `anchoropt.learning.external_evaluation` is the seam for exactly that:
propose emits every arm, evaluation happens out of band, and selection is core's argmax. Faking an
inline evaluator would collapse UNEVALUATED into NO_BENEFIT, which is how a null gets believed.

WHY REPLICATE/RESCORE/VALIDATE EXIST. `select` compares every arm to ONE incumbent run and promotes the
max of the measured arms. On tau-bench one run is not a stable reference -- qwen3.6/airline's P0 drew
16/30 while repeats of the same policy scored 21-23 -- so a train win is criterion 1 of four and means
little until checked against replicates and on held-out data. See docs/TAU2_SELFTEACH_ROUND1.md.

One arm, end to end (the commands behind docs/TAU2_SELFTEACH_ROUND1.md; see
benchmarks/tau2/README.md for the environment):

    D=benchmarks/tau2/run_anchoropt_round.py; OUT=<arm dir>
    python $D baseline  --domain airline --split train --agent-llm granite-4.1-30b \
                        --user-llm claude-sonnet-5 --max-steps 200 --concurrency 8 --out $OUT
    python $D propose   --out $OUT --teacher self
    python $D evaluate  --out $OUT --max-arms 8 --concurrency 8 --resume
    python $D select    --out $OUT
    python $D replicate --out $OUT --k 4                     # optional: P0 as a distribution
    python $D rescore   --out $OUT
    python $D validate  --out $OUT --split test --k 4 --p0-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _add_tau2_to_path() -> None:
    """The harness is referenced, not vendored: point at a checkout or fail with a clear message."""
    repo = os.environ.get("ANCHOROPT_TAU2_REPO", "")
    if repo:
        src = Path(repo) / "src"
        if not src.exists():
            raise SystemExit(f"FATAL: $ANCHOROPT_TAU2_REPO={repo} has no src/ -- not a tau2 checkout")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    try:
        import tau2  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"FATAL: tau-bench is not importable ({exc}).\n"
            "       The corpus is 714 MB and is NOT vendored. Point at a checkout with\n"
            "       $ANCHOROPT_TAU2_REPO, and run with that checkout's interpreter:\n"
            "         PYTHONPATH=<anchoropt> <tau2>/.venv/bin/python "
            "benchmarks/tau2/run_anchoropt_round.py ...") from exc


def _flds_gate() -> str:
    from tau2_fields import GATE
    return GATE


def _env_kwargs(domain: str, retrieval_variant: str | None) -> dict:
    """Per-domain environment configuration.

    banking_knowledge only: its default `alltools` variant embeds its knowledge base with
    `text-embedding-3-large`, and this gateway's allowlist contains no embedding model at all, so the
    environment cannot even be constructed. `bm25_grep` retrieves without an embedding service.
    """
    if domain != "banking_knowledge":
        return {}
    return {"retrieval_variant": retrieval_variant or "bm25_grep"}


def _preflight(agent, user, *, timeout: int = 90) -> tuple[bool, str]:
    """Probe both endpoints before spending anything. Returns (ok, detail).

    WHY THIS EXISTS. The first matrix run walked into a gateway outage and spent 90 minutes recording
    episodes that produced no outcome. A void episode is indistinguishable from a hard task at a glance,
    so the arm looked finished. One cheap probe per endpoint turns that into an immediate, honest
    failure -- and an arm that fails preflight has spent nothing.
    """
    import time

    import litellm

    # AN OUTAGE IS PERSISTENT; A SLOW MOMENT IS NOT, and the preflight only earns its place if it tells
    # them apart. A single attempt does not: a healthy minimax endpoint that answers in 1-3s was seen to
    # time out once on a transient blip, which would have condemned a good arm. Three attempts with a
    # short backoff, and `max_tokens` generous enough that a reasoning model's thinking does not eat the
    # whole budget and return empty content -- the exact trap that made two models look dead on the
    # first probe of this matrix.
    attempts, backoff = 3, 5
    # THE TOOL-CALL GATE, for a portal we serve ourselves. A wrong or missing `--tool-call-parser`
    # does NOT error: the server returns a normal completion with the call rendered as prose, no
    # tool_calls are parsed, every agent turn silently becomes a no-op, and the arm scores near zero
    # with exit 0. Qwen3.5/3.6 need `qwen3_xml` and Granite 4.1 needs `granite4`; `hermes` parses their
    # XML as nothing. Scoped to the local portal because the shared gateway is verified to parse tool
    # calls for all three models, and a mid-matrix change must not alter the path pending arms already take.
    tool_probe = [{"type": "function", "function": {
        "name": "get_reservation_details", "description": "Look up a reservation by its id.",
        "parameters": {"type": "object", "properties": {"reservation_id": {"type": "string"}},
                       "required": ["reservation_id"]}}}]
    if str(getattr(agent, "provider", "")) == "vllm":
        args = dict(agent.llm_args)
        args["timeout"] = min(int(args.get("timeout") or timeout), timeout)
        args.pop("temperature", None)
        parsed = False
        for attempt in range(attempts):
            try:
                r = litellm.completion(
                    model=agent.model, tools=tool_probe, max_tokens=1024, num_retries=0,
                    messages=[{"role": "user",
                               "content": "Look up reservation ABC123 using the available tool."}],
                    **args)
                if (r.choices[0].message.tool_calls or ()):
                    parsed = True
                    break
            except Exception as exc:
                if attempt == attempts - 1:
                    return False, (f"agent endpoint {agent.label} (vllm) tool probe failed: "
                                   f"{type(exc).__name__}: {exc}"[:300])
            if attempt < attempts - 1:
                time.sleep(backoff)
        if not parsed:
            return False, (
                f"agent endpoint {agent.label} (vllm) answered but parsed NO tool call in "
                f"{attempts} probes. This never raises at run time: every agent turn would become a "
                f"no-op and the arm would score near zero with exit 0. Check --tool-call-parser "
                f"(qwen3_xml for Qwen3.5/3.6, granite4 for Granite 4.1).")

    for role, ep in (("agent", agent), ("user", user)):
        args = dict(ep.llm_args)
        args["timeout"] = min(int(args.get("timeout") or timeout), timeout)
        args.pop("temperature", None)
        last = ""
        for attempt in range(attempts):
            try:
                r = litellm.completion(
                    model=ep.model, messages=[{"role": "user", "content": "Reply READY."}],
                    max_tokens=1024, num_retries=0, **args)
                msg = r.choices[0].message
                if getattr(msg, "content", None) or getattr(msg, "reasoning_content", None):
                    last = ""
                    break
                last = "answered with no content at all"
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"[:200]
            if attempt < attempts - 1:
                time.sleep(backoff)
        if last:
            return False, (f"{role} endpoint {ep.label} failed {attempts} probes over "
                           f"~{backoff * (attempts - 1)}s -- {last}")
    return True, f"both endpoints answered (agent {agent.label}, user {user.label})"


W = 98


def rule(title: str = "") -> None:
    print("=" * W if not title else f"\n{'=' * W}\n{title}\n{'=' * W}")


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    print(f"  wrote {path}")


def _read(path: Path):
    if not path.exists():
        raise SystemExit(f"FATAL: {path} is missing -- run the earlier phase first")
    return json.loads(path.read_text())


class _RecordedRun:
    """A saved baseline (`baseline.json`), in the shape the residual miner reads."""

    def __init__(self, blob):
        r = blob["run"]
        self.reward = {k: float(v) for k, v in r["reward"].items()}
        self.solved = {k: bool(v) for k, v in r["solved"].items()}
        self.termination = dict(r.get("termination") or {})
        self.events = {k: list(v) for k, v in (blob.get("events") or {}).items()}

    def failing(self):
        return sorted(c for c, ok in self.solved.items() if not ok)


# ================================================================================================
# baseline -- the frozen incumbent
# ================================================================================================
def cmd_baseline(a) -> int:
    _add_tau2_to_path()
    import tau2_episodes as E
    from tau2_runtime import ADAPTER

    import tau2_models as MODELS

    out = Path(a.out)
    tasks = E.load_tasks(a.domain, limit=(a.limit or None), task_ids=a.task_ids or (),
                         split=a.split)
    if not tasks:
        raise SystemExit(f"FATAL: no tasks selected for domain {a.domain!r}")

    agent = MODELS.agent_endpoint(a.agent_llm, timeout=a.timeout)
    user = MODELS.gateway_endpoint(a.user_llm, timeout=a.timeout)
    env_kwargs = _env_kwargs(a.domain, a.retrieval_variant)

    if not a.no_preflight:
        ok, detail = _preflight(agent, user)
        print(f"  preflight   : {detail}")
        if not ok:
            print("\n  FATAL: refusing to start. Every episode would fail at the harness and be\n"
                  "         recorded as an unmeasured void, which is not a result.")
            _write(out / "preflight_failed.json", {"detail": detail, "domain": a.domain,
                                                  "agent_llm": a.agent_llm})
            return 4

    rule(f"BASELINE -- the frozen incumbent P0 on {a.domain}, {len(tasks)} task(s)")
    print(f"  agent model : {a.agent_llm}  ({agent.provider}: {agent.model})")
    print(f"  user model  : {a.user_llm}  ({user.provider})")
    if env_kwargs:
        print(f"  env_kwargs  : {env_kwargs}")
    print(f"  split       : {a.split or '(none: sorted prefix)'}  -> {len(tasks)} task(s)")
    print(f"  tasks       : {[t.id for t in tasks]}")

    done = {"n": 0}

    def tick(cid, reward):
        done["n"] += 1
        print(f"    [{done['n']}/{len(tasks)}] {cid}: reward={reward:.3f}", flush=True)

    run = E.run_corpus(None, domain=a.domain, tasks=tasks, agent_llm=agent.model,
                       user_llm=user.model, max_steps=a.max_steps,
                       max_concurrency=a.concurrency, label="P0", on_episode=tick,
                       agent_llm_args=dict(agent.llm_args), user_llm_args=dict(user.llm_args),
                       env_kwargs=env_kwargs, sim_timeout=a.sim_timeout)

    print(f"\n  P0: {run.n_solved}/{run.n} solved   mean reward {run.mean_reward:.4f}")
    print(f"  residual: {len(run.failing())} failing  {run.failing()}")
    if run.truncated:
        tot = sum(run.truncated.values())
        print(f"  truncated : {tot}/{run.n} episodes ended prematurely {run.truncated}")
        if tot >= 0.5 * run.n:
            print(f"            WARNING: {100*tot//run.n}% of episodes never finished. Each scores 0 "
                  f"by construction,\n            so this baseline measures the budget as much as the "
                  f"model. Raise --max-steps / --sim-timeout.")
    # THE INCUMBENT DEFINES THE ROUND'S CASE SET, and it must be one every arm can be measured on.
    # A void episode is not part of it. Two different situations, and conflating them cost real work
    # in both directions: an outage that voids everything must stop the round, but a shared service
    # dropping one connection in 25 must NOT -- an earlier all-or-nothing rule discarded three arms
    # that had completed 21-24 episodes each.
    scored = list(run.scored_cases)
    void_ids = sorted(set(run.solved) - set(scored))
    if run.harness_errors:
        frac = run.harness_errors / max(1, run.n)
        print(f"  void      : {run.harness_errors}/{run.n} episodes produced no outcome "
              f"({frac:.0%})  {void_ids}")
        if frac > E.VOID_TOLERANCE or not scored:
            print(f"\n  FATAL: too much of the baseline is unmeasured (>{E.VOID_TOLERANCE:.0%}).")
            print("         This is NOT a 0-score incumbent -- it is an unmeasured one, and every arm\n"
                  "         compared against it would inherit the contamination. Fix the endpoint and\n"
                  "         re-run. Raise ANCHOROPT_TAU2_VOID_TOLERANCE only if you mean it.")
            _write(out / "baseline_void.json", {"harness_errors": run.harness_errors, "n": run.n,
                                               "void_task_ids": void_ids,
                                               "termination": run.termination})
            return 3
        print(f"            within tolerance: the round runs on the {len(scored)} SCORED case(s), and "
              f"every arm is\n            measured on exactly that set. The denominator is "
              f"{len(scored)}, not {run.n}.")

    # The catalog is recorded WITH the round so grounding is reproducible from the artifact alone.
    catalog = E.tool_catalog(E.make_env(a.domain, env_kwargs))
    _write(out / "baseline.json", {
        "domain": a.domain, "agent_llm": a.agent_llm, "user_llm": a.user_llm,
        "agent_model_id": agent.model, "agent_provider": agent.provider,
        "user_model_id": user.model, "env_kwargs": env_kwargs,
        "max_steps": a.max_steps, "timeout": a.timeout, "sim_timeout": a.sim_timeout,
        "split": a.split,
        "task_ids": [str(t.id) for t in tasks],
        # THE ROUND'S CASE SET. `evaluate` loads these, not `task_ids`, so every arm faces exactly the
        # cases the incumbent was scored on.
        "scored_task_ids": scored, "void_task_ids": void_ids,
        "run": run.to_json(), "events": run.events, "tool_catalog": catalog,
        "adapter": ADAPTER.name, "capability_audit": ADAPTER.capability_audit(),
    })
    return 0


# ================================================================================================
# propose -- mine, localize, expand, ground. NAME NO WINNER.
# ================================================================================================
def cmd_propose(a) -> int:
    import tau2_attribution as ATTR
    from anchoropt.learning.external_evaluation import arm_manifest
    from anchoropt.learning.structured_search import localize, optimize_residual
    from anchoropt.learning.search_state import REALIZABLE_UNMEASURED
    from tau2_runtime import ADAPTER

    out = Path(a.out)
    base = _read(out / "baseline.json")
    ADAPTER.reset_expanded_signals()
    ADAPTER.set_tool_catalog(base.get("tool_catalog") or {})

    run = _RecordedRun(base)
    rule("[1] RESIDUAL -- what the frozen incumbent leaves")
    diagnoses = ATTR.mine(run)
    problems = ATTR.problems(diagnoses, n_total=len(run.solved))
    print(f"  {len(run.failing())} failing case(s) -> {len(diagnoses)} diagnosis(es) "
          f"-> {len(problems)} ranked residual problem(s)")
    for p in problems:
        print(f"    R{p.rank}  support {p.support:2d}  coverage {p.coverage:.2f}  {p.key}")
    if not problems:
        _write(out / "arm_manifest.json", {"state": "NO_RESIDUAL", "n_arms": 0, "arms": []})
        print("\n  NO RESIDUAL: the incumbent solved every case. Nothing to fit an anchor to.")
        return 0

    target = problems[0] if not a.rank else problems[min(a.rank, len(problems)) - 1]
    print(f"\n  FROZEN TARGET: R{target.rank}  {target.key}")
    print(f"  cases: {list(target.case_ids)}")

    # ---- the teacher, if one was asked for. OFF BY DEFAULT: absent, the round runs on shipped eta
    # alone, which is what makes history-only vs history+teacher a measurable comparison.
    teacher_record = {"invoked": False, "reason": "no teacher configured",
                      "provenance": "history_only"}
    ADAPTER.reset_instruction_proposals()
    if a.teacher:
        import tau2_models as MODELS
        import tau2_teacher as TEACH
        ep, provenance = MODELS.teacher_endpoint(a.teacher, agent_model=base["agent_llm"])
        if ep is None:
            print("\n  TEACHER: disabled -- history-only")
        else:
            fields = sorted(ADAPTER.synthesis_fields(
                ADAPTER.boundary_from_key(_flds_gate())) or {})
            res = TEACH.propose_instructions(
                target, endpoint=ep, provenance=provenance, boundary_fields=fields,
                known_tools=frozenset(base.get("tool_catalog") or {}), n=a.teacher_n)
            teacher_record = res.to_json()
            # THE GROUNDINGS, not just the text. `evaluate` and `select` re-run the propose phase's
            # construction in a separate process, and without these the teacher's arms would be absent
            # from the rebuilt set -- the manifest would list arms that could never be measured.
            teacher_record["groundings"] = [prop.as_grounding() for prop in res.proposals]
            print(f"\n  TEACHER: {provenance}  invoked={res.invoked}  {res.reason}")
            for prop in res.proposals:
                print(f"     + {prop.variant}: {prop.instruction[:88]}")
            for bad in res.rejected[:4]:
                print(f"     - refused: {bad['why']}")
            ADAPTER.set_instruction_proposals([p.as_grounding() for p in res.proposals])

    # Only the target problem's own trajectories. Passing the whole residual lets the search answer
    # about a different, easier problem.
    events = [e for cid in target.case_ids for e in (run.events.get(cid) or ())]
    states = [dict(e) for e in events]
    print(f"  events: {len(events)}  states: {len(states)}")

    rule("[2] WHERE -- backward localization over the boundaries the trajectory CONTAINS")
    ordered = localize(events, runtime=ADAPTER)
    for i, b in enumerate(ordered):
        n_here = sum(1 for s in states if ADAPTER.boundary_key(s) == b.key)
        print(f"  {i}: {b.key:22s} {ADAPTER.label_for(b.key):58s} states={n_here}")
    print("  the search starts at the LATEST boundary and moves earlier only on exhaustion")

    # CYCLE 2, and why it exists. Signal expansion is driven by MEASURED exhaustion: core widens Phi at
    # a boundary only once the arms it could already express have been tried and found wanting. With
    # `evaluate=None` nothing is ever exhausted, so a first propose pass legitimately reports
    # REALIZABLE_UNMEASURED without expanding -- that is the schedule working, not a gap.
    #
    # To continue the schedule without re-running anything, `--results` replays the PREVIOUS cycle's
    # real measurements: a measured arm returns its recorded ThetaResult (so `improves` can reject it
    # and the boundary exhausts), and an unmeasured one returns None (so it is not scored). No new
    # episodes are run here and no measurement is invented.
    replay = None
    if a.results:
        from anchoropt.learning.external_evaluation import ExternalEvaluation
        replay = ExternalEvaluation.from_json(Path(a.results))
        print(f"  replaying {len(replay.results)} recorded measurement(s) from {a.results}")
    rule("[3-4] WHAT expansion and HOW grounding"
         + ("" if replay is None else " -- CONTINUED from recorded measurements"))
    outcome = optimize_residual(target, runtime=ADAPTER, host=ADAPTER.HOST,
                               events=events, states=states, evaluate=replay,
                               improves=(None if replay is None
                                         else (lambda res: res is not None and res.net > 0)))
    for att in outcome.attempts:
        print(f"  {att.boundary:22s} {att.state:24s} built={att.candidates_built:3d} "
              f"evaluated={att.candidates_evaluated:3d} -> next {att.coordinate_changed}")
    if outcome.signals_installed:
        print(f"\n  Phi EXPANDED ({len(outcome.signals_installed)} installed): "
              f"{list(outcome.signals_installed)[:10]}"
              + (" ..." if len(outcome.signals_installed) > 10 else ""))
    print(f"\n  moves_earlier = {outcome.moves_earlier}   state = {outcome.state}   "
          f"candidates = {len(outcome.candidates)}")

    arms = list(outcome.candidates)
    if not arms:
        _write(out / "arm_manifest.json", {"state": outcome.state, "n_arms": 0, "arms": [],
                                          "residual_key": target.key})
        print(f"\n  NO ARM BUILT. state={outcome.state} -- read it as a structural finding, not a null.")
        return 0

    rule("[5] ARM MANIFEST -- every arm emitted, no winner named")
    manifest = arm_manifest(arms, incumbent_id="P0", incumbent_token=str(out / "baseline.json"))
    manifest["state"] = REALIZABLE_UNMEASURED
    manifest["residual_key"] = target.key
    manifest["residual_case_ids"] = list(target.case_ids)
    manifest["signals_installed"] = list(outcome.signals_installed)
    manifest["replayed_results"] = (str(a.results) if a.results else None)
    manifest["teacher"] = teacher_record
    manifest["agent_llm"] = base.get("agent_llm")
    manifest["domain"] = base.get("domain")
    manifest["moves_earlier"] = outcome.moves_earlier
    # The predicate each arm installs, so `evaluate` runs the signal CORE installed rather than a
    # re-derived guess.
    fires = {}
    for arm in arms:
        pred = ADAPTER.expanded.get(arm.signal)
        n_fire = 0
        for s in states:
            try:
                n_fire += bool(pred(s)) if pred is not None else bool(
                    ADAPTER.evaluate_signal(arm.signal, s))
            except Exception:
                pass
        fires[arm.label] = n_fire
    for row in manifest["arms"]:
        row["fires_on_states"] = fires.get(row["arm_label"], 0)
        row["total_states"] = len(states)
    print(f"  {len(arms)} arm(s), state={manifest['state']}, no selected_arm key: "
          f"{'selected_arm' not in manifest}")
    by_cell = {}
    for row in manifest["arms"]:
        by_cell[f"{row['boundary']}/{row['action']}"] = by_cell.get(
            f"{row['boundary']}/{row['action']}", 0) + 1
    for cell, n in sorted(by_cell.items()):
        print(f"    {cell:42s} {n:3d} arm(s)")
    print("\n  UNEVALUATED: arms are built and grounded and NOBODY RAN THEM. That is not a "
          "negative result.")
    _write(out / "arm_manifest.json", manifest)
    return 0


# ================================================================================================
# evaluate -- real paired runs against the SAME frozen incumbent
# ================================================================================================
def cmd_evaluate(a) -> int:
    _add_tau2_to_path()
    import tau2_episodes as E
    from tau2_mechanism import Controller
    from tau2_runtime import ADAPTER

    out = Path(a.out)
    base = _read(out / "baseline.json")
    manifest = _read(out / "arm_manifest.json")
    arms = list(manifest.get("arms") or ())
    if not arms:
        raise SystemExit("FATAL: the manifest contains no arms")

    import tau2_models as MODELS

    ADAPTER.set_tool_catalog(base.get("tool_catalog") or {})
    domain = base["domain"]
    timeout = int(base.get("timeout", 600))
    agent = MODELS.agent_endpoint(base["agent_llm"], timeout=timeout)
    user = MODELS.gateway_endpoint(base["user_llm"], timeout=timeout)
    case_ids = list(base.get("scored_task_ids") or base["task_ids"])
    tasks = E.load_tasks(domain, task_ids=case_ids)
    inc_solved = {k: bool(v) for k, v in base["run"]["solved"].items() if k in set(case_ids)}
    if len(case_ids) < len(base["task_ids"]):
        print(f"  NOTE: the incumbent scored {len(case_ids)} of {len(base['task_ids'])} cases; arms are "
              f"measured on those {len(case_ids)}.")

    # THE PREDICATE THAT RUNS MUST BE THE ONE CORE INSTALLED, not a re-derived guess. Core owns no
    # name->predicate rebuilder (and adding one would be a core change this port does not need), so the
    # propose phase's construction is replayed in-process: `optimize_residual` reinstalls every expanded
    # signal into the adapter, and the predicate is then read out of `ADAPTER.expanded` by name.
    # Deterministic given the same baseline.json, which is what makes propose and evaluate agree.
    _rebuild_arms(out, base, manifest, results=manifest.get("replayed_results"))
    todo = arms[:a.max_arms] if a.max_arms else arms

    rule(f"EVALUATE -- {len(todo)} of {len(arms)} arm(s), each PAIRED against the frozen P0")
    print(f"  incumbent : {base['run']['n_solved']}/{base['run']['n']} solved "
          f"(token {manifest.get('incumbent_token', '')})")
    print("  NOTE: an arm not run is OMITTED from results.json, never written as zero.\n")

    # RESUME. Measuring an arm costs a few hundred model calls, so an interrupted or extended round
    # must not re-run what it already measured. Existing rows are carried forward verbatim: they were
    # measured against this same frozen incumbent, which the token check confirms.
    results: list[dict] = []
    voided: list[dict] = []
    consecutive_void = 0
    aborted = False
    already: set[str] = set()
    prior_path = out / a.results_name
    if a.resume and prior_path.exists():
        prior = json.loads(prior_path.read_text())
        results = list(prior.get("results") or ())
        already = {str(r.get("arm_label")) for r in results}
        voided = list(prior.get("voided") or ())
        print(f"  resuming: {len(already)} arm(s) already measured, carried forward\n")

    def _flush_results() -> None:
        """Persist the measurements so far.

        WRITTEN AFTER EVERY ARM, not once at the end. Per-arm runs are already durable in
        `arm_NN.json`, but the round's manifest was written only after the whole loop -- so a killed or
        crashed evaluate discarded every measurement it had actually made, and `select` and `--resume`
        both read this file. One arm had already measured a clean net +1 when this was noticed.
        """
        _write(out / a.results_name, {"incumbent_id": "P0",
                                     "incumbent_token": manifest.get("incumbent_token", ""),
                                     "n_arms_in_manifest": len(arms),
                                     "n_evaluated": len(results),
                                     "n_voided": len(voided),
                                     "voided": voided, "aborted_on_outage": aborted,
                                     "results": results})

    for i, row in enumerate(todo, start=1):
        label = row["arm_label"]
        if label in already:
            print(f"  [{i}/{len(todo)}] already measured, skipping: {label}")
            continue
        pred = _predicate_for(row["signal"])
        if pred is None:
            print(f"  [{i}/{len(todo)}] SKIP {label}: predicate could not be rebuilt -- OMITTED")
            continue
        controller = Controller(boundary=ADAPTER.boundary_key({"boundary": row["boundary"]}),
                               action=row["action"], predicate=pred, eta=dict(row.get("eta") or {}),
                               label=label, variant=str(row.get("variant") or ""))
        print(f"  [{i}/{len(todo)}] {label}")
        print(f"        eta={dict(row.get('eta') or {})}")
        run = E.run_corpus(controller, domain=domain, tasks=tasks, agent_llm=agent.model,
                           user_llm=user.model, max_steps=base["max_steps"],
                           max_concurrency=a.concurrency, label=label,
                           agent_llm_args=dict(agent.llm_args), user_llm_args=dict(user.llm_args),
                           env_kwargs=base.get("env_kwargs") or None,
                           sim_timeout=float(base.get("sim_timeout") or 3600.0))
        pr = E.paired(run, _AsRun(inc_solved))
        if run.harness_errors and run.harness_errors / max(1, run.n) <= E.VOID_TOLERANCE:
            # A few void cases do not void the arm. They are EXCLUDED from the comparison (`paired`
            # does that) and the row records its own smaller `n`, so nothing is fabricated and the
            # measurement is not thrown away either.
            print(f"        {run.harness_errors}/{run.n} void case(s) excluded; measured on "
                  f"{pr['n']}", flush=True)
        elif run.harness_errors:
            # THE CARDINAL RULE: an arm you did not evaluate must be OMITTED, never reported as zero.
            # A harness-errored episode produced no outcome, so scoring it as unsolved would count an
            # infrastructure failure as the intervention's effect -- and against a clean incumbent it
            # reads as a loss on every case the incumbent solved. Recorded as void, with the count, so
            # the round classifies BUDGET_EXHAUSTED on a missing measurement rather than banking a
            # fabricated one.
            print(f"        VOID: {run.harness_errors}/{run.n} episodes failed at the harness "
                  f"(no outcome). OMITTED from results.json.", flush=True)
            voided.append({"arm_label": label, "harness_errors": run.harness_errors, "n": run.n,
                           "reason": "episodes did not produce a scored outcome"})
            E.save_run(run, out / f"arm_{i:02d}.json")
            _flush_results()
            # OUTAGE ABORT. Two arms in a row where NOTHING ran is an endpoint outage, not two bad
            # arms. Continuing burns the rest of the budget into voids -- which is exactly how one
            # round recorded six void measurements and still named a winner. Stop, keep what was
            # genuinely measured, and let the arm be requeued.
            consecutive_void = consecutive_void + 1 if run.harness_errors == run.n else 0
            if consecutive_void >= 2:
                print(f"\n  ABORTING: {consecutive_void} consecutive arms ran NOTHING -- the endpoint "
                      f"is down.\n            {len(results)} genuine measurement(s) are kept; the rest "
                      f"stay OPEN for a re-run.")
                aborted = True
                break
            continue
        consecutive_void = 0
        print(f"        {run.n_solved}/{run.n} solved   net {pr['net']:+d}  "
              f"+{len(pr['gains'])}/-{len(pr['losses'])}  "
              f"{pr['accuracy_delta_pp']:+.2f}pp  executed={run.interventions_executed} "
              f"cases_fired={len(run.cases_fired)}", flush=True)
        results.append({
            "arm_label": label, "gains": list(pr["gains"]), "losses": list(pr["losses"]),
            "firings": run.interventions_executed, "cases_fired": len(run.cases_fired),
            "n": pr["n"], "interventions_executed": run.interventions_executed,
            "accuracy_delta_pp": pr["accuracy_delta_pp"],
            "denominator_identical": pr["denominator_identical"],
            "complete": pr["complete"],
            "detail": f"{run.n_solved}/{run.n} vs incumbent {pr['incumbent_solved']}/{pr['n']}",
        })
        E.save_run(run, out / f"arm_{i:02d}.json")
        _flush_results()

    _flush_results()
    if aborted:
        print("\n  This round is INCOMPLETE because the endpoint failed, not because its arms did "
              "not help.\n  Re-run `evaluate --resume` once the endpoint is back.")
    if voided:
        print(f"\n  {len(voided)} arm(s) VOID -- their episodes did not run, so they are OMITTED "
              f"rather than scored. They remain OPEN and should be re-measured:")
        for v in voided:
            print(f"     {v['harness_errors']:2d}/{v['n']} failed   {v['arm_label'][:78]}")
    if len(results) < len(arms):
        print(f"\n  {len(arms) - len(results)} arm(s) left UNMEASURED and OMITTED. If the round is "
              f"selected now it is BUDGET_EXHAUSTED, not NO_BENEFIT.")
    return 0


class _AsRun:
    """The frozen incumbent's solved map, in the shape `paired` reads."""

    def __init__(self, solved):
        self.solved = dict(solved)
        self.firings = {}

    @property
    def interventions_executed(self):
        return 0

    @property
    def cases_fired(self):
        return []


# ================================================================================================
# select -- core's own argmax over J_train on what was MEASURED
# ================================================================================================
def cmd_select(a) -> int:
    from anchoropt.learning.external_evaluation import (
        ExternalEvaluation, select_on_measurement,
    )
    from tau2_runtime import ADAPTER

    out = Path(a.out)
    base = _read(out / "baseline.json")
    manifest = _read(out / "arm_manifest.json")
    ADAPTER.reset_expanded_signals()
    ADAPTER.set_tool_catalog(base.get("tool_catalog") or {})

    # Rebuild the SAME arms this round emitted, then hand core the measurements. The driver must not
    # construct proposals or call the optimizer itself.
    arms = _rebuild_arms(out, base, manifest, results=manifest.get("replayed_results"))
    if not arms:
        raise SystemExit("FATAL: could not rebuild the manifest's arms")

    ev = ExternalEvaluation.from_json(out / a.results_name, budget=a.eval_budget)
    rule("SELECT -- core's optimizer over J_train, on real measurements only")
    sel = select_on_measurement(arms, runtime=ADAPTER, host=ADAPTER.HOST, incumbent_id="P0",
                                incumbent_token=manifest.get("incumbent_token", ""),
                                evaluation=ev)
    print(f"  arms rebuilt        : {len(arms)}")
    print(f"  measurements served : {len(ev.served)}")
    print(f"  measurements missing: {len(ev.missing)}")
    print(f"  outcome class       : {sel.outcome_class}")
    print(f"  evaluations counted : {sel.evaluations_completed}")
    print(f"  unevaluated         : {len(sel.unevaluated)}")
    # `sel.winner` is core's ARGMAX AMONG MEASURED ARMS, and it is populated whatever the sign. It is
    # NOT a promotion: `sel.improved` (train net > 0) is criterion 1 of four, and `sel.accepted` is
    # always False here because dev and held-out validation have not run. An earlier version of this
    # driver printed "SELECTED" for a net -2 arm on a BUDGET_EXHAUSTED round, which is exactly the
    # collapse the outcome classes exist to prevent -- so the two are reported separately.
    if sel.winner is not None:
        w, r = sel.winner, sel.winner_result
        # A WINNER THAT NEVER FIRED IS NOT A REPAIR. `train_objective` orders by net first and
        # engagement second, so on a small residual a run-to-run flip can be promoted by an arm whose
        # controller executed zero interventions -- there is nothing for the engagement tiebreak to
        # outrank when few arms were measured. Core's ordering is what it is; what this driver must not
        # do is call that a win. Reported, not silently relabelled, and the spec is still emitted so the
        # arm can be re-measured.
        zero_exec = (r is not None and int(getattr(r, "interventions_executed", 0) or 0) == 0)
        if sel.improved and zero_exec:
            headline = ("TRAIN NET POSITIVE BUT THE CONTROLLER NEVER FIRED -- the delta is not "
                        "attributable to the intervention")
        elif sel.improved:
            headline = "TRAIN WINNER (criterion 1 of 4, pending validation)"
        else:
            headline = "BEST MEASURED -- NOT AN IMPROVEMENT, do not install"
        print(f"\n  {headline}")
        print(f"     arm    : {w.label}")
        print(f"     locus  : {w.boundary.value}")
        print(f"     phi    : {w.signal}")
        print(f"     action : {w.action.value} ({w.instantiated.operator.value})")
        print(f"     eta    : {dict(w.eta)}")
        if r is not None:
            print(f"     measured: net {r.net:+d}  +{len(r.gains)}/-{len(r.losses)}  "
                  f"{r.accuracy_delta_pp:+.2f}pp  executed={r.interventions_executed}")
            if zero_exec:
                print("     WARNING: interventions_executed=0. The arm's condition never held, so "
                      "this delta is\n              run-to-run variance, not a repair. Measure more "
                      "arms before reading anything\n              into it -- especially on a small "
                      "residual, where one flip is the whole margin.")
    else:
        print("\n  NO ARM MEASURED. With nothing measured there is no winner, and the outcome class "
              "says so rather than reporting a null about arms nobody ran.")
    if sel.unevaluated:
        print(f"\n  {len(sel.unevaluated)} arm(s) still UNMEASURED -- the round is open, not settled.")

    from tau2_mechanism import Controller, controller_spec
    # The installable spec is emitted ONLY for a train win. Writing one for a measured loss invites a
    # downstream runner to install a controller the round measured as harmful.
    spec = {}
    if sel.improved and sel.winner is not None:
        spec = controller_spec(Controller(
            boundary=ADAPTER.boundary_key({"boundary": sel.winner.boundary.value}),
            action=sel.winner.action.value, predicate=lambda _s: False,
            eta=dict(sel.winner.eta), label=sel.winner.label,
            variant=str(getattr(sel.winner.instantiated, "variant", "") or "")))
        spec["signal"] = sel.winner.signal
    _write(out / "selection.json", {
        "outcome_class": sel.outcome_class,
        "improved": bool(sel.improved),
        # Kept separate from `improved` on purpose: a reader must be able to see that a positive net
        # came from an arm that never executed.
        "winner_interventions_executed": (int(getattr(sel.winner_result, "interventions_executed", 0) or 0)
                                          if sel.winner_result is not None else None),
        "winner_engaged": (bool(sel.winner_result is not None
                                and int(getattr(sel.winner_result, "interventions_executed", 0) or 0) > 0)),
        "accepted": bool(sel.accepted),
        "is_negative_result": bool(sel.is_negative_result),
        "best_measured_arm": (sel.winner.label if sel.winner is not None else None),
        "best_measured_net": (sel.winner_result.net if sel.winner_result is not None else None),
        "promoted": (sel.winner.label if (sel.improved and sel.winner is not None) else None),
        "controller_spec": spec,
        "evaluations_completed": sel.evaluations_completed,
        "unevaluated": list(sel.unevaluated),
        "served": list(ev.served), "missing": list(ev.missing),
        "report": dict(sel.report or {}),
    })
    return 0


def _rebuild_arms(out: Path, base, manifest, results: str | None = None):
    """Re-run the propose phase's construction so `select` optimizes over the arms it emitted.

    `results` must match what the propose phase was given, or a cycle-2 manifest's expanded signals are
    not reinstalled and the rebuilt arms differ from the emitted ones.
    """
    import tau2_attribution as ATTR
    from anchoropt.learning.structured_search import optimize_residual
    from tau2_runtime import ADAPTER

    run = _RecordedRun(base)
    # REINSTALL THE TEACHER'S ETA before rebuilding. The proposals are read back from the manifest
    # rather than re-requested: asking the teacher again could return different text, and the arms
    # would then not be the arms that were emitted.
    ADAPTER.set_instruction_proposals((manifest.get("teacher") or {}).get("groundings") or [])
    diagnoses = ATTR.mine(run)
    problems = ATTR.problems(diagnoses, n_total=len(run.solved))
    key = manifest.get("residual_key")
    target = next((p for p in problems if p.key == key), problems[0] if problems else None)
    if target is None:
        return []
    events = [e for cid in target.case_ids for e in (run.events.get(cid) or ())]
    states = [dict(e) for e in events]
    replay = None
    if results:
        from anchoropt.learning.external_evaluation import ExternalEvaluation
        replay = ExternalEvaluation.from_json(Path(results))
    outcome = optimize_residual(target, runtime=ADAPTER, host=ADAPTER.HOST, events=events,
                               states=states, evaluate=replay,
                               improves=(None if replay is None
                                         else (lambda res: res is not None and res.net > 0)))
    return list(outcome.candidates)


def _predicate_for(signal: str):
    """The predicate for `signal` exactly as core installed it, or None if it cannot be rebuilt.

    Call after `_rebuild_arms`, which replays the propose phase and so reinstalls every expanded signal
    into the adapter. A shipped signal is evaluated by the adapter directly.
    """
    from tau2_runtime import ADAPTER

    pred = ADAPTER.expanded.get(signal)
    if pred is not None:
        return pred
    if signal in set(ADAPTER.declared_signals()):
        return lambda st, _s=signal: bool(ADAPTER.evaluate_signal(_s, st))
    return None


def _controller_for(out: Path, base, manifest, label: str):
    """Rebuild the installable controller for one emitted arm, by its manifest label.

    The single place a phase turns an arm label back into something that runs, so `evaluate`,
    `validate` and `refill` cannot drift apart in how they rebuild the same controller.
    """
    from tau2_mechanism import Controller
    from tau2_runtime import ADAPTER

    row = next((r for r in manifest.get("arms") or () if r["arm_label"] == label), None)
    if row is None:
        raise SystemExit(f"FATAL: {label!r} is not in the manifest, so its controller cannot be rebuilt")
    _rebuild_arms(out, base, manifest, results=manifest.get("replayed_results"))
    pred = _predicate_for(row["signal"])
    if pred is None:
        raise SystemExit(f"FATAL: signal {row['signal']!r} could not be rebuilt")
    return Controller(boundary=ADAPTER.boundary_key({"boundary": row["boundary"]}),
                      action=row["action"], predicate=pred, eta=dict(row.get("eta") or {}),
                      label=label, variant=str(row.get("variant") or ""))



# ================================================================================================
# validate -- the promoted controller on the HELD-OUT split
# ================================================================================================
def cmd_validate(a) -> int:
    """P0-test and P1-test: the incumbent and the promoted controller on the HELD-OUT split, K times each.

    WHY K RUNS EACH. A single incumbent draw is not a stable reference on tau-bench. For qwen3.6/airline
    the train incumbent drew 16/30 while four repeats of the same policy scored far higher, and every
    measured arm looked positive against the low draw. The held-out check inherits the same problem if
    it compares one P1 run to one P0 run, so both sides are replicated and compared by their MEANS.

    WHAT THIS SETTLES, in docs/ACCEPTANCE_RULE.md's vocabulary. `select` gave criterion 1 (positive
    paired net on train). This addresses criterion 2 -- NO AGGREGATE REGRESSION on an independent split,
    i.e. mean P1 - mean P0 >= 0 (no aggregate regression, net >= 0) -- and gathers criterion 3 evidence:
    are the gains on cases where the controller actually fired? It does NOT establish acceptance.

    `--p0-only` runs P0-test even when there is no controller to test (no train winner, or a winner that
    never fired), so every arm gets a held-out incumbent number.
    """
    import statistics as st

    _add_tau2_to_path()
    import tau2_episodes as E
    import tau2_models as MODELS
    from tau2_runtime import ADAPTER

    out = Path(a.out)
    base = _read(out / "baseline.json")
    manifest = _read(out / "arm_manifest.json") if (out / "arm_manifest.json").exists() else {}
    sel = _read(out / "selection.json") if (out / "selection.json").exists() else {}

    test_p1, why_not = True, ""
    if not sel.get("improved"):
        test_p1, why_not = False, f"no train winner (outcome {sel.get('outcome_class') or 'none'})"
    elif not sel.get("winner_engaged", True):
        test_p1, why_not = False, "the train winner executed ZERO interventions"
    if not test_p1 and not a.p0_only:
        print(f"  nothing to validate: {why_not}. Criterion 2 does not arise.")
        return 5 if "ZERO" in why_not else 0

    domain = base["domain"]
    # A DEAD REQUEST IS ABANDONED QUICKLY. tau-bench retries 3x at the per-call timeout, so one hung call
    # held a whole run for ~50 minutes. This changes only how fast a dead request gives up -- it becomes a
    # void episode, excluded from scoring -- not how the model behaves on a live one.
    agent = MODELS.agent_endpoint(base["agent_llm"], timeout=a.call_timeout,
                                 portal=(a.portal or base.get("agent_provider") or ""))
    user = MODELS.gateway_endpoint(base["user_llm"], timeout=a.call_timeout)
    ADAPTER.set_tool_catalog(base.get("tool_catalog") or {})
    if agent.provider != (base.get("agent_provider") or agent.provider):
        # Informational only: P0-test and P1-test below always run on the SAME portal as each other, so
        # the held-out comparison is not cross-portal. The controller was fitted on the other portal.
        print(f"  NOTE: the train round ran on portal {base.get('agent_provider')!r}; both P0-test and "
              f"P1-test run on {agent.provider!r}.")

    tasks = E.load_tasks(domain, split=a.split)
    if not tasks:
        raise SystemExit(f"FATAL: domain {domain!r} has no {a.split!r} split")

    controller, label = None, None
    if test_p1:
        label = sel.get("promoted") or sel.get("best_measured_arm")
        controller = _controller_for(out, base, manifest, label)

    def _args(ep):
        d = dict(ep.llm_args)
        d["num_retries"] = a.retries
        return d

    common = dict(domain=domain, tasks=tasks, agent_llm=agent.model, user_llm=user.model,
                  max_steps=int(base.get("max_steps", 200)), max_concurrency=a.concurrency,
                  agent_llm_args=_args(agent), user_llm_args=_args(user),
                  env_kwargs=base.get("env_kwargs") or None,
                  sim_timeout=float(base.get("sim_timeout") or 3600.0))

    rule(f"VALIDATE on {a.split} x{a.k} -- {base['agent_llm']} / {domain}")
    print(f"  {len(tasks)} held-out task(s)   agent via {agent.provider}   call timeout {a.call_timeout}s, "
          f"{a.retries} retr{'y' if a.retries == 1 else 'ies'}")
    print(f"  P1 controller: {label if test_p1 else '(none -- ' + why_not + ')'}\n")

    def _load(path):
        blob = json.loads(path.read_text())
        r = E.RunResult(label=blob.get("label", ""))
        r.reward = {k: float(v) for k, v in blob["reward"].items()}
        r.solved = {k: bool(v) for k, v in blob["solved"].items()}
        r.termination = dict(blob.get("termination") or {})
        r.firings = dict(blob.get("firings") or {})
        return r

    def _runs(kind, ctrl):
        got = []
        for k in range(1, a.k + 1):
            f = out / f"heldout_{kind}_{a.split}_rep{k}.json"
            legacy = out / f"heldout_{'incumbent' if kind == 'p0' else 'arm'}_{a.split}.json"
            if not f.exists() and k == 1 and legacy.exists():
                f.write_text(legacy.read_text())      # the earlier single-draw run is replicate 1
            if f.exists():
                r = _load(f)
                print(f"  {kind.upper()}-test rep {k}: cached  {r.n_solved}/{r.n}  void={r.harness_errors}")
            else:
                r = E.run_corpus(ctrl, label=f"{kind}_{a.split}_rep{k}", **common)
                E.save_run(r, f)
                print(f"  {kind.upper()}-test rep {k}: {r.n_solved}/{r.n}  void={r.harness_errors}"
                      + (f"  executed={r.interventions_executed} fired={len(r.cases_fired)}" if ctrl else ""),
                      flush=True)
            got.append(r)
        return got

    p0 = _runs("p0", None)
    p1 = _runs("p1", controller) if test_p1 else []

    def _scored(r):
        return {c: r.solved[c] for c in r.solved
                if not str(r.termination.get(c, "")).startswith("harness_error")}

    cases = sorted(set.intersection(*[set(_scored(r)) for r in p0 + p1]))
    p0_tot = [sum(_scored(r)[c] for c in cases) for r in p0]
    rec = {"split": a.split, "n": len(cases), "k": a.k, "agent_provider": agent.provider,
           "p0_totals": p0_tot, "p0_mean": st.mean(p0_tot),
           "p0_sd": st.stdev(p0_tot) if len(p0_tot) > 1 else 0.0,
           "p0_void": [r.harness_errors for r in p0], "arm_label": label, "tested_p1": test_p1,
           "why_not_p1": why_not, "accepted": False,
           "train_p0": f"{base['run']['n_solved']}/{len(base.get('scored_task_ids') or base['task_ids'])}",
           "train_net": sel.get("best_measured_net")}

    rule(f"RESULT on {a.split} ({len(cases)} cases scored in every run)")
    print(f"  P0-test: {p0_tot}  mean {rec['p0_mean']:.1f} / {len(cases)}  sd {rec['p0_sd']:.2f}")
    if test_p1:
        p1_tot = [sum(_scored(r)[c] for c in cases) for r in p1]
        pinc = {c: st.mean(_scored(r)[c] for r in p0) for c in cases}
        parm = {c: st.mean(_scored(r)[c] for r in p1) for c in cases}
        fired = set().union(*[set(r.cases_fired) for r in p1])
        diff = st.mean(p1_tot) - st.mean(p0_tot)
        pr = {"net": diff}
        sd = (st.stdev(p0_tot) ** 2 / len(p0_tot) + st.stdev(p1_tot) ** 2 / len(p1_tot)) ** 0.5 \
            if len(p0_tot) > 1 and len(p1_tot) > 1 else 0.0
        up = sorted(c for c in cases if parm[c] > pinc[c])
        down = sorted(c for c in cases if parm[c] < pinc[c])
        att, unatt, coll = sorted(set(up) & fired), sorted(set(up) - fired), sorted(set(down) & fired)
        ex = [r.interventions_executed for r in p1]
        print(f"  P1-test: {p1_tot}  mean {st.mean(p1_tot):.1f} / {len(cases)}  "
              f"sd {st.stdev(p1_tot) if len(p1_tot) > 1 else 0.0:.2f}   executed {ex}")
        print(f"  mean P1 - mean P0 = {diff:+.2f}  (se {sd:.2f}{', z ' + format(diff / sd, '+.2f') if sd else ''})")
        print(f"  criterion 2 (no aggregate regression, net >= 0) : {'PASS' if pr['net'] >= 0 else 'FAIL'}")
        print(f"  criterion 3 evidence: cases improved where it fired {len(att)}, where it never fired "
              f"{len(unatt)}, cases worsened where it fired {len(coll)}")
        print(f"                        fired on {len(fired)}/{len(cases)} cases in at least one run"
              + ("  (fires everywhere: attribution is uninformative)" if len(fired) >= len(cases) else ""))
        rec.update({"p1_totals": p1_tot, "p1_mean": st.mean(p1_tot),
                    "p1_sd": st.stdev(p1_tot) if len(p1_tot) > 1 else 0.0,
                    "p1_void": [r.harness_errors for r in p1], "p1_executed": ex,
                    "mean_diff": diff, "se": sd, "z": (diff / sd) if sd else None,
                    "criterion_2_no_aggregate_regression": bool(pr["net"] >= 0),
                    "cases_fired_any_run": sorted(fired),
                    "criterion_3_gains_attributable": att,
                    "criterion_3_gains_not_attributable": unatt,
                    "criterion_3_collateral_losses": coll})
    rec["p0_pass_hat_k"] = pass_hat_k([_scored(r) for r in p0], cases)
    line = "  pass^k   " + "  ".join(f"k={k}" .ljust(12) for k in rec["p0_pass_hat_k"])
    print("\n" + line)
    print("  P0-test  " + "  ".join(f"{v:.3f}".ljust(12) for v in rec["p0_pass_hat_k"].values()))
    if test_p1:
        rec["p1_pass_hat_k"] = pass_hat_k([_scored(r) for r in p1], cases)
        print("  P1-test  " + "  ".join(f"{v:.3f}".ljust(12) for v in rec["p1_pass_hat_k"].values()))
    print("\n  Not acceptance: criterion 3 in full and criterion 4 (safety/invariants) are not decided here.")
    _write(out / f"validation_{a.split}.json", rec)
    return 0


# ================================================================================================
# replicate / rescore -- the incumbent is a DISTRIBUTION, not one draw
# ================================================================================================
def cmd_replicate(a) -> int:
    """Re-run the frozen incumbent K more times on EXACTLY the round's case set and portal.

    WHY. Every arm is paired against ONE incumbent run, and the winner is the max of the measured arms.
    On tau-bench that is not a stable reference: for qwen3.6/airline the incumbent scored 16/30 while
    five repeats of the same policy (the unfired cases of arms that fired on 3 of 30) averaged 76%, and
    15 of 28 cases flipped outcome under an identical policy. Against the low draw all 8 arms looked
    positive. This measures how much the reference itself moves.
    """
    _add_tau2_to_path()
    import tau2_episodes as E
    import tau2_models as MODELS

    out = Path(a.out)
    base = _read(out / "baseline.json")
    timeout = int(base.get("timeout", 600))
    # THE SAME PORTAL as the incumbent it replicates; a portal change is not a replicate.
    agent = MODELS.agent_endpoint(base["agent_llm"], timeout=timeout,
                                 portal=(base.get("agent_provider") or ""))
    user = MODELS.gateway_endpoint(base["user_llm"], timeout=timeout)
    case_ids = list(base.get("scored_task_ids") or base["task_ids"])
    tasks = E.load_tasks(base["domain"], task_ids=case_ids)
    common = dict(domain=base["domain"], tasks=tasks, agent_llm=agent.model, user_llm=user.model,
                  max_steps=int(base.get("max_steps", 200)), max_concurrency=a.concurrency,
                  agent_llm_args=dict(agent.llm_args), user_llm_args=dict(user.llm_args),
                  env_kwargs=base.get("env_kwargs") or None,
                  sim_timeout=float(base.get("sim_timeout") or 3600.0))
    rule(f"REPLICATE the incumbent x{a.k} -- {base['agent_llm']} / {base['domain']} via {agent.provider}")
    for k in range(1, a.k + 1):
        f = out / f"incumbent_rep_{k}.json"
        if f.exists():
            print(f"  rep {k}: cached")
            continue
        run = E.run_corpus(None, label=f"P0_rep{k}", **common)
        E.save_run(run, f)
        print(f"  rep {k}: {run.n_solved}/{run.n}  void={run.harness_errors}", flush=True)
    return 0


def pass_hat_k(runs: list[dict], cases: list[str]) -> dict[int, float]:
    """pass^k for k = 1..len(runs), as tau-bench computes it (src/tau2/metrics/agent_metrics.py).

    Per task: C(c, k) / C(n, k), with n trials and c successes -- the probability that k trials drawn
    from the n all succeed. Averaged over tasks. `runs` are {case: solved} maps, one per independent
    trial; `cases` must be scored in every run, because tau-bench requires the same number of trials for
    every task and excludes infrastructure errors (void episodes) rather than counting them as failures.
    """
    import math

    n = len(runs)
    out = {}
    for k in range(1, n + 1):
        vals = [math.comb(sum(bool(r[c]) for r in runs), k) / math.comb(n, k) for c in cases]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out


def _replicates(out: Path, base) -> list[dict]:
    """The original incumbent plus every replicate, as {case: solved} with void cases dropped."""
    runs = []
    orig = base["run"]
    runs.append({c: bool(v) for c, v in orig["solved"].items()
                 if not str((orig.get("termination") or {}).get(c, "")).startswith("harness_error")})
    for f in sorted(out.glob("incumbent_rep_*.json")):
        r = json.loads(f.read_text())
        runs.append({c: bool(v) for c, v in r["solved"].items()
                     if not str((r.get("termination") or {}).get(c, "")).startswith("harness_error")})
    return runs


def cmd_rescore(a) -> int:
    """Score every measured arm against the incumbent DISTRIBUTION rather than its single draw.

    Two statistics, both reported, neither hidden behind the other:
      * expected net  = sum over cases of (arm solved) - P(incumbent solves that case)
      * z             = (arm total - incumbent mean total) / incumbent sd of totals
    And one blunt test: does the arm's total exceed the BEST incumbent replicate? With the winner
    chosen as the max of N arms, an arm inside the incumbent's own range is not evidence of anything.
    """
    import re as _re
    import statistics as st

    out = Path(a.out)
    base = _read(out / "baseline.json")
    cases = list(base.get("scored_task_ids") or base["task_ids"])
    runs = _replicates(out, base)
    if len(runs) < 3:
        raise SystemExit(f"FATAL: {len(runs)} incumbent run(s); run `replicate` first (need >= 3)")
    p = {c: st.mean([r[c] for r in runs if c in r]) for c in cases if any(c in r for r in runs)}
    totals = [sum(r.get(c, False) for c in cases) for r in runs]
    mean_t, sd_t = st.mean(totals), (st.stdev(totals) if len(totals) > 1 else 0.0)
    flips = sum(1 for c in p if 0.0 < p[c] < 1.0)

    rule(f"RESCORE against {len(runs)} incumbent runs -- {base['agent_llm']} / {base['domain']}")
    print(f"  incumbent totals : {totals}   mean {mean_t:.1f}  sd {sd_t:.2f}   (original draw: {totals[0]})")
    print(f"  cases that flip under the SAME policy: {flips}/{len(p)}")
    print()
    sel = json.loads((out / "selection.json").read_text()) if (out / "selection.json").exists() else {}
    promoted = sel.get("promoted") or sel.get("best_measured_arm")
    rows = []
    pat = _re.compile(r"^arm_\d+\.json$")
    for f in sorted(x for x in out.glob("arm_*.json") if pat.match(x.name)):
        r = json.loads(f.read_text())
        valid = [c for c in cases if c in p and
                 not str((r.get("termination") or {}).get(c, "")).startswith("harness_error")]
        tot = sum(bool(r["solved"].get(c)) for c in valid)
        exp_net = sum(bool(r["solved"].get(c)) - p[c] for c in valid)
        old_net = (sum(bool(r["solved"].get(c)) and not runs[0].get(c, False) for c in valid)
                   - sum((not r["solved"].get(c)) and runs[0].get(c, False) for c in valid))
        z = (tot - mean_t) / sd_t if sd_t else float("nan")
        rows.append({"arm_label": r["label"], "total": tot, "n": len(valid), "old_net": old_net,
                     "expected_net": round(exp_net, 2), "z": round(z, 2),
                     "above_best_incumbent": tot > max(totals),
                     "executed": int(r["firings"].get("interventions_executed", 0)),
                     "cases_fired": len(r["firings"].get("cases_fired") or []),
                     "promoted": r["label"] == promoted})
    print(f"  {'arm':52s} {'solved':>7s} {'old net':>7s} {'exp net':>7s} {'z':>6s} {'>best?':>6s}")
    for r in rows:
        star = " <- promoted" if r["promoted"] else ""
        print(f"  {r['arm_label'][:52]:52s} {r['total']:4d}/{r['n']:<2d} {r['old_net']:+7d} "
              f"{r['expected_net']:+7.2f} {r['z']:+6.2f} {('YES' if r['above_best_incumbent'] else 'no'):>6s}{star}")
    beats = [r for r in rows if r["above_best_incumbent"]]
    print()
    if beats:
        print(f"  {len(beats)} arm(s) scored above the incumbent's BEST replicate. Still single runs, and "
              f"still the max\n  of {len(rows)} -- worth replicating those arms before believing them.")
    else:
        print("  NO arm scored above the incumbent's best replicate. Every measured arm is inside the "
              "range the\n  unmodified pipeline produces on its own; the original nets were measured "
              "against a single draw.")
    _write(out / "rescore.json", {"incumbent_totals": totals, "incumbent_mean": mean_t,
                                  "incumbent_sd": sd_t, "flip_cases": flips, "n_cases": len(p),
                                  "arms": rows, "promoted": promoted})
    return 0



def cmd_refill(a) -> int:
    """Re-run ONLY the void episodes of saved held-out runs, on the same portal, and patch them in.

    A void episode produced no outcome (the harness failed mid-conversation), so re-running it is how the
    missing measurement is obtained -- not a second chance at a result. Every patched case is recorded
    under `refilled`, with the original error, so the run's provenance stays visible. P1 runs are refilled
    with the same controller; P0 runs with none.
    """
    _add_tau2_to_path()
    import tau2_episodes as E
    import tau2_models as MODELS
    from tau2_runtime import ADAPTER

    out = Path(a.out)
    base = _read(out / "baseline.json")
    agent = MODELS.agent_endpoint(base["agent_llm"], timeout=a.call_timeout,
                                 portal=(a.portal or base.get("agent_provider") or ""))
    user = MODELS.gateway_endpoint(base["user_llm"], timeout=a.call_timeout)
    ADAPTER.set_tool_catalog(base.get("tool_catalog") or {})

    def _args(ep):
        d = dict(ep.llm_args)
        d["num_retries"] = a.retries
        return d

    controller = None
    runs = sorted(out.glob(f"heldout_p*_{a.split}_rep*.json"))
    for f in runs:
        blob = json.loads(f.read_text())
        void = sorted(c for c, t in (blob.get("termination") or {}).items()
                      if str(t).startswith("harness_error"))
        if not void:
            continue
        is_p1 = f.name.startswith("heldout_p1_")
        if is_p1 and controller is None:
            manifest, sel = _read(out / "arm_manifest.json"), _read(out / "selection.json")
            label = sel.get("promoted") or sel.get("best_measured_arm")
            controller = _controller_for(out, base, manifest, label)
        print(f"  {f.name}: refilling void case(s) {void} ({'P1' if is_p1 else 'P0'}, portal {agent.provider})")
        r = E.run_corpus(controller if is_p1 else None, domain=base["domain"],
                         tasks=E.load_tasks(base["domain"], task_ids=void),
                         agent_llm=agent.model, user_llm=user.model,
                         max_steps=int(base.get("max_steps", 200)), max_concurrency=max(1, len(void)),
                         agent_llm_args=_args(agent), user_llm_args=_args(user),
                         env_kwargs=base.get("env_kwargs") or None,
                         sim_timeout=float(base.get("sim_timeout") or 3600.0), label="refill")
        log = blob.setdefault("refilled", [])
        for c in void:
            t = str(r.termination.get(c, ""))
            if t.startswith("harness_error") or c not in r.solved:
                print(f"    case {c}: STILL void ({t[:80]}) -- left as is")
                continue
            log.append({"case": c, "original": str(blob["termination"][c])[:200],
                        "refill_termination": t, "solved": bool(r.solved[c]), "portal": agent.provider})
            blob["solved"][c], blob["reward"][c], blob["termination"][c] = r.solved[c], r.reward[c], t
            print(f"    case {c}: {'SOLVED' if r.solved[c] else 'failed'} ({t})")
        blob["n_solved"] = sum(1 for v in blob["solved"].values() if v)
        blob["harness_errors"] = sum(1 for t in blob["termination"].values()
                                     if str(t).startswith("harness_error"))
        f.write_text(json.dumps(blob, indent=2, default=str) + "\n")
    return 0


# ================================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="run the frozen incumbent and record trajectories")
    b.add_argument("--domain", default="airline")
    b.add_argument("--limit", type=int, default=0,
                   help="subsample WITHIN the split; 0 (default) uses the whole split")
    b.add_argument("--split", default=os.environ.get("ANCHOROPT_TAU2_SPLIT", "train"),
                   help="one of tau-bench's OWN splits: train (default) / test / base. Optimizing on "
                        "train keeps test clean for validation")
    b.add_argument("--task-ids", nargs="*", default=[])
    b.add_argument("--agent-llm", default=os.environ.get("ANCHOROPT_TAU2_AGENT_LLM",
                                                        "granite-4.1-30b"),
                   help="the model under test: a HOSTED_MODELS name, a local vLLM model, or a gateway model")
    b.add_argument("--user-llm", default=os.environ.get("ANCHOROPT_TAU2_USER_LLM",
                                                       "claude-sonnet-5"),
                   help="the user simulator, held CONSTANT across the matrix")
    b.add_argument("--retrieval-variant", default=os.environ.get(
        "ANCHOROPT_TAU2_RETRIEVAL_VARIANT", "bm25_grep"),
                   help="banking_knowledge only; the default needs no embedding service")
    b.add_argument("--max-steps", type=int, default=40)
    b.add_argument("--concurrency", type=int, default=4)
    # AN EXPLICIT TIMEOUT ON BOTH SIDES. Without one LiteLLM applies its 600 s default to the user
    # simulator while the agent gets far longer, so a slow gateway path kills a task on the USER turn
    # when the agent turn would have survived. One recorded job burned four hours completing 0 of 114
    # tasks that way, every slot stuck on retry 3.
    b.add_argument("--timeout", type=int, default=600)
    b.add_argument("--sim-timeout", type=float, default=float(
        os.environ.get("ANCHOROPT_TAU2_SIM_TIMEOUT", "3600")),
                   help="wall-clock cap per simulation; 0 disables. A cap that fires often is "
                        "measuring the cap")
    b.add_argument("--no-preflight", action="store_true",
                   help="skip the endpoint probe (not recommended: an outage then records voids)")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("propose", help="mine, localize, expand, ground -- names no winner")
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=1, help="which ranked residual problem to target")
    p.add_argument("--teacher", default=os.environ.get("ANCHOROPT_TAU2_TEACHER", ""),
                   help="'self' (the model under test teaches itself), a gateway model, or empty "
                        "for history-only")
    p.add_argument("--teacher-n", type=int, default=3,
                   help="how many instruction candidates to ask for")
    p.add_argument("--results", default=None,
                   help="a previous cycle's results.json: replays those measurements so the "
                        "block-coordinate schedule continues and Phi can expand on exhaustion")
    p.set_defaults(fn=cmd_propose)

    e = sub.add_parser("evaluate", help="run arms paired against the frozen incumbent")
    e.add_argument("--out", required=True)
    e.add_argument("--max-arms", type=int, default=0, help="0 = every arm")
    e.add_argument("--resume", action="store_true",
                   help="carry forward arms already present in the results file instead of "
                        "re-measuring them")
    e.add_argument("--results-name", default="results.json",
                   help="where to write measurements, so a later cycle does not clobber an earlier "
                        "cycle's record")
    e.add_argument("--concurrency", type=int, default=4)
    e.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("select", help="core's argmax over J_train on measured arms")
    s.add_argument("--out", required=True)
    s.add_argument("--eval-budget", type=int, default=None)
    s.add_argument("--results-name", default="results.json")
    s.set_defaults(fn=cmd_select)

    v = sub.add_parser("validate", help="run the promoted controller on the HELD-OUT split")
    v.add_argument("--out", required=True)
    v.add_argument("--split", default="test", help="the held-out split (default: test)")
    v.add_argument("--concurrency", type=int, default=8)
    v.add_argument("--portal", default="", help="default: the same portal the train round used")
    v.add_argument("--k", type=int, default=4,
                   help="runs of P0-test and of P1-test each (4 gives pass^1..pass^4); saved runs are "
                        "always reused")
    v.add_argument("--p0-only", action="store_true",
                   help="run P0-test even when there is no controller to test")
    v.add_argument("--call-timeout", type=int, default=180,
                   help="per-call timeout; a dead request becomes a void episode instead of stalling")
    v.add_argument("--retries", type=int, default=1)
    v.set_defaults(fn=cmd_validate)

    r = sub.add_parser("replicate", help="re-run the frozen incumbent K times on the same cases/portal")
    r.add_argument("--out", required=True)
    r.add_argument("--k", type=int, default=4)
    r.add_argument("--concurrency", type=int, default=15)
    r.set_defaults(fn=cmd_replicate)

    rf = sub.add_parser("refill", help="re-run only the VOID episodes of saved held-out runs")
    rf.add_argument("--out", required=True)
    rf.add_argument("--split", default="test")
    rf.add_argument("--portal", default="")
    rf.add_argument("--call-timeout", type=int, default=180)
    rf.add_argument("--retries", type=int, default=1)
    rf.set_defaults(fn=cmd_refill)

    rs = sub.add_parser("rescore", help="score measured arms against the incumbent DISTRIBUTION")
    rs.add_argument("--out", required=True)
    rs.set_defaults(fn=cmd_rescore)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
