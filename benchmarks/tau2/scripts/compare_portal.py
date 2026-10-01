#!/usr/bin/env python3
"""Compare a locally-served model against the same model on the shared gateway: latency, tool calls,
agreement.

Two portals, ONE model. The question is whether serving it ourselves is faster, and whether the four
things that are OUR responsibility -- chat template, tool-call parser, quantization, context length --
line up. Anything else that differs given identical input is the model's business, not ours.

    python compare_portal.py --model qwen3.6-35b-a3b [--n 6]

WHAT THE LATENCY NUMBER DOES AND DOES NOT MEAN. It is per AGENT CALL. An episode is also user-simulator
calls on the gateway, which a portal switch does not touch: on telecom an episode is ~20.5 agent calls
and ~13.5 user calls, the user side is ~58% of the time, and a 5x faster agent call measured as only
~1.3-1.5x per episode. The real benefit of the local portal is CAPACITY -- an arm served locally makes
no shared-gateway requests, so it can run beside the gateway arms without adding gateway load (measured
1.92x throughput on qwen3.6/airline at concurrency 8 vs the gateway at 5).

THE TOOL-CALL CHECK IS A GATE, NOT A STATISTIC. A wrong or missing `--tool-call-parser` does not
error: the server returns a normal completion with the call rendered as prose, vLLM parses no
tool_calls, every agent turn becomes a no-op, and the arm scores near zero with exit 0. That is the
same silent-inertness shape this project keeps finding, so a portal that cannot return a parsed
tool call is refused here rather than measured.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# No default -- see the note in benchmarks/tau2/tau2_models.py. Any OpenAI-compatible base URL works.
HOSTED_BASE = os.environ.get("HOSTED_API_BASE", "").rstrip("/")
HOSTED_MODELS = {
    "granite-4.1-30b": ("granite-4-1-30b", "ibm-granite/granite-4.1-30b"),
    "minimax-m2.5": ("minimax-m2-5", "MiniMaxAI/MiniMax-M2.5"),
    "qwen3.6-35b-a3b": ("qwen3-6-35b-a3b-a100", "Qwen/Qwen3.6-35B-A3B"),
}
# No default -- there is no shippable shared location for this. Point it at wherever your own
# serving job writes `endpoint_<name>.txt` files.
ENDPOINT_DIR = Path(os.environ.get("ANCHOROPT_TAU2_SERVING_DIR", "")) if os.environ.get(
    "ANCHOROPT_TAU2_SERVING_DIR") else None

TOOLS = [{"type": "function", "function": {
    "name": "get_reservation_details", "description": "Look up a reservation by its id.",
    "parameters": {"type": "object", "properties": {"reservation_id": {"type": "string"}},
                   "required": ["reservation_id"]}}}]
# A realistic agent turn: a long policy-ish system prompt, a little history, an obvious tool call.
MESSAGES = [
    {"role": "system", "content": "You are a customer service agent. "
                                  + "Follow the operative policy carefully. " * 120},
    {"role": "user", "content": "Hi, I need to check my reservation ABC123."},
    {"role": "assistant", "content": "Of course -- let me look that up for you."},
    {"role": "user", "content": "Thanks, go ahead."},
]


def local_endpoint(model: str) -> tuple[str, int, str] | None:
    """Read the endpoint file, then PROVE the port answers.

    The file is written only after /v1/models replies, but it also SURVIVES the job's death -- a stale
    file pointed at a dead server for a day. Presence of the file is not liveness.
    """
    if ENDPOINT_DIR is None:
        return None
    f = ENDPOINT_DIR / f"endpoint_{model}.txt"
    if not f.exists():
        return None
    parts = f.read_text().split()
    if len(parts) < 2:
        return None
    host, port = parts[0], int(parts[1])
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=15) as r:
            listing = json.loads(r.read())
        # The server advertises whatever --served-model-name it was given; addressing it by the
        # weights path returned NotFoundError on every call.
        served = str((listing.get("data") or [{}])[0].get("id") or "")
        return (host, port, served) if served else None
    except Exception:
        return None


def probe(label: str, *, model_id: str, api_base: str, headers=None, extra_body=None,
          n: int = 6) -> dict:
    import litellm

    lat, tool_ok, contents = [], 0, []
    for _ in range(n):
        kw = dict(model=f"openai/{model_id}", api_base=api_base, api_key="dummy",
                  messages=MESSAGES, tools=TOOLS, max_tokens=800, temperature=0.0,
                  num_retries=0, timeout=180)
        if headers:
            kw["extra_headers"] = headers
        if extra_body:
            kw["extra_body"] = extra_body
        t = time.time()
        try:
            m = litellm.completion(**kw).choices[0].message
        except Exception as exc:
            print(f"    {label}: call failed {type(exc).__name__}: {exc}"[:180], flush=True)
            continue
        lat.append(time.time() - t)
        calls = [c.function.name for c in (m.tool_calls or [])]
        if "get_reservation_details" in calls:
            tool_ok += 1
        contents.append((str(m.content or "")[:200], tuple(calls)))
    good = [x for x in lat if x == x]
    return {"label": label, "n": len(good), "median": statistics.median(good) if good else None,
            "min": min(good) if good else None, "max": max(good) if good else None,
            "tool_calls_parsed": tool_ok, "samples": contents}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(HOSTED_MODELS))
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    slug, served = HOSTED_MODELS[a.model]
    key = os.environ.get("HOSTED_API_KEY", "")
    if not key:
        raise SystemExit("HOSTED_API_KEY is not set; source the tau-bench .env")

    eb = {"chat_template_kwargs": {"enable_thinking": False}} if a.model.startswith("qwen") else None
    print(f"model: {a.model}   (extra_body: {eb})\n")

    hosted = probe("hosted", model_id=served, api_base=f"{HOSTED_BASE}/{slug}/v1",
                   headers={"HOSTED_API_KEY": key}, extra_body=eb, n=a.n)

    ep = local_endpoint(a.model)
    if ep is None:
        print("  local endpoint: NOT SERVING (no live port; a stale endpoint file does not count)")
        local = None
    else:
        host, port, served_local = ep
        print(f"  local endpoint: {host}:{port}  serving id {served_local!r}")
        local = probe("local", model_id=served_local,
                      api_base=f"http://{host}:{port}/v1", extra_body=eb, n=a.n)

    print(f"  {'portal':8s} {'n':>3s} {'median':>8s} {'min':>7s} {'max':>7s} {'tool_calls':>11s}")
    for r in (hosted, local):
        if r is None:
            continue
        med = f"{r['median']:.2f}s" if r["median"] else "-"
        mn = f"{r['min']:.2f}s" if r["min"] else "-"
        mx = f"{r['max']:.2f}s" if r["max"] else "-"
        print(f"  {r['label']:8s} {r['n']:3d} {med:>8s} {mn:>7s} {mx:>7s} "
              f"{r['tool_calls_parsed']:>4d}/{r['n']:<6d}")

    if local and local["median"] and hosted["median"]:
        print(f"\n  speedup: {hosted['median'] / local['median']:.2f}x "
              f"({hosted['median']:.2f}s -> {local['median']:.2f}s)")

    # ---- the gate ----
    print()
    if local is None:
        print("  VERDICT: local portal not serving. Nothing to decide yet.")
        return 1
    if local["n"] == 0:
        # DIFFERENT FAULT, DIFFERENT FIX. No call SUCCEEDED, so nothing can be said about parsing --
        # an earlier version blamed the tool parser for what was a wrong model id.
        print("  VERDICT: REFUSE. No local call succeeded at all, so tool parsing is untested. "
              "Fix the\n           transport/model-id error above first; this is not a parser "
              "problem.")
        return 4
    if local["tool_calls_parsed"] == 0:
        print("  VERDICT: REFUSE. The local portal answered but parsed ZERO tool calls. This does not raise an "
              "error at\n           run time -- every agent turn would silently become a no-op and "
              "the arm would score\n           near zero with exit 0. Check --tool-call-parser "
              "(qwen3_xml for Qwen3.5/3.6, granite4\n           for Granite 4.1; hermes parses "
              "their XML as nothing).")
        return 2
    if local["tool_calls_parsed"] < local["n"]:
        print(f"  WARNING: only {local['tool_calls_parsed']}/{local['n']} local calls returned a "
              f"parsed tool call.")
    if local["median"] and hosted["median"] and local["median"] >= hosted["median"]:
        print("  VERDICT: usable, but no per-call speedup. It still adds capacity (no gateway load).")
        return 3
    print("  VERDICT: local portal is live and parses tool calls. The speedup above is PER CALL; per\n"
          "           episode expect much less, since the user simulator on the gateway does not move.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
