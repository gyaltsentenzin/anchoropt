"""Where a run's artifacts go, and what identifies a run. **One owner for the whole layout.**

    results/<model>/<split>_<config>/
        response.jsonl          the transcripts
        analysis.jsonl          the per-turn sidecar (--analyze-tool-calls)
        scores.json             the aggregate, from evaluation.py
        detail.jsonl            per-episode acc / SR / PSR / termination
        anchor_trace.jsonl      proposed -> intervened -> executed, per evaluation
        anchor_telemetry.json   the aggregate firing counts
        run_manifest.json       exactly what was run, hashed

WHY THIS IS A MODULE AND NOT THREE COPIES OF A FORMAT STRING
------------------------------------------------------------
`benchmarks/bfcl_v4/adapter.py` exists because one case-id regex had been reimplemented five times and
three of the copies disagreed -- two of them silently misfiling a case under the wrong backend. The
same shape was already starting here: `run_hosted.py`, `analyze_response.py` and `run_vllm.py` each built
the output path from their own format string, and they had already diverged (`run_vllm.py` wrote
`response_granite41.jsonl` into the results root while `run_hosted.py` wrote
`response_<config>_<model>.jsonl` into a `trial_N_<split>` subdirectory, and `analyze_response.py`
looked for a third shape under `results/granite`). A layout implemented three ways is a layout that
loses artifacts.

WHY PER-MODEL, AND WHY A RUN SUBDIRECTORY UNDER IT
-------------------------------------------------
Per-model is isolation, and `analyze_response.py` already had the reason written down: "runs are kept
under a per-MODEL root so two models' outputs cannot be differenced by accident -- anchors are mined
from one model's residual and do not transfer as artifacts."

The run subdirectory is not a re-run of `trial_N_<split>`. `detail.jsonl`, `anchor_trace.jsonl` and
`anchor_telemetry.json` all have FIXED names -- `evaluation.py` writes `detail.jsonl` beside the
response file, and the trace and telemetry go "beside the response file" as their flags promise. Two
arms sharing one directory would therefore overwrite each other's trace and each other's per-episode
detail, which are precisely the two files a paired comparison is computed from. So each run gets its
own directory, keyed by what actually distinguishes runs: the split and the policy.

`<config>` is `baseline` for the control and the controller spec's stem for an arm, which is what
`analyze_response.py` already documented: the tag "names the POLICY the run used ... It used to be
derived from --enable-anchor-a1, a flag that enabled no mechanism, so 'a1' named a file
byte-identical to the baseline."

WHY `--trial` IS GONE
--------------------
It was a second, unmodelled replicate axis that only renamed directories. `response_generator.py`
declared it and never read it; `run_hosted.py` used it solely to build `trial_N_<split>/`.

The replicate axis that the code actually understands is `--repeat`: it gives each replicate its own
episode id (`<query>_<i>`), and `evaluation.py` aggregates across replicates and reports the spread --
including the honest note that with `--repeat 1` the spread "is identically 0.00 and means nothing at
all". So the variance floor is measured with `--repeat 2` in ONE run, under one set of conditions,
rather than by differencing two directories. Two runs launched separately differ in more than their
label, which is the whole reason `COLLABORATORS.md` says never to difference accuracies across jobs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Both spellings of each model map to ONE directory: the short slug a vLLM run passes as `--model`,
# and the HuggingFace id a the hosted API run passes. `granite-4-1-8b` and `ibm-granite/granite-4.1-8b` are the
# same model, and a layout that filed them apart would invite differencing one against the other --
# the exact accident the per-model root exists to prevent.
MODEL_DIRS: dict[str, str] = {
    "granite-4-1-8b": "granite",
    "granite-4.1-8b": "granite",
    "ibm-granite/granite-4.1-8b": "granite",
    "qwen-3-6-35b": "qwen",
    "Qwen/Qwen3.6-35B-A3B": "qwen",
    "minimax-m2-5": "minimax",
    "MiniMaxAI/MiniMax-M2.5": "minimax",
    "deepseek-v3-2": "deepseek",
    "deepseek-ai/DeepSeek-V3.2": "deepseek",
}

ARTIFACTS = ("response.jsonl", "analysis.jsonl", "scores.json", "detail.jsonl",
             "anchor_trace.jsonl", "anchor_telemetry.json", "run_manifest.json")


def slugify(name: str) -> str:
    """A filesystem-safe, injective-enough form of a model id.

    Injective matters and prettiness does not: the fallback's only job is to keep two models' results
    in different directories. A the hosted API id carries a `/` (`ibm-granite/granite-4.1-8b`), which would
    otherwise create a nested directory and put the family name where a model name belongs.
    """
    out = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")
    return out or "unknown-model"


def model_dir(model: str) -> str:
    """The per-model directory name for `model`.

    Falls back to the slugified id rather than raising, and that is a deliberate difference from
    `cctu_adapter.parse_case_id`, which refuses to guess. The reason the case-id parser must raise is
    that a wrong guess FILES A CASE UNDER THE WRONG BACKEND -- it corrupts an aggregate. A wrong guess
    here only produces an ugly directory name, and the isolation property still holds because the
    slug is injective. Refusing to run a new model until someone edits a table would be the worse
    trade.
    """
    return MODEL_DIRS.get(str(model)) or slugify(model)


def config_tag(controllers: str | Path | None) -> str:
    """`baseline` for the control, else the controller spec's stem.

    The tag names the POLICY, never the intent. A run labelled by what someone meant to enable rather
    than by the file that was loaded is how an arm ends up byte-identical to the baseline under a
    different name.
    """
    if not controllers:
        return "baseline"
    return Path(str(controllers)).stem or "baseline"


def run_dir(*, root: str | Path, model: str, split: str,
            controllers: str | Path | None = None) -> Path:
    """`<root>/<model>/<split>_<config>` -- the directory every artifact of one run lives in."""
    return Path(root) / model_dir(model) / f"{split}_{config_tag(controllers)}"


def artifact(run: str | Path, name: str) -> Path:
    """One artifact inside a run directory, refusing a name the layout does not declare.

    A typo in an artifact name would otherwise write a file nothing reads, and a missing artifact is
    indistinguishable from a run that did not produce one.
    """
    if name not in ARTIFACTS:
        raise ValueError(f"{name!r} is not a declared artifact; declared: {list(ARTIFACTS)}")
    return Path(run) / name


def file_digest(path: str | Path) -> str | None:
    """`sha256:<hex>` of a file, or None when it is absent. Used by the manifest."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def write_manifest(run: str | Path, payload: dict[str, Any]) -> Path:
    """Record exactly what was run, so a reproduction can be COMPARED rather than assumed.

    `COLLABORATORS.md` asks for the criteria of a round to be frozen with a hash, for the reason that
    "a stopping rule chosen after seeing the result is a rationalisation, not a rule". The same
    argument applies one level down to the run itself: a baseline whose decode settings, corpus and
    policy are not recorded cannot be reproduced, only re-attempted -- and two runs that differ in an
    unrecorded parameter look like a variance floor.

    Written at the START of a run, so a crashed run still says what it was trying to do.

    ENDPOINTS ACCUMULATE ACROSS INVOCATIONS, and that is the one field this function does not simply
    overwrite. `response_generator.py` resumes by episode id, so a run interrupted by an allocation
    ending is finished by a second invocation -- usually on another node, because the first one is
    gone. Overwriting `endpoint` made the manifest name only the LAST server while the transcripts came
    from several, which is the failure this whole function exists to prevent: an unrecorded parameter
    two runs differ in reads as a variance floor.

    So `endpoint` keeps its meaning -- the server THIS invocation used, which is what
    `analyze_residual.compare` reads -- and `endpoints` is the ordered list of every server that
    contributed, earliest first. One entry means one job. More than one means a split run, which is
    not by itself a defect: episodes share no state, so a split is a PROVENANCE fact that
    `check_pairing.py` then either clears or does not. What is a defect is a split that no artifact
    mentions.
    """
    path = artifact(run, "run_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    prior: list[Any] = []
    if path.exists():
        try:
            was = json.loads(path.read_text())
            prior = list(was.get("endpoints") or [])
            if not prior and was.get("endpoint"):
                prior = [was["endpoint"]]        # a manifest written before this field existed
        except (OSError, ValueError):
            prior = []
    here = payload.get("endpoint")
    seen = prior + ([here] if here and (not prior or prior[-1] != here) else [])
    if seen:
        payload["endpoints"] = seen
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path
