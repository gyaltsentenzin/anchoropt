"""
Shared, dependency-light helpers for the eval/train scripts.

Two concerns live here so `run_memory_eval.py` and `run_memory_train.py` stop
duplicating them:

1. Split loading — prereq/query filtering and the train+val+test "all" union,
   previously copy-pasted (and drifting) across both scripts' ~20 inline
   `"prereq" in id` checks.
2. Model config — loading `model_config.json` and seeding an argparse parser's
   defaults from it, so the eval-model identity has one source of truth.

Kept free of any `bfcl_eval` import (mirrors template_engine.py) and of any heavy
runtime deps, so it is cheap to import and unit-test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

_PathLike = Union[str, Path]

# The three on-disk splits. Each file is self-contained: train/val share the same
# prereq chains; test's prereqs are fully disjoint (confirmed by id/chain inspection).
SPLITS = ("train", "val", "test")


# ── Split filtering ───────────────────────────────────────────────────────────

def split_prereqs(cases: Sequence[Dict]) -> List[Dict]:
    """Return only the prereq (state-setup) cases — those whose id contains 'prereq'."""
    return [c for c in cases if "prereq" in c["id"]]


def split_queries(cases: Sequence[Dict]) -> List[Dict]:
    """Return only the query (scored) cases — the complement of split_prereqs()."""
    return [c for c in cases if "prereq" not in c["id"]]


def read_split_file(cases_dir: _PathLike, split: str) -> List[Dict]:
    """Load <cases_dir>/memory_<split>_cases.json (raises FileNotFoundError if absent)."""
    path = Path(cases_dir) / f"memory_{split}_cases.json"
    with open(path) as f:
        return json.load(f)


def prereq_union(cases_dir: _PathLike, splits: Sequence[str] = SPLITS) -> List[Dict]:
    """Deduped union of prereq cases across the given splits (order-preserving).

    Needed for a '--split all' run: it scores every split's queries, so the store
    must be seeded from every split's prereqs — not just one split's.
    """
    seen: set = set()
    out: List[Dict] = []
    for s in splits:
        for c in read_split_file(cases_dir, s):
            if "prereq" in c["id"] and c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)
    return out


def load_split_cases(cases_dir: _PathLike, split: str) -> List[Dict]:
    """Cases for a split.

    - A single split ('train'/'val'/'test') → the full file contents (queries + prereqs).
    - 'all' → the union of *query* cases across train+val+test (prereqs come from
      prereq_union(), since 'all' scores every split's queries).
    """
    if split == "all":
        out: List[Dict] = []
        for s in SPLITS:
            out.extend(split_queries(read_split_file(cases_dir, s)))
        return out
    return read_split_file(cases_dir, split)


# ── Model config ────────────────────────────────────────────────────────────────

# argparse dest → model_config.json top-level key. Only these top-level keys seed
# argparse defaults; the `decode` block is consumed directly by the evaluator.
_CONFIG_TO_ARG = {
    "model_path": "model_path",
    "port": "port",
    "dtype": "dtype",
    "max_model_len": "max_model_len",
    "served_model_name": "served_model_name",
    "handler_module": "handler_module",
    "handler_class": "handler_class",
    "registry_name": "registry_name",
}


def load_model_config(path: _PathLike) -> Dict:
    """Load a model_config.json into a dict (drops the leading _comment field)."""
    with open(path) as f:
        cfg = json.load(f)
    cfg.pop("_comment", None)
    return cfg


def apply_model_config(parser: argparse.ArgumentParser, cfg: Dict) -> None:
    """Seed an argparse parser's defaults from a model_config dict.

    Explicit CLI flags still override these (set_defaults only changes defaults),
    and omitting the config entirely leaves the parser's hardcoded defaults intact
    — the byte-identical-stock guarantee at the config layer.

    Null-valued config fields (e.g. model_path/served_model_name defaulting to null)
    are skipped so they don't clobber a real argparse default with None.
    """
    defaults: Dict = {}
    for cfg_key, arg_dest in _CONFIG_TO_ARG.items():
        if cfg_key in cfg and cfg[cfg_key] is not None:
            defaults[arg_dest] = cfg[cfg_key]
    decode = cfg.get("decode", {})
    if "seed" in decode:
        defaults["seed"] = decode["seed"]
    parser.set_defaults(**defaults)


def decode_params(cfg: Optional[Dict]) -> Dict:
    """Extract evaluator decode kwargs from a config dict, defaulting to the inline
    literals when the config (or its decode block) is absent — byte-identical stock."""
    decode = (cfg or {}).get("decode", {})
    return {
        "temperature": decode.get("temperature", 0),
        "max_tokens_cap": decode.get("max_tokens_cap", 4096),
        "max_tokens_floor": decode.get("max_tokens_floor", 256),
        "stop": decode.get("stop") or None,
    }
