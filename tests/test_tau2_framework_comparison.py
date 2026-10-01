"""docs/TAU2_FRAMEWORK_COMPARISON.md is the output of compare_frameworks.py, and the script reads the other
frameworks' data the way they do.

  * every table in the document is exactly what the script prints from the artifacts, so no number in it
    can drift from the trials it was computed from;
  * the script's pass^k for GEPA's and Self-Harness's candidates equals what each framework published in
    its own hand-off (GEPA_RESULTS_HANDOFF.md §7, DATA_MAP.md §4) -- an independent check that the
    per-task outcomes are read the way their owners read them;
  * a GEPA cell that returned the seed prompt ships stock, and is not counted as a second measurement.

GEPA's and Self-Harness's data live outside this repo; without them mounted the tests skip.
"""

from __future__ import annotations

import importlib.util
import os
import statistics
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "TAU2_FRAMEWORK_COMPARISON.md"
GEPA = Path(os.environ["GEPA_TAU2_MATRIX"]) if os.environ.get("GEPA_TAU2_MATRIX") else None
SH = Path(os.environ["SELF_HARNESS_TAU2_DIR"]) if os.environ.get("SELF_HARNESS_TAU2_DIR") else None

pytestmark = pytest.mark.skipif(not (GEPA and SH and GEPA.is_dir() and SH.is_dir()),
                                reason="GEPA_TAU2_MATRIX / SELF_HARNESS_TAU2_DIR not set or not mounted")


@pytest.fixture(scope="module")
def cf():
    spec = importlib.util.spec_from_file_location(
        "compare_frameworks", REPO / "benchmarks" / "tau2" / "scripts" / "compare_frameworks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rows(cf):
    return cf.build()


def _cell(rows, model, domain, teacher="self"):
    return next(r for r in rows if (r["model"], r["domain"], r["teacher"]) == (model, domain, teacher))


def test_every_table_in_the_document_is_the_scripts_output(cf, rows):
    doc = DOC.read_text()
    for title, fn in cf.TABLES:
        assert fn(rows) in doc, f"the {title!r} table in {DOC.name} differs from compare_frameworks.py"


def test_gepa_pass_hat_k_equals_its_own_handoff(rows):
    # GEPA_RESULTS_HANDOFF.md §7: the six self-teach arms with a changed prompt, 4 trials each
    published = {("retail", "minimax-m2.5"): (0.863, 0.808, 0.787, 0.775),
                 ("retail", "qwen3.6-35b-a3b"): (0.838, 0.771, 0.738, 0.725),
                 ("airline", "qwen3.6-35b-a3b"): (0.800, 0.683, 0.600, 0.550),
                 ("retail", "granite-4.1-30b"): (0.719, 0.600, 0.519, 0.450),
                 ("airline", "minimax-m2.5"): (0.688, 0.567, 0.475, 0.400),
                 ("airline", "granite-4.1-30b"): (0.338, 0.250, 0.212, 0.200)}
    got = {}
    for (domain, model), want in published.items():
        c = _cell(rows, model, domain)["gepa"]["candidate"]
        got[domain, model] = [c["pass_hat_k"][k] for k in (1, 2, 3, 4)]
        assert c["runs"] == 4
        assert got[domain, model] == pytest.approx(want, abs=6e-4), (domain, model)
    means = [statistics.fmean(v[k] for v in got.values()) for k in range(4)]
    assert means == pytest.approx((0.707, 0.613, 0.555, 0.517), abs=6e-4)


def test_self_harness_pass_hat_k_equals_its_own_handoff(rows):
    # DATA_MAP.md §4 / RESULTS_PASSK.md: its two accepted self-teach arms, 4 trials each
    for (domain, model), want, per_run in (
            (("retail", "granite-4.1-30b"), (0.7063, 0.5875, 0.5188, 0.4750), [26, 28, 31, 28]),
            (("telecom", "minimax-m2.5"), (0.9750, 0.9750, 0.9750, 0.9750), [39, 39, 39, 39])):
        c = _cell(rows, model, domain)["self_harness"]["candidate"]
        assert c["per_run"] == per_run
        assert [c["pass_hat_k"][k] for k in (1, 2, 3, 4)] == pytest.approx(want, abs=6e-5)


def test_a_gepa_cell_that_returned_the_seed_ships_stock(rows):
    seed_returned = [r for r in rows if r["gepa"]["status"] == "returned the seed prompt unchanged"]
    assert len(seed_returned) == 4                    # the three telecom self-teach arms, qwen retail sonnet
    for r in seed_returned:
        assert r["gepa"]["candidate"] is r["gepa"]["stock"], (r["model"], r["domain"], r["teacher"])


def test_paired_difference_is_the_difference_of_the_means(cf, rows):
    fw = _cell(rows, "granite-4.1-30b", "airline")["anchoropt"]
    for k in (1, 2, 3, 4):
        d, se = cf.paired(fw["stock"], fw["candidate"], k)
        assert d == pytest.approx(fw["candidate"]["pass_hat_k"][k] - fw["stock"]["pass_hat_k"][k])
        assert se > 0
    assert cf.paired(fw["stock"], fw["candidate"], 5) is None     # pass^5 needs five trials
