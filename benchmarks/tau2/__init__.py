"""AnchorOpt adapter for tau-bench (tau2).

`tau2_runtime.ADAPTER` is the contract surface; `tau2_mechanism` is the executing code its capability
bindings name; `run_anchoropt_round.py` drives the phases. Only `tau2_mechanism` and `tau2_episodes`
import tau2, so the contract and observability tests run with no tau-bench checkout and no model.

Start at README.md in this directory; results are in docs/TAU2_SELFTEACH_ROUND1.md.
"""
