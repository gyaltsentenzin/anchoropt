# The pipeline is deterministic at fixed seed — measured, not assumed

Established accidentally, from a duplicated arm. `w2a` and `w2b` were submitted as separate LSF jobs on
separate hosts with different eta content, but — because the corrective instruction turned out to be
inert (`rounds/WRITE2/INSTRUCTION_IS_INERT.md`) — they are the **same intervention**. That makes them an
unintended replication.

**Result: 13 of 13 storage episodes have byte-identical call sequences**, identical step counts and
identical stored facts, across two independent runs on different machines.

## Why this matters for every number in this project

Every delta reported here has been a **single run against a single control**: A9's +20.00/+21.54/+23.08pp
on the clean 65, the probe's +3.37pp, the c2inc storage comparison. Without a replication, none of those
could be distinguished from run-to-run variation in the model's own trajectory.

They now can. At fixed seed and fixed configuration:

* **a single run IS the result** for a given arm;
* an observed difference between arms is **not** sampling noise from the generation process;
* a difference that appears between two runs of the *same* configuration would indicate a real
  non-determinism to hunt down — and none appeared.

This does **not** make the deltas statistically significant. `+10/−7` on n=89 is still a net of 3 cases,
and determinism says nothing about whether the effect generalises to another corpus, seed, or model. It
removes one specific alternative explanation — trajectory noise — and nothing more.

## What it also validates retrospectively

The earlier claim that episodes the controller never fired on are **identical** between control and arm
(`mean turn delta +0.00, mean facts delta +0.00, n=11`) rested on the same assumption. That assumption is
now independently measured, which is what licenses reading the fired-episode deltas (−1.62 turns,
−1.88 facts) as causal.

## Caveat

Determinism was measured on the storage phase of one cell (vector) at seed 42 with greedy decoding. The
project's own history records a case where seed variation was inert *because* decoding was greedy — so
this is consistent with that, and equally bounded by it.
