# Why this directory has no per-case JSON

A9's four shards (jobs 1146571–1146574) ran on the cluster and their per-case artifacts have **not**
been copied into this repo yet. Every other round ships its `result/*.json`, so the gap is recorded
rather than left to be noticed.

What that means for the numbers here:

* the **vector-shard** figures (train 54/89 vs 39/89, dev 21/40 vs 14/40) are **transcribed** from the
  acceptance document, not recomputed by this repo's tests;
* the **whole-corpus** figures are a **projection** on top of them, and are labelled as such
  everywhere they appear;
* `rounds/T8_A8_dedup_clear/` remains the last round whose totals this repo can recompute end to end,
  which is why T8 is called the measured endpoint.

To close this, copy the four shard result files here and add them to the recompute path the way
`test_acceptance_and_stack.py` does for T8. The three-shard confirmation run described in
[`../FROZEN.md`](../FROZEN.md) should land at the same time.
