# T4 — nothing installed

**Result: 38.61 % → 38.61 %, +0.00 pp. No anchor.**

This round is kept as a first-class directory because **the decision it records is the one that made A4
possible.** Deleting it would leave A4 looking like it came from nowhere.

| | |
|---|---|
| incumbent | N0 + A1 + A2 + A3 — 117/303 = 38.61 % |
| candidates considered | every locus in the error-keyed vocabulary |
| verdict | **`EXPAND_ATTRIBUTION`** |
| criteria | [`STOPPING_CRITERIA_FROZEN.md`](STOPPING_CRITERIA_FROZEN.md), frozen before the A2 verdicts landed |

## What happened

Every locus was **simultaneously S1-STOP and S2-STOP** — below the 10 % coverage floor *and* mostly
already settled by the installed anchors. The ranked list still had entries; none of them could support
an arm. The largest explained candidate covered 8 % of the residual, under S1's 10 % floor, while the
**unexplained** share stood at 97 %.

So the loop faced a question that looks identical from the ranking and demands opposite responses.

## The lesson: two terminal states that must not be conflated

    STOP                 nothing survives the gates  AND  unexplained mass is small
                         -> the residual is genuinely addressed. Done.

    EXPAND_ATTRIBUTION   nothing survives the gates  AND  unexplained mass DOMINATES
                         -> the bottleneck moved from POLICY to SIGNAL. Not done; blind.

"Nothing survives the gates" **alone** does not mean the work is done — it means nothing survives *in
the current representation*. Here the unexplained residual was 97 %, so forcing another anchor would
have meant optimizing a vocabulary that could not see almost anything that was left.

`phase_switch.py` makes this call **quantitatively, from frozen S1/S2/S5 thresholds**, rather than by a
human noticing the ranking looks thin. That matters: at this exact point the temptation is to relax a
threshold and keep going, and a threshold relaxed after seeing the ranking is not a threshold.

## Expanding the vocabulary: what makes a *detector*

A backward-trace class (`never_stored`, `stored_then_cleared`, `stored_but_not_retrieved`) is a
**post-hoc label on a whole episode**. A miner needs something else: a condition **checkable at a
decision point**, with no hindsight. That distinction is the whole content of the expansion step.

Each candidate condition is scored on the only two numbers that decide it:

- **coverage** — share of unexplained failures it fires on
- **precision** — fires-on-failure / (fires-on-failure + fires-on-**success**)

| candidate | coverage | precision | verdict |
|---|---|---|---|
| **`no_tool_call_at_all`** | 48 failing | **0.80** | **accepted → becomes A4** |
| `short_episode` | 77 % | 0.57 | rejected — fires on 89 of 117 successes |
| `answered_after_read` | — | 0.50 | rejected |
| `single_read_only` | — | 0.47 | rejected |

**Coverage alone never proposes a detector.** `short_episode` covered 77 % of the residual and reads
exactly like premature answering — and carries almost no information, because it fires on successes
nearly as often as failures. A condition that cannot separate passes from failures is not a signal,
however intuitive it sounds.

This is the same shape as an earlier detector built for the same reason: a whole failure class that
returned *success-looking* payloads and therefore had no error string to key on.

## What this round does **not** license

The residual's largest single class is *"retrieved OK but answered wrong"* — 82 % of one measured
residual. It is **deliberately not anchored**, and the rule is written down: it is an observed
**outcome class**, not an attributed **mechanism**. No answer-formation signal gets added unless
counterfactual attribution shows a recurring decision point the existing signals cannot represent.

Wanting a signal for the biggest bucket is not the same as having one.

## What it changed for the next round

The expanded vocabulary now contains `no_tool_call_at_all`, and the next phase check runs against it
rather than the error-keyed set alone. That detector becomes **[T5 / A4](../T5_A4_no_tool_call/)** —
the round that proves the expansion was worth doing rather than merely defensible.
