# Figures

| file | used by | what it shows |
|---|---|---|
| `anchoropt_agent_loop.png` | [`../../README.md`](../../README.md), [`../THE_LOOP.md`](../THE_LOOP.md) | baseline agent loop vs the AnchorOpt loop, with the three incision points marked |

**Keep the alt text meaningful.** These figures are referenced from documents whose whole claim is that
the numbers in them are checkable; a figure with `![](...)` and no description is unreadable to anyone
using a screen reader and unsearchable for everyone else.

**If you replace `anchoropt_agent_loop.png`, keep the three incision points labelled with the same names
used in code** — `pre_generation`, `post_generation_pre_exec`, `post_execution` (see
[`../../anchoropt/anchor.py`](../../anchoropt/anchor.py)). A figure that renames them is worse than no
figure, because the reader cannot map it onto `IncisionPoint`.
