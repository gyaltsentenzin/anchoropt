# BFCL v4 Memory Evaluation — Mechanics & Training-Time Fidelity Audit

> ## ⚠️ Read this before trusting any number in this repo
>
> **`benchmarks/bfcl_v4/run.py` is not the official BFCL pipeline.** It drives
> `evaluator/memory_evaluator.py`, a reimplementation of the episode loop that has **no upstream
> counterpart** — `bfcl generate` is never invoked. What it *does* borrow from BFCL is the part that
> decides correctness: the official tool executor, the official `MemoryAPI_*` backends, the official
> handler methods for context assembly, and the official `agentic_checker` scoring against the official
> ground truth. So a scored case means what BFCL says it means; the *loop around it* is ours.
>
> §4 below enumerates **six divergences**, file:line cited. Status in **this** repo:
>
> | # | divergence | in this repo |
> |---|---|---|
> | 1 | prereq seeding runs in a different environment than scoring | **PRESENT** — highest impact; retrieval success is fully determined by what prereqs stored |
> | 2 | G3/G4 gates absent from the evaluator but present in the official harness | **PRESENT** (and here they are also retired from the registry by design) |
> | 3 | step budget 15 vs official 20 | **FIXED** — 20 is the default in all five places (3 runners + 2 evaluator constructors), pinned to the vendored `MAXIMUM_STEP_LIMIT` by `tests/test_step_budget.py` |
> | 4 | `force_quit` → hard fail; official grades the last message anyway | **PRESENT** |
> | 5 | step-0 injections ephemeral; official gate nudges persist | **PRESENT** |
> | 6 | token accounting `len(prompt)//4` vs real tokenizer | **PRESENT** — only bites near the context limit |
>
> **What this means for the reproduction in [`../REPRODUCE.md`](../REPRODUCE.md).** Running A1–A4 here
> and getting 42.24 % shows that *this pipeline plus these artifacts are self-consistent* — the code and
> the frozen results agree. It is **not** evidence about the official BFCL pipeline, and the anchors'
> measured gain is a gain *within this evaluator*. Divergence #1 in particular is why transfer to an
> official run cannot be assumed: storage-affecting policies are rewarded here in a seeding environment
> the official harness does not reproduce.
>
> Nothing in §4 undermines the *method* — the loop, the attribution, the acceptance ordering are
> independent of it. It bounds what the numbers license.

---

Read-only investigation. Every claim cites `file:line`. **CONFIRMED** = I read the code;
**SUSPECTED** = inferred but not fully proven.

> Path roots below refer to the ORIGINAL working-repo layout the audit was written against. In this
> repo they map to: `BFCL/` → `benchmarks/bfcl_v4/harness/bfcl_eval/`, `AO/` → `benchmarks/bfcl_v4/`
> (with `AO/anchoropt/` → `benchmarks/bfcl_v4/evaluator/` and `AO/scripts/` → `benchmarks/bfcl_v4/`).

Path roots (abbreviated below):
- `BFCL/` = the working repo's `berkeley-function-call-leaderboard/bfcl_eval/`
- `AO/`   = the working repo's `anchoropt/`

Scope: (1) official memory eval mechanics, (2) whether the AnchorOpt training-time
evaluator (`AO/anchoropt/memory_evaluator.py`) faithfully reproduces them. The AnchorOpt
optimizer's accept/reject logic is out of scope (a separate audit).

---

## 1. Episode structure (memory tasks)

### 1.1 Categories, datasets, backends — CONFIRMED
- Memory is one of the **agentic** categories, not "multi_turn": `MEMORY_CATEGORY = [memory_kv, memory_vector, memory_rec_sum]` (`BFCL/constants/category_mapping.py:46`), `AGENTIC_CATEGORY = MEMORY_CATEGORY + WEB_SEARCH_CATEGORY` (`:57`). `is_agentic` = `web_search or memory` (`BFCL/utils.py:215`); `is_memory` (`:203`); `is_memory_prereq` = `"prereq" in id` (`:211`).
- Three backends → classes in `BFCL/eval_checker/multi_turn_eval/func_source_code/`:
  - **kv** `memory_kv.py:18` `MemoryAPI_kv` — two dicts `core_memory` / `archival_memory`; tools are key/value (`core_memory_add(key,value)` `:98`, `archival_memory_key_search` `:320`, etc.). Caps: core ≤7 entries/300 chars, archival ≤50/2000 (`:12-15`).
  - **vector** `memory_vector.py:30` `MemoryAPI_vector` — two `VectorStore`s (FAISS + `all-MiniLM-L6-v2` `:26`); tools take free `text` and return integer ids (`core_memory_add(text)` `:88`).
  - **rec_sum** `memory_rec_sum.py:12` `MemoryAPI_rec_sum` — single `memory` string blob, 10 000-char cap (`:9`); tools `memory_append` `:64`, `memory_update` `:84`, `memory_replace` `:113`, `memory_retrieve` `:136`.
  - All subclass `MemoryAPI` (`memory_api_metaclass.py:14`).
- Data files:
  - Scored questions: `BFCL/data/BFCL_v4_memory.json` — 155 entries, each `{id, question, involved_classes:["MemoryAPI"], scenario}` (CONFIRMED by load). `question` is a list-of-turns; scored memory entries are **single-turn, single user message** (e.g. `memory_0-customer-0` = "What is my first name?").
  - Prereq conversations: `BFCL/data/memory_prereq_conversation/memory_{scenario}.json` (JSONL), 5 scenarios (`student, customer, finance, healthcare, notetaker` = `category_mapping.py:47`). Each prereq entry is itself **multi-turn** (e.g. notetaker-0 has 5 user turns).
  - Tool docs: `BFCL/data/multi_turn_func_doc/memory_{kv,vector,rec_sum}.json` (mapping `executable_backend_config.py:11-13`).
  - Ground truth: `BFCL/data/possible_answer/BFCL_v4_memory.json` — `{id, ground_truth:[strings], source}` (e.g. `ground_truth:["Michael"]`).

### 1.2 Turn vs step; the loops — CONFIRMED
Handler entrypoint dispatches memory (prompting model) to `inference_multi_turn_prompting` (`BFCL/model_handler/base_handler.py:394`; FC twin at `:95`, but the Granite **prompting** handler is used so the prompting path with gates is authoritative — the FC path has **no** gates).
- **Episode** = one test entry = one call to `inference_multi_turn_prompting`. The "episode loop" (ordering of prereqs before scored entries) lives in the runner scheduler `BFCL/_llm_response_generation.py:223 generate_results` (dependency-aware `ThreadPoolExecutor`, `:291-349`), driven by `depends_on`.
- **Turn** = one element of `test_entry["question"]` — outer `for turn_idx …` (`base_handler.py:468`). Turn 0 seeded by `add_first_turn_message_prompting` (`:485`); later turns by `_add_next_turn_user_message_prompting` (`:489`).
- **Step** = one model call + optional tool execution — inner `while True` (`:522`), counter `count` (`:502`).
- **Step-loop terminates** on: (a) empty/no tool call → `is_empty_execute_response(...)` → `break` (`:643`, `:688`); (b) decode exception → `break` (`:690-699`); (c) `count > MAXIMUM_STEP_LIMIT` → `force_quit=True`, break (`:800-808`). `MAXIMUM_STEP_LIMIT = 20` (`BFCL/constants/default_prompts.py:1`). No `FORCE_QUIT_MESSAGE` constant — inline text at `:805`.
- **Persists across steps AND turns**: `inference_data["message"]` (never reset mid-episode) and the backend singleton (globals cache, below). **Ephemeral / per-turn**: gate flags `_d3_*`, `_g1g3_fired`, `_core_full_seen`, `_g4_fired` (reset at `:503-521`), logs, token counters.

---

## 2. Prerequisite seeding & context assembly  *(most detail here)*

### 2.1 How initial memory state is SEEDED — CONFIRMED
Seeding is **not** static data injected into the prompt. The prereq conversations are *run as full inference episodes first*, and memory is persisted to **on-disk snapshots keyed by scenario**; the scored entry then loads that snapshot. Chain:

1. **Data-load / dependency wiring** — `process_memory_test_case` (`BFCL/utils.py:714`):
   - Loads `memory_{scenario}.json` prereqs (`:727-729`).
   - Derives backend class: `backend_class_name = f"MemoryAPI_{backend_type}"` (`:731-732`).
   - Renames ids `memory→memory_kv/…`, sets `depends_on` = chain of prior prereq ids, sets `involved_classes=[backend_class_name]` on prereqs (`:736-742`) and on scored entries (`:745-750`). So each scored entry **depends_on the full prereq chain**.
2. **initial_config construction** — `populate_initial_settings_for_memory_test_cases` (`BFCL/utils.py:837`, called from runner `_llm_response_generation.py`): sets `entry["initial_config"] = { backend_class: {model_result_dir, scenario, test_id, test_category} }` (`:847-855`).
3. **Backend instantiation** (per inference call, empty bootstrap call): `execute_multi_turn_func_call([], initial_config, involved_classes, model_name, test_entry_id, …)` (`base_handler.py:421-429`). Inside `multi_turn_utils.py:13`, instances are globals-cached as `{model}_{test_id}_{class}_instance` (`:33-37`) and `class_instance._load_scenario(deepcopy(class_initial_config), …)` is called (`:44-46`); reused across turns from `globals()` (`:49-50`) — this is how memory persists **within** an episode.
4. **Snapshot load** — `_load_scenario` → `_prepare_snapshot(initial_config)` (`memory_kv.py:29-36`; vector `:47-53`; rec_sum `:22-31`). `_prepare_snapshot` (`memory_api_metaclass.py:25`, `@final`) reads `model_result_dir/test_id/scenario` (`:39-41`); returns `None` for the first prereq entry (`is_first_memory_prereq_entry` → clean start, `:59-61`); otherwise loads `{scenario}_final.json` (`:57,75-76`).
5. **Snapshot flush** — after a **prereq** episode only: `if is_memory_prereq(test_entry_id): memory_instance._flush_memory_to_local_file()` (`base_handler.py:847-852`). Flush writes both `{test_id}.json` and the shared `{scenario}_final.json` (`memory_kv.py:38-63`). Scored entries do NOT flush; they only read.

Net: prereq episodes accumulate memory across their own turns **and** across chained prereq entries (all sharing `{scenario}_final.json`); the scored single-turn question inherits that state. What the model can retrieve at score time is entirely determined by what the prereq episodes wrote.

### 2.2 Exact per-call context assembly (order) — CONFIRMED
For a scored memory entry the message list for each model call is built as follows.

1. **Agentic response-format system message** injected at data-load: `process_agentic_test_case` inserts `ADDITIONAL_SYSTEM_PROMPT_FOR_AGENTIC_RESPONSE_FORMAT` at position 0 of turn 0 for non-prereq agentic entries (`BFCL/utils.py:760-767`; text at `default_prompts.py:79`). (Prereq entries are excluded — `not is_memory_prereq`.)
2. **Memory backend instruction + current core memory** prepended by `add_memory_instruction_system_prompt` (`BFCL/model_handler/utils.py:610`), called from `base_handler.py:437` right after bootstrap instantiation:
   - `scenario_setting = MEMORY_AGENT_SETTINGS[scenario]` (`:623`; per-scenario persona, `default_prompts.py:82-88`).
   - `memory_content = memory_instance._dump_core_memory_to_context()` (`:625`) — **current core-memory contents loaded from the snapshot** (kv dumps `core_memory` dict `memory_kv.py:65-68`; vector dumps `core_memory._store` `memory_vector.py:82-86`; rec_sum dumps the whole blob `memory_rec_sum.py:58-62`).
   - Template: `MEMORY_BACKEND_INSTRUCTION_UNIFIED` if `rec_sum` else `MEMORY_BACKEND_INSTRUCTION_CORE_ARCHIVAL` (`:627-630`; templates `default_prompts.py:157` / `:170`).
   - Placement: if a system message already exists (the step-1 agentic one), the memory block is **prepended** to it (`:639-642`); else inserted at index 0 (`:644-648`). Final system prompt order = **[memory backend instruction + core-memory dump] + "\n\n" + [agentic response-format instruction]**.
   - **IMPORTANT (this repo):** the P1–P6 nudges are **commented out** of both templates (`default_prompts.py:166-168, 177-178`) — "now owned by `on_memory_preamble` in AnchorOpt template policy". So the *official* eval system prompt here carries **no** P1–P6 guidance; those nudges exist only if AnchorOpt injects them.
3. **Function/tool docs** folded into the system prompt in prompting mode: `_pre_query_processing_prompting` → `system_prompt_pre_processing_chat_model(question[0], functions, id)` (`base_oss_handler.py:373-381`). For Granite, `_format_prompt` re-emits tools in `<tools>…</tools>` (`granite_4.py:365-389`).
4. **Turn user message(s)** appended (`base_handler.py:485/489`).
5. **Assistant message** appended after each query (`:548` → `_add_assistant_message_prompting`).
6. **Tool execution results** appended back into history after each executed step: `_add_execution_results_prompting` (`base_handler.py:715`). For Granite/OSS prompting these become `{"role":"tool", "name":…, "content":execution_result}` entries (`base_oss_handler.py:415-429`); the OpenAI prompting variant instead wraps them into a single `{"role":"user"}` message (`openai_completion.py:266-276`).

Model call itself: prompting handler builds one prompt string via `_format_prompt` and calls `self.client.completions.create(model, temperature=self.temperature, prompt, max_tokens=min(4096, ctx-in_tokens-2), [extra_body])` (`base_oss_handler.py:322-370`). `extra_body` carries `stop_token_ids`/`skip_special_tokens` only if the handler defines them — **Granite defines neither** (grep of `granite_4.py`/`base_oss_handler.py` shows only the `hasattr` guards, no assignment), so `extra_body` is empty for Granite.

### 2.3 Tool call → backend method → return value — CONFIRMED
Decode: `decode_execute(model_responses, has_tool_call_tag=False)` (`base_handler.py:572`). Execute: `execute_multi_turn_func_call(decoded, …)` (`:702`) → `_process_method_calls` rewrites bare names to `instance.method` (`multi_turn_utils.py:111-144`), blocks destructive builtins (`:75-81`), runs `eval(func_call)` (`:83`), serializes returns (dict→JSON, else str; exceptions → `"Error during execution: …"`) (`:85-98`).

### 2.4 The four "Memory Gates" (non-upstream, present in THIS repo) — CONFIRMED
These live only in `inference_multi_turn_prompting`, only for memory, mostly KV/Vector (`rec_sum` excluded). They mutate the trajectory:
- **G1** (`base_handler.py:591-641`): pre-execution, strips `core_memory_clear` calls unless the user said "forget/clear/reset everything"; if that empties the step, injects a redirect user turn and `continue`s.
- **D3** (`:585-589` track, `:652-687` fire): forces an archival search before accepting an IDK answer (fires once/turn when archival not yet searched and the response contains an IDK phrase).
- **G3** (`:727-763`): post-execution, on a "core/archival is full" tool error, injects an overflow-redirect user turn and `continue`s (once/turn).
- **G4** (`:765-796`, **KV only**): post-execution, on `"Key not found"`, injects a key-search fallback user turn and `continue`s (once/turn).
Gate injection text is **hardcoded** in `base_handler.py` (`:631-635`, `:681`, `:754-757`, `:788-789`).

---

## 3. Scoring — CONFIRMED

- **Response-based only for memory.** Memory routes to the agentic path (`eval_runner.py:730-739`), scored by `agentic_checker(last_non_function_call_message, ground_truth_list)` (`eval_runner.py:145-148`). `agentic_checker` (`BFCL/eval_checker/agentic_eval/agentic_checker.py:6-35`) standardizes strings (strip `,./-_*^()`, lowercase, quote-normalize `:41-49`) and returns `valid` if any ground-truth string appears as a **word-bounded substring** of the model's final text. **No backend state comparison, no execution, no ground-truth replay** for memory.
- **Ground truth** = the `ground_truth` list of acceptable answer strings, loaded from `possible_answer/BFCL_v4_memory.json` for all three backends (`utils.py:453-454`), extracted as `possible_answer[i]["ground_truth"]` (`eval_runner.py:421`).
- **Which message is graded**: `_evaluate_single_agentic_entry` walks per-step outputs, uses `decode_execute` merely to find the last message that does **not** decode as a tool call (the final chat answer) and matches that (`eval_runner.py:112-124, 145-148`).
- **Passed turn vs episode / all-or-nothing**: memory scored entries are single-turn, so entry == episode; one point per entry if `valid` (`eval_runner.py:532-536`). (Contrast: the separate *multi_turn* categories are both state-based `multi_turn_checker.py:107` and response-based `:114`, all-or-nothing across turns via early return — but memory does **not** use that path.)
- **kv / vector / rec_sum scored identically** — same file, same checker, no backend branch (`agentic_checker` takes no category arg).

---

## 4. Fidelity cross-check — AnchorOpt training-time evaluator vs official harness

The training-time evaluator is `AO/anchoropt/memory_evaluator.py::MemoryAnchorOptEvaluator`. It is used by **both** `AO/scripts/run_memory_train.py` (training) and `AO/scripts/run_memory_eval.py` (the researcher's "eval") — i.e. the researcher's eval is this evaluator, **not** the true BFCL pipeline (`_llm_response_generation.py` + `eval_runner.py`). `AO/anchoropt/final_eval.py` uses the older v3 `AnchorOptEvaluator` (`final_eval.py:22`) and is not the memory path. The table below compares `MemoryAnchorOptEvaluator` against the true official harness.

### 4.1 What is FAITHFUL (good) — CONFIRMED
- **No reimplementation of tools / no contamination.** Executes via the official `execute_multi_turn_func_call` (`memory_evaluator.py:112-118, 309-317`), instantiates the official `MemoryAPI_*` classes, uses the official `add_memory_instruction_system_prompt` (`:122`) and official `agentic_checker` (`:105`). Snapshots are written to `tempfile.mkdtemp()` / an AnchorOpt cache dir (`:216`, `model_result_dir=self._snapshot_dir` at `:526`), **never** into `BFCL/eval_checker/`. No writes/mutations to the official tree found.
- **Scoring math identical.** Grades the last non-decodable message (`_last_non_fc_message` `:154-171`, using `decode_execute(..., has_tool_call_tag=False)` like `eval_runner`) with the official `agentic_checker(final_answer, possible_answer)` (`:869`). `possible_answer` in the built cases is exactly the official `ground_truth` list — `build_memory_cases.py:79,101` merges `ground_truth` into `possible_answer`; verified a built case has `possible_answer:["Cereal"]`. Same shape the official checker receives.
- **Context assembly order preserved.** Bootstrap `execute_multi_turn_func_call([])` → `add_memory_instruction_system_prompt` → `_pre_query_processing_prompting` → per-turn `add_first_turn/_add_next_turn` → `_add_assistant_message` → `_add_execution_results` all delegate to the **same official handler methods** (`memory_evaluator.py:536-562, 595-601, 659-661, 791-793`). Prompt string built by the official `handler._format_prompt` (`:237`). So system-prompt/tool-doc/turn/tool-result assembly matches.
- **Snapshot continuity + prereq flush.** Reconstructs `initial_config` per class with `model_result_dir/test_id/scenario` (`:524-528`) and flushes prereqs via the official `_flush_memory_to_local_file` (`:838-843`). `_reset_episode_state` deletes only the exact `(model,test_id,class)` singleton (`:321-351`) so chains persist. Same query endpoint (`/v1/completions`) and temperature (0.001) as official Granite prompting.

### 4.2 DIVERGENCES (ranked by impact on training signal)

| # | Divergence | Where | Impact on training signal | Status |
|---|---|---|---|---|
| **1** | **Prereq seeding runs in a different environment, so the stored memory the scored turn reads differs.** In `run_memory_train`, prereqs are (re)seeded **with the candidate `on_memory_preamble`** and are **re-run per candidate** (`run_memory_train.py:581,686,719,732`); official prereq generation uses the base (nudge-stripped) system prompt with **no** AnchorOpt injection. Also prereqs run under the *evaluator's* gate/step set (see #2,#3), not the official one. And `run_memory_eval` seeds **without** templates (`run_memory_eval.py:189`) while training seeds **with** them → eval-time seeded state ≠ training seeded state. | `memory_evaluator.py:355-474`, `build_snapshot_store`; `run_memory_train.py:581,686`; `run_memory_eval.py:189` | **Highest.** Retrieval success is fully determined by what prereqs stored (§2.1). If storage differs between train, the researcher's eval, and the true harness, the measured gain need not transfer. Storage-focused templates are rewarded in training but their storage effect is absent from `run_memory_eval`'s seeding and from any un-patched official run. | CONFIRMED |
| **2** | **G3 and G4 gates are missing.** Evaluator implements only **G1** (`memory_evaluator.py:699-723`) and **D3** (`:730-757`). Official prompting harness also has **G3** (core-full → archival redirect, `base_handler.py:727-763`) and **G4** (KV "Key not found" → key-search fallback, `:765-796`). | `memory_evaluator.py` (no G3/G4) vs `base_handler.py:727-796` | **High**, esp. KV (both G3+G4) and vector (G3). The training env withholds two corrective nudges the official harness gives, so training is *harder* than official on overflow / wrong-key retrieval → pessimistic pass-rate and pressure to optimize templates that duplicate G3/G4's job (redundant at official eval, may not transfer). | CONFIRMED |
| **3** | **Step budget 15 vs 20.** Evaluator default `max_steps_per_turn=15` and both run scripts pass 15 (`memory_evaluator.py:206`; `run_memory_eval.py:79`; `run_memory_train.py:804`). Official `MAXIMUM_STEP_LIMIT=20` (`default_prompts.py:1`, `base_handler.py:800`). | `memory_evaluator.py:206,827` vs `base_handler.py:800` | **Moderate.** Episodes force-quit ~5 steps earlier in training; long-but-correct trajectories fail in training that would pass officially. Biases templates toward brevity. | CONFIRMED |
| **4** | **`force_quit` → hard fail.** Evaluator sets `valid=False` whenever `force_quit` (`memory_evaluator.py:879-885`). Official agentic scoring grades the last non-FC message regardless of force-quit (`eval_runner.py:112-148`) — a force-quit episode can still pass if the final message contains the answer. | `memory_evaluator.py:879-885` vs `eval_runner.py` agentic path | **Moderate** (compounds #3). Extra false negatives on borderline-long episodes vs official. | CONFIRMED (evaluator) / SUSPECTED (official never hard-fails agentic on force-quit) |
| **5** | **Injection persistence semantics.** Step-0 (`turn_start`) injections are **ephemeral** — appended to the opening user message then stripped after the query (`memory_evaluator.py:646-668`); `trailing_user` injections persist. Official gate nudges are always **persistent** user turns. | `memory_evaluator.py:639-668` | **Low–Moderate**, only when templates are non-empty. The turn-opening nudge does not accumulate in context like official gate turns; changes multi-step dynamics for storage-heavy prereqs. | CONFIRMED |
| **6** | **Query token accounting.** Evaluator estimates input tokens as `len(prompt)//4` and floors `max_tokens` at 256 (`memory_evaluator.py:238-239`); official uses the real tokenizer count and no floor (`base_oss_handler.py:332-342`). Both cap at 4096, temp 0.001, same endpoint, same `_format_prompt`. Note: omitting `stop_token_ids`/`skip_special_tokens` is a **non-issue** — Granite defines neither. | `memory_evaluator.py:228-285` vs `base_oss_handler.py:322-370` | **Low.** Only bites near the context limit (long rec_sum blobs / long_context), where the char/4 heuristic can mis-budget output length. | CONFIRMED |

### 4.3 Note on the P1–P6 nudges
`default_prompts.py:166-168, 177-178` show P1–P6 are removed from the official memory templates and "now owned by `on_memory_preamble`". Consequence: a **true** official leaderboard run (base_handler with these templates, no AnchorOpt injection) would score the *un-nudged* baseline. For AnchorOpt's optimized preamble/gate text to appear at official eval, `base_handler`/`default_prompts` (and the prereq generation path) must be patched to inject `on_memory_preamble` and to use the optimized gate strings. This is the crux of transfer.

---

## 5. How AnchorOpt injection must align (to make training signal transfer)

1. **Seed with the same policy you score with.** The `on_memory_preamble` (and any gate text) must be applied to **prereq storage episodes** as well as the scored turn, at official eval time, at the same position (prepended to the memory backend instruction). Fix the internal inconsistency first: `run_memory_eval.py:189` should pass the same templates to `build_snapshot_store` that scoring uses (currently it seeds with empty templates → divergence #1 sub-point).
2. **Match the gate set and step budget.** Either add **G3/G4** to `memory_evaluator.py` and set `max_steps_per_turn=20`, or remove G3/G4 and lower `MAXIMUM_STEP_LIMIT` to 15 in the official harness. The fidelity-correct direction is to make the evaluator match `base_handler.py` (add G3/G4, 20 steps). Decide whether force-quit should hard-fail (align #4 accordingly).
3. **Gate text ownership.** The official gate strings are hardcoded (`base_handler.py:631-635, 681, 754-757, 788-789`). If AnchorOpt optimizes `on_core_clear_blocked` / `on_premature_idk` (and G3/G4 equivalents), those must replace the hardcoded strings at official eval or the optimization is invisible to the leaderboard.
4. **Injection persistence.** Confirm whether turn-start nudges should persist; if official gate turns persist, ephemeral training injections under-represent context accumulation (#5).
5. **Non-issues to preserve.** Keep executing via the official `execute_multi_turn_func_call` and grading via the official `agentic_checker` (already faithful); keep snapshots out of `BFCL/eval_checker/`.

---

## Key file:line index
- Episode/turn/step loop: `BFCL/model_handler/base_handler.py:394` (`inference_multi_turn_prompting`), turn loop `:468`, step loop `:522`, force-quit `:800`.
- Gates: G1 `:591-641`, D3 `:585-589/652-687`, G3 `:727-763`, G4 `:765-796`.
- Seeding: `BFCL/utils.py:714` (`process_memory_test_case`), `:837` (`populate_initial_settings_for_memory_test_cases`); `multi_turn_utils.py:13-50` (instantiate/`_load_scenario`); `memory_api_metaclass.py:25` (`_prepare_snapshot`); flush `base_handler.py:847-852`.
- Context assembly: `BFCL/model_handler/utils.py:610` (`add_memory_instruction_system_prompt`); templates `constants/default_prompts.py:79,82-88,157,170`; nudges commented out `:166-168,177-178`.
- Scoring: `BFCL/eval_checker/eval_runner.py:110-159, 421, 501, 730-739`; `agentic_eval/agentic_checker.py:6-49`; ground-truth load `utils.py:453-454`.
- Training evaluator: `AO/anchoropt/memory_evaluator.py` — imports `:105,112,122`; `_query` `:228`; episode loop `:478-889`; gates G1 `:699`, D3 `:730`; force-quit fail `:879`; snapshot store `:355`; grade `:851-889`.
- Run scripts: `AO/scripts/run_memory_eval.py:79,139-144,189`; `AO/scripts/run_memory_train.py:581,686,804,879-885`; cases build `AO/scripts/build_memory_cases.py:72,79,88,101`.
