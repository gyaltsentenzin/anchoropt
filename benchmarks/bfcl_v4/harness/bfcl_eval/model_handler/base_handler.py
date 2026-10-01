import json
from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from bfcl_eval.constants.category_mapping import VERSION_PREFIX
from bfcl_eval.constants.default_prompts import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    MAXIMUM_STEP_LIMIT,
)
from bfcl_eval.constants.enums import ModelStyle, ReturnFormat
from bfcl_eval.constants.eval_config import RESULT_PATH
from bfcl_eval.constants.executable_backend_config import (
    OMIT_STATE_INFO_CLASSES,
    STATELESS_CLASSES,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.memory_gates import (
    RECENT_CALL_SIGNATURE_WINDOW,
    IDK_FALLBACK_THRESHOLD,
    G1_USER_WANTS_CLEAR_PHRASES,
    REGISTRY_BY_KEY,
    archival_memory_searched,
    blob_pressure_threshold,
    find_failing_core_full_call,
    gate_applies,
    gate_enabled,
    gate_fallback,
    gate_match,
    gate_match_any,
    is_error_result,
    is_retrieval_success,
    is_empty_or_failed_retrieval,
    loop_repeat_detected,
    loop_repeat_threshold,
    memory_backend,
    remedy_enabled,
    reprompt_specs,
    reroute_specs,
    seed_text,
    synthesize_core_full_reroute,
    synthesize_forced_key_search_call,
    synthesize_forced_retrieval_call,
    suppress_specs,
    suppress_user_exempt,
)
from bfcl_eval.model_handler.utils import (
    add_memory_instruction_system_prompt,
    anchoropt_template,
    load_anchoropt_policy,
    render_anchoropt_slots,
)
from bfcl_eval.utils import *
from overrides import final

if TYPE_CHECKING:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_api_metaclass import (
        MemoryAPI,
    )


class BaseHandler:
    model_name: str
    is_fc_model: bool
    registry_name: str
    temperature: float
    registry_dir_name: str
    model_name_underline_replaced: str
    model_style: ModelStyle

    def __init__(
        self, model_name, temperature, registry_name, is_fc_model, **kwargs
    ) -> None:
        """
        Args:
            model_name: The name of the model as used in the vendor API or on Hugging Face.
            temperature: The temperature of the model.
            registry_name: The name of the model as used internally in BFCL, used for result directory naming.
            is_fc_model: Whether the model is a function calling model.
            **kwargs: Additional attributes passed via kwargs.
        """
        self.model_name = model_name
        self.is_fc_model = is_fc_model
        self.registry_name = registry_name

        # Replace the dash and dot with underscore for valid variable name
        self.model_name_underline_replaced = (
            model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        )
        # The directory name for the model
        # Replace the slash with underscore to avoid creating subdirectories
        self.registry_dir_name = registry_name.replace("/", "_")
        self.temperature = temperature

        # Set any additional attributes passed via kwargs
        for _key, _value in kwargs.items():
            setattr(self, _key, _value)

    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        # This method is used to retrive model response for each model.

        # FC model
        # TODO: Let all models have the is_fc_model attribute and remove the "FC" check
        if "FC" in self.registry_name or self.is_fc_model:
            if contain_multi_turn_interaction(test_entry["id"]):
                return self.inference_multi_turn_FC(
                    test_entry, include_input_log, exclude_state_log
                )
            else:
                return self.inference_single_turn_FC(test_entry, include_input_log)
        # Prompting model
        else:
            if contain_multi_turn_interaction(test_entry["id"]):
                return self.inference_multi_turn_prompting(
                    test_entry, include_input_log, exclude_state_log
                )
            else:
                return self.inference_single_turn_prompting(test_entry, include_input_log)

    @final
    def inference_multi_turn_FC(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ) -> tuple[list[list], dict]:
        initial_config: dict = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        # This is only for the miss function category
        # A mapping from turn index to function to holdout
        holdout_function: dict[int, list] = test_entry.get("missed_function", {})

        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency: list[list[float]] = []
        all_model_response: list[list] = (
            []
        )  # The model response that will be used for later evaluation
        all_inference_log: list[list[dict]] = (
            []
        )  # The debugging log for human to understand
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed.

        all_reasoning_content: list[list] = []

        # Execute no function call, but just to get a reference to all the instances to get the initial state for logging purpose
        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=("long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )

        if is_memory(test_category):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."

            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"],
                test_category,
                test_entry["scenario"],
                memory_instance,
            )

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
                    continue
                # Avoid modification in future turns
                class_instance = deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if len(state_log) > 0:
                all_inference_log.append(state_log)

        inference_data: dict = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)

        all_multi_turn_messages: list[list[dict]] = test_entry["question"]
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]

            if str(turn_idx) in holdout_function:
                test_entry["function"].extend(holdout_function[str(turn_idx)])
                # Since we have added new functions, we need to recompile the tools
                inference_data = self._compile_tools(inference_data, test_entry)
                assert (
                    len(current_turn_message) == 0
                ), "Holdout turn should not have user message."
                # TODO: Move this to before pre_query_processing_FC.
                # Shouldn't be happening in the inference loop.
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                    }
                ]

            if turn_idx == 0:
                inference_data = self.add_first_turn_message_FC(
                    inference_data, current_turn_message
                )
            else:
                inference_data = self._add_next_turn_user_message_FC(
                    inference_data, current_turn_message
                )

            current_turn_response = []
            current_turn_inference_log: list[dict] = {
                "begin_of_turn_query": current_turn_message
            }
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency: list[float] = []
            current_turn_reasoning_content = []

            count = 0
            while True:
                print("-" * 100)
                print(
                    f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}, Step: {count}"
                )
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                api_response, query_latency = self._query_FC(inference_data)

                # This part of logging is disabled by default because it is too verbose and will make the result file extremely large
                # It is only useful to see if the inference pipeline is working as expected (eg, does it convert all the inputs correctly)
                if include_input_log:
                    current_step_inference_log.append(
                        {
                            "role": "inference_input",
                            "content": inference_data.get("inference_input_log", ""),
                        }
                    )

                # Try parsing the model response
                model_response_data = self._parse_query_response_FC(api_response)
                model_responses = model_response_data["model_responses"]

                # Add the assistant message to the chat history
                inference_data = self._add_assistant_message_FC(
                    inference_data, model_response_data
                )

                # Process the metadata
                current_turn_input_token_count.append(model_response_data["input_token"])
                current_turn_output_token_count.append(model_response_data["output_token"])
                current_turn_latency.append(query_latency)

                current_turn_response.append(model_responses)

                reasoning_content = model_response_data.get("reasoning_content", "")
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {
                    "role": "assistant",
                    "content": model_responses,
                }
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content

                current_step_inference_log.append(log_entry)

                # Try decoding the model response
                try:
                    decoded_model_responses = self.decode_execute(
                        model_responses, has_tool_call_tag=False
                    )
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )

                    if is_empty_execute_response(decoded_model_responses):
                        print("Empty response from the model. Proceed to next turn.")
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": f"Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        break

                except Exception as e:
                    print("Failed to decode the model response. Proceed to next turn.")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Error decoding the model response. Proceed to next turn.",
                            "error": str(e),
                        }
                    )
                    break

                # Obtain the execution results
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # Add the execution results to the chat history for the next turn
                inference_data = self._add_execution_results_FC(
                    inference_data, execution_results, model_response_data
                )

                for execution_result in execution_results:
                    current_step_inference_log.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                        }
                    )

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )

                    break

            # Add to the total list
            all_model_response.append(current_turn_response)
            all_inference_log.append(current_turn_inference_log)
            all_reasoning_content.append(current_turn_reasoning_content)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)

            if not exclude_state_log:
                state_log = []
                for class_name, class_instance in involved_instances.items():
                    if (
                        class_name in STATELESS_CLASSES
                        or class_name in OMIT_STATE_INFO_CLASSES
                    ):
                        continue
                    # Avoid modification in future turns
                    class_instance = deepcopy(class_instance)
                    state_log.append(
                        {
                            "role": "state_info",
                            "class_name": class_name,
                            "content": {
                                key: value
                                for key, value in vars(class_instance).items()
                                if not key.startswith("_")
                            },
                        }
                    )
                if len(state_log) > 0:
                    all_inference_log.append(state_log)

            if force_quit:
                break

        # Special handling for the memory category
        # Need to flush the memory to local file at the end of the conversation
        if is_memory_prereq(test_entry_id):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."
            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            memory_instance._flush_memory_to_local_file()

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
        }

        if not all(
            all(content == "" for content in single_turn_reasoning_content)
            for single_turn_reasoning_content in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content

        return all_model_response, metadata

    @final
    def inference_multi_turn_prompting(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ) -> tuple[list[list], dict]:
        initial_config: dict = test_entry.get("initial_config", {})
        involved_classes: list = test_entry["involved_classes"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        # This is only for the miss function category
        # A mapping from turn index to function to holdout
        holdout_function: dict[int, list] = test_entry.get("missed_function", {})

        total_input_token_count: list[list[float]] = []
        total_output_token_count: list[list[float]] = []
        total_latency: list[list[float]] = []
        # The model response that will be used for later evaluation
        all_model_response: list[list] = []
        # Only for reasoning models, reasoning content will be stored as part of metadata and in inference log
        all_reasoning_content: list[list] = []
        # The debugging log for human to understand
        all_inference_log: list[list[dict]] = []
        force_quit = False  # Whether the model has been forced to quit. If True, this whole entry will be failed.

        # Execute no function call, but just to get a reference to all the instances to get the initial state for logging purpose
        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=("long_context" in test_category or "composite" in test_category),
            is_evaL_run=False,
        )

        if is_memory(test_category):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."

            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            test_entry["question"] = add_memory_instruction_system_prompt(
                test_entry["question"],
                test_category,
                test_entry["scenario"],
                memory_instance,
            )

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
                    continue
                # Avoid modification in future turns
                class_instance = deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if len(state_log) > 0:
                all_inference_log.append(state_log)

        inference_data: dict = self._pre_query_processing_prompting(test_entry)

        # === LOOP GATE: per-EPISODE state (all categories — on_loop is domain="general") ===
        # Rolling window of the most recent NON-ERRORED (tool, args) call signatures,
        # spanning turns exactly like memory_evaluator.py's `recent_call_signatures`
        # deque (declared once per episode, outside the turn loop, maxlen shared via
        # RECENT_CALL_SIGNATURE_WINDOW). This is the first rolling-window/counter state
        # in this method — every other gate's state is a per-turn bool — so it is kept
        # deliberately narrow: ONE local, appended in one place, read in one place.
        # NOT scoped to memory categories: gate_applies returns True unconditionally for
        # domain="general", and nothing here touches a memory backend.
        # To remove: delete this line and the LOOP GATE block below.
        _recent_call_signatures: deque = deque(maxlen=RECENT_CALL_SIGNATURE_WINDOW)
        # === END LOOP GATE INIT ===

        # === IDK FALLBACK GATE: per-EPISODE state (on_idk_fallback, memory KV/Vector only) ===
        # Consecutive failed/empty memory-retrieval count, spanning turns exactly like
        # memory_evaluator.py's `failed_search_streak` (also declared once per episode,
        # outside the turn loop — see that module's docstring note: "The idk_fallback_signal
        # is tracked per-episode as failed_search_streak, reset [on success]"). Updated in
        # the LOOP/IDK STATE UPDATE block below and read by the IDK FALLBACK GATE at the
        # tail of the per-step gate chain.
        # To remove: delete this line and the IDK FALLBACK GATE block below.
        _failed_search_streak = 0
        # === END IDK FALLBACK GATE INIT ===

        all_multi_turn_messages: list[list[dict]] = test_entry["question"]
        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]

            if str(turn_idx) in holdout_function:
                assert (
                    len(current_turn_message) == 0
                ), "Holdout turn should not have user message."
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                            functions=holdout_function[str(turn_idx)]
                        ),
                    }
                ]

            if turn_idx == 0:
                inference_data = self.add_first_turn_message_prompting(
                    inference_data, current_turn_message
                )
            else:
                inference_data = self._add_next_turn_user_message_prompting(
                    inference_data, current_turn_message
                )

            current_turn_response = []
            current_turn_reasoning_content = []
            current_turn_inference_log: list[dict] = {
                "begin_of_turn_query": current_turn_message
            }
            current_turn_input_token_count: list[float] = []
            current_turn_output_token_count: list[float] = []
            current_turn_latency: list[float] = []

            count = 0
            # === D3 GATE: per-turn state (memory KV/Vector only) ===
            # Tracks whether archival memory was searched this turn.
            # Fires once per turn to force archival search before an IDK answer.
            # To remove: delete these two lines and the D3 gate block below.
            _d3_archival_searched = False
            _d3_gate_fired = False
            # === END D3 GATE INIT ===
            # === G1G3 GATE: per-turn state (memory KV/Vector only) ===
            # G1: pre-execution, hard-suppresses core_memory_clear calls.
            # G3: post-execution, injects redirect after "core full" tool error.
            # To remove: delete these two lines and the G1/G3 gate blocks below.
            _core_full_seen = False
            _g1g3_fired = False
            # === END G1G3 GATE INIT ===
            # === G4 GATE: per-turn state (memory KV only) ===
            # Post-execution fallback when exact-match retrieve returns "Key not found".
            # To remove: delete this line and the G4 gate block below.
            _g4_fired = False
            # === END G4 GATE INIT ===
            # === G5/BLOB-PRESSURE GATE: per-turn state (memory rec_sum only) ===
            # G5: post-execution, injects redirect after a blob-overflow tool error.
            # Blob-pressure: post-execution, proactively nudges before the blob overflows.
            # Both opt-in (empty fallback unless a policy sets non-empty text), so they
            # are no-ops on stock — see the two blocks below.
            # To remove: delete these two lines and the two gate blocks below.
            # Reprompt-family latches, keyed by spec (was a single _g5_fired bool
            # when this block hardcoded one gate).
            _rp_fired = set()
            # G3: per-spec latch for the generic recovery-substitution family.
            _rr_fired = set()
            _blob_pressure_fired = False
            # === END G5/BLOB-PRESSURE GATE INIT ===
            while True:
                print("-" * 100)
                print(
                    f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}, Step: {count}"
                )
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                api_response, query_latency = self._query_prompting(inference_data)

                # This part of logging is disabled by default because it is too verbose and will make the result file extremely large
                # It is only useful to see if the inference pipeline is working as expected (eg, does it convert all the inputs correctly)
                if include_input_log:
                    current_step_inference_log.append(
                        {
                            "role": "inference_input",
                            "content": inference_data.get("inference_input_log", ""),
                        }
                    )

                # Try parsing the model response
                model_response_data = self._parse_query_response_prompting(api_response)
                model_responses = model_response_data["model_responses"]

                # Add the assistant message to the chat history
                inference_data = self._add_assistant_message_prompting(
                    inference_data, model_response_data
                )

                # Process the metadata
                current_turn_input_token_count.append(model_response_data["input_token"])
                current_turn_output_token_count.append(model_response_data["output_token"])
                current_turn_latency.append(query_latency)

                current_turn_response.append(model_responses)
                reasoning_content = model_response_data.get("reasoning_content", "")
                current_turn_reasoning_content.append(reasoning_content)

                log_entry = {
                    "role": "assistant",
                    "content": model_responses,
                }
                if reasoning_content:
                    log_entry["reasoning_content"] = reasoning_content

                current_step_inference_log.append(log_entry)

                # Try decoding the model response
                try:
                    decoded_model_responses = self.decode_execute(
                        model_responses, has_tool_call_tag=False
                    )
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )

                    model_response_data["model_responses_decoded"] = decoded_model_responses

                    # === D3 GATE: track archival search from this step's tool calls ===
                    if gate_applies(REGISTRY_BY_KEY["on_premature_idk"], test_category):
                        if archival_memory_searched(model_responses):
                            _d3_archival_searched = True
                    # === END D3 GATE TRACK ===

                    # === G1 GATE: suppress core_memory_clear unless user explicitly requested it ===
                    # Hard-blocks the destructive clear that P1/P3 only probabilistically prevent.
                    # The tool error "Please clear some entries" wins recency over the system prompt;
                    # this gate enforces the P1 rule deterministically before execution.
                    # To remove: delete this block and the _g1g3_fired/_core_full_seen init above.
                    # Generic over GateSpecs: DC ordering is signal -> policy ->
                    # execution, so EXECUTION iterates whatever pre_exec_decoded
                    # suppress specs the registry+policy enable. Mirrors
                    # memory_evaluator.py exactly -- both sites call suppress_specs(),
                    # so a newly learned suppress gate is installable as pure registry
                    # data with no execution change at either site.
                    _sup_text = None
                    for _sup_spec in suppress_specs(load_anchoropt_policy(), test_category):
                        if not gate_match_any(_sup_spec, decoded_model_responses):
                            continue
                        # Per-gate user-intent exemption (G1 skips when the user asked
                        # to forget everything). Keyed by gate, NOT inherited by other
                        # suppress gates whose semantics differ.
                        if suppress_user_exempt(_sup_spec.key, str(current_turn_message)):
                            continue
                        _n_before = len(decoded_model_responses)
                        # ---- PER-SPEC PREDICATE, mirrored from memory_evaluator.py --------------
                        # This site filters by NAME ALONE, which is correct for gates whose call is
                        # unconditionally wrong (G1's core_memory_clear). It is WRONG for a gate whose
                        # call is only sometimes wrong.
                        #
                        # N0-R1 exposed the hazard concretely. Its predicate lived only in
                        # memory_evaluator.py, and PREREQ episodes run through THIS site -- so the
                        # controller was inert: 28 destructive calls executed unchanged with only
                        # OPEN/CLOSE telemetry recorded. Had the name-only filter fired instead, it
                        # would have suppressed EVERY destructive call unconditionally, which is not
                        # the bounded controller at all but a blanket ban. Both failure modes come from
                        # the same gap, and both make the arm uninterpretable.
                        #
                        # So a predicate-bearing spec must be evaluated identically here.
                        _sup_pred = None
                        if _sup_spec.key == "on_destructive_authorization":
                            try:
                                import sys as _sR
                                from pathlib import Path as _PathR
                                _dR = "/path/to/remote-checkout/scripts"
                                if _dR not in _sR.path:
                                    _sR.path.insert(0, _dR)
                                import recovery_txn as _rtR
                                _instsR = list((involved_instances or {}).values()) \
                                    if isinstance(involved_instances, dict) else (involved_instances or [])
                                if not hasattr(self, "_r1_txn_bh"):
                                    self._r1_txn_bh = {}
                                _obsR = []

                                def _sup_pred(_call, _rt=_rtR, _ii=_instsR, _obs=_obsR):
                                    _cs = str(_call)
                                    _wh = _rt.container_of(_cs)
                                    _store = _rt.live_store(_ii, _wh)
                                    _v, _o = _rt.diagnose(_cs, _store, _wh)
                                    _txn = self._r1_txn_bh.get(_wh)
                                    if _txn is None:
                                        _obs.append("No pending memory-capacity failure. " + _o)
                                        return True
                                    _txn.proposals += 1
                                    _txn.verdicts.append(_v)
                                    if _v in ("D1", "D2", "D3"):
                                        _obs.append(_o)
                                        return True
                                    if "clear" in _cs:
                                        _obs.append(_o + " A clear would destroy every entry; "
                                                    "remove a single entry instead.")
                                        return True
                                    if _txn.authorized >= 1:
                                        _obs.append(_o + " One entry has already been removed for this "
                                                    "write; no further removal is authorized.")
                                        return True
                                    _txn.authorized += 1
                                    return False
                            except Exception:
                                _sup_pred = None

                        def _sup_hits(_c, _spec=_sup_spec, _p=_sup_pred):
                            if not gate_match(_spec, _c):
                                return False
                            return _p(_c) if _p is not None else True

                        decoded_model_responses = [
                            c for c in decoded_model_responses
                            if not _sup_hits(c)
                        ]
                        if len(decoded_model_responses) == _n_before:
                            continue
                        model_response_data["model_responses_decoded"] = decoded_model_responses
                        current_step_inference_log.append({
                            "role": "handler_log",
                            "content": (
                                f"{_sup_spec.key}: suppressed "
                                f"{_n_before - len(decoded_model_responses)} call(s) "
                                f"matching {_sup_spec.match_substrings}."
                            ),
                        })
                        _t = anchoropt_template(_sup_spec.key, gate_fallback(_sup_spec.key))
                        if _t:
                            _sup_text = _t
                    # If suppression left no executable calls, inject redirect and retry
                    # so the turn doesn't silently end without storing the fact.
                    if (
                        _sup_text is not None
                        and is_empty_execute_response(decoded_model_responses)
                        and count < MAXIMUM_STEP_LIMIT
                    ):
                        inference_data = self._add_next_turn_user_message_prompting(
                            inference_data,
                            [{"role": "user", "content": _sup_text}],
                        )
                        count += 1
                        continue
                    # === END SUPPRESS GATES ===

                    if is_empty_execute_response(decoded_model_responses):
                        print("Empty response from the model. Proceed to next turn.")
                        current_step_inference_log.append(
                            {
                                "role": "handler_log",
                                "content": f"Empty response from the model. Proceed to next turn.",
                                "model_response_decoded": decoded_model_responses,
                            }
                        )
                        # === D3 GATE: force archival search before accepting IDK answer ===
                        # Fires once per turn when: memory KV/Vector task, archival not yet
                        # searched this turn, model response looks like an IDK answer.
                        # To remove: delete this block and restore plain `break`.
                        if (
                            gate_applies(REGISTRY_BY_KEY["on_premature_idk"], test_category)
                            and gate_enabled(load_anchoropt_policy(), "on_premature_idk")
                            and not _d3_archival_searched
                            and not _d3_gate_fired
                            and count < MAXIMUM_STEP_LIMIT
                            and gate_match_any(REGISTRY_BY_KEY["on_premature_idk"], [str(model_responses).lower()])
                        ):
                            _d3_gate_fired = True
                            current_step_inference_log.append(
                                {
                                    "role": "handler_log",
                                    "content": "D3 gate: archival memory not searched — forcing retrieval before IDK.",
                                }
                            )

                            # Phase 6 (opt-in via enable_forced_retrieval): instead of only
                            # asking the model to search (which it already ignores once,
                            # in the standing system prompt), execute the search for it.
                            _forced_result = None
                            if remedy_enabled(load_anchoropt_policy(), "enable_forced_retrieval"):
                                _forced_result = self._try_forced_archival_retrieval(
                                    test_category,
                                    initial_config,
                                    involved_classes,
                                    test_entry_id,
                                    "long_context" in test_category or "composite" in test_category,
                                )

                            if _forced_result is not None:
                                current_step_inference_log.append({
                                    "role": "handler_log",
                                    "content": "D3 gate (forced retrieval): dispatched archival dump on the model's behalf.",
                                })
                                current_step_inference_log.append({
                                    "role": "tool",
                                    "content": _forced_result,
                                })
                                _d3_content = anchoropt_template(
                                    "on_forced_archival_retrieval",
                                    gate_fallback("on_forced_archival_retrieval"),
                                )
                            else:
                                # W2c: optimized text from the AnchorOpt policy if configured, else the registry's default fallback.
                                _d3_content = anchoropt_template(
                                    "on_premature_idk",
                                    gate_fallback("on_premature_idk"),
                                )
                            inference_data = self._add_next_turn_user_message_prompting(
                                inference_data,
                                [{"role": "user", "content": _d3_content}],
                            )
                            count += 1
                            continue  # force one more retrieval step; skip tool execution below
                        # === END D3 GATE ===
                        break

                except Exception as e:
                    print("Failed to decode the model response. Proceed to next turn.")
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Error decoding the model response. Proceed to next turn.",
                            "error": str(e),
                        }
                    )
                    break

                # Obtain the execution results
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name_underline_replaced,
                    test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                    is_evaL_run=False,
                )

                # ---- N0-R1: OPEN / CLOSE the recovery transaction (mirrored) ----------------
                # The predicate above CONSUMES self._r1_txn_bh; this populates it. Without this the
                # dict stays empty, every proposal reads as an orphan, and the controller degenerates
                # into a BLANKET BAN on destruction -- the opposite failure from the inert one that
                # killed an earlier measured run, and equally uninterpretable.
                #
                # CLOSE fires only on a successful retry of the ORIGINAL value: of 67 apparent retry
                # successes measured offline, only 40 re-attempted the same value.
                try:
                    import sys as _sO
                    _dO = "/path/to/remote-checkout/scripts"
                    if _dO not in _sO.path:
                        _sO.path.insert(0, _dO)
                    import recovery_txn as _rtO
                    if not hasattr(self, "_r1_txn_bh"):
                        self._r1_txn_bh = {}
                    _instsO = list((involved_instances or {}).values()) \
                        if isinstance(involved_instances, dict) else (involved_instances or [])
                    for _oi, _oc in enumerate(decoded_model_responses or []):
                        _ocs = str(_oc)
                        if not _rtO.ADD_RE.search(_ocs):
                            continue
                        _orr = str(execution_results[_oi]) if _oi < len(execution_results or []) else ""
                        _owh = _rtO.container_of(_ocs)
                        _ov = _rtO.VALARG.search(_ocs)
                        _ovc = _rtO.canon(_ov.group(2)) if _ov else ""
                        if "error" in _orr.lower() and _rtO.FULL_RE.search(_orr):
                            if _owh not in self._r1_txn_bh:
                                _lv = _rtO.live_store(_instsO, _owh)
                                self._r1_txn_bh[_owh] = _rtO.RecoveryTransaction(
                                    _owh, len(_lv), _ovc)
                        elif "error" not in _orr.lower():
                            _tx = self._r1_txn_bh.get(_owh)
                            if _tx is not None and _ovc and _ovc == _tx.pending_value:
                                del self._r1_txn_bh[_owh]
                except Exception:
                    pass

                # Add the execution results to the chat history for the next turn
                inference_data = self._add_execution_results_prompting(
                    inference_data, execution_results, model_response_data
                )

                for execution_result in execution_results:
                    current_step_inference_log.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                        }
                    )

                # === LOOP/IDK STATE UPDATE: capture this step's outcome (all categories) ===
                # Mirrors memory_evaluator.py's "Update AnchorOpt state after execution"
                # block placement exactly: state consumed by the LOOP GATE and IDK
                # FALLBACK GATE below is captured HERE, immediately after execution and
                # BEFORE G3/G4/G5/blob-pressure — not inside those later gate blocks —
                # because each of those gates ends in `continue` on firing, which would
                # otherwise skip this step's contribution to the signature deque / streak
                # counter entirely. The evaluator updates last_result/
                # recent_call_signatures/failed_search_streak unconditionally right after
                # execution, before its own G3/G4/G5/blob checks, so this step's outcome
                # must be captured the same way here regardless of which (if any)
                # structural gate fires afterward.
                # To remove: delete this block along with the LOOP GATE and IDK FALLBACK
                # GATE blocks below (all three share this state).
                _step_errored = any(is_error_result(r) for r in execution_results)
                if not _step_errored:
                    for _call in decoded_model_responses:
                        _recent_call_signatures.append(str(_call).strip())
                _idk_last_call_str = decoded_model_responses[0] if decoded_model_responses else ""
                _idk_last_result = execution_results[-1] if execution_results else ""
                if is_empty_or_failed_retrieval(_idk_last_result, _idk_last_call_str):
                    _failed_search_streak += 1
                elif is_retrieval_success(_idk_last_result):
                    _failed_search_streak = 0
                # === END LOOP/IDK STATE UPDATE ===

                # === G3 GATE: inject corrective after "core full" tool error ===
                # Fires once per turn when the executor returns a "core/archival is full" error.
                # Injects the redirect AFTER the hostile tool message so it wins the recency battle
                # that defeats P1/P3. Pairs with G1 (G1 blocks the clear; G3 redirects to archival).
                # Phase 3 (opt-in via enable_reroute, default off): instead of only telling the
                # model to use archival_memory_add, synthesize the corrected call and re-dispatch
                # it through the same executor so it mutates the live cached instance identically
                # to a model call. Falls back to the reprompt text below whenever the reroute
                # can't be synthesized/dispatched — the fact is never silently dropped.
                # To remove: delete this block and the _g1g3_fired/_core_full_seen init above.
                if (
                    gate_applies(REGISTRY_BY_KEY["on_domain_error_core_full"], test_category)
                    and gate_enabled(load_anchoropt_policy(), "on_domain_error_core_full")
                    and not _g1g3_fired
                    and count < MAXIMUM_STEP_LIMIT
                ):
                    _full_now = gate_match_any(REGISTRY_BY_KEY["on_domain_error_core_full"], execution_results)
                    if _full_now:
                        _core_full_seen = True
                        _g1g3_fired = True

                        _reroute_result = None
                        if remedy_enabled(load_anchoropt_policy(), "enable_reroute"):
                            _reroute_result = self._try_core_full_reroute(
                                decoded_model_responses,
                                execution_results,
                                involved_instances,
                                test_category,
                                initial_config,
                                involved_classes,
                                test_entry_id,
                                "long_context" in test_category or "composite" in test_category,
                            )

                        if _reroute_result is not None:
                            current_step_inference_log.append({
                                "role": "handler_log",
                                "content": "G3 gate (reroute): core/archival full — re-dispatched fact via archival_memory_add.",
                            })
                            current_step_inference_log.append({
                                "role": "tool",
                                "content": _reroute_result,
                            })
                            inference_data = self._add_next_turn_user_message_prompting(
                                inference_data,
                                [{
                                    "role": "user",
                                    "content": anchoropt_template(
                                        "on_core_full_rerouted",
                                        gate_fallback("on_core_full_rerouted"),
                                    ),
                                }],
                            )
                        else:
                            current_step_inference_log.append({
                                "role": "handler_log",
                                "content": "G3 gate: core/archival full error — injecting overflow redirect.",
                            })
                            inference_data = self._add_next_turn_user_message_prompting(
                                inference_data,
                                [{
                                    "role": "user",
                                    # W2c: optimized text from the AnchorOpt policy if configured, else the registry's default fallback.
                                    "content": anchoropt_template(
                                        "on_domain_error_core_full",
                                        gate_fallback("on_domain_error_core_full"),
                                    ),
                                }],
                            )
                        count += 1
                        continue
                # === END G3 GATE ===

                # === G4 GATE: key-search fallback on "Key not found" (KV only) ===
                # Fills D3's blind spot: D3 marks archival "searched" even after a failed exact-match
                # retrieve (archival_memory_retrieve / core_memory_retrieve → "Key not found").
                # G4 fires once per turn on that error and redirects to key_search / list_keys
                # so the model can discover the correct stored key before answering.
                # Phase 7 (opt-in via enable_forced_key_search, default off): instead of only
                # telling the model to list the keys, execute archival_memory_list_keys() for
                # it and show the result — G4's analogue of D3's forced retrieval and G3's
                # reroute. Falls back to the reprompt text below whenever the dispatch fails.
                # To remove: delete this block and the _g4_fired init above.
                if (
                    gate_applies(REGISTRY_BY_KEY["on_domain_error_key_not_found"], test_category)
                    and gate_enabled(load_anchoropt_policy(), "on_domain_error_key_not_found")
                    and not _g4_fired
                    and count < MAXIMUM_STEP_LIMIT
                    and gate_match_any(REGISTRY_BY_KEY["on_domain_error_key_not_found"], execution_results)
                ):
                    _g4_fired = True

                    _key_search_result = None
                    if remedy_enabled(load_anchoropt_policy(), "enable_forced_key_search"):
                        _key_search_result = self._try_forced_key_search(
                            test_category,
                            initial_config,
                            involved_classes,
                            test_entry_id,
                            "long_context" in test_category or "composite" in test_category,
                        )

                    if _key_search_result is not None:
                        current_step_inference_log.append({
                            "role": "handler_log",
                            "content": "G4 gate (forced key search): dispatched archival_memory_list_keys() on the model's behalf.",
                        })
                        current_step_inference_log.append({
                            "role": "tool",
                            "content": _key_search_result,
                        })
                        _g4_content = anchoropt_template(
                            "on_forced_key_search",
                            gate_fallback("on_forced_key_search"),
                        )
                    else:
                        current_step_inference_log.append({
                            "role": "handler_log",
                            "content": "G4 gate: retrieve returned 'Key not found' — injecting key-search fallback.",
                        })
                        # W2c: optimized text from the AnchorOpt policy if configured, else the registry's default fallback.
                        _g4_content = anchoropt_template(
                            "on_domain_error_key_not_found",
                            gate_fallback("on_domain_error_key_not_found"),
                        )
                    inference_data = self._add_next_turn_user_message_prompting(
                        inference_data,
                        [{"role": "user", "content": _g4_content}],
                    )
                    count += 1
                    continue
                # === END G4 GATE ===

                # === GENERIC RECOVERY SUBSTITUTION (G3, §40) ===
                # Dispatch a MINED substitute as the next recovery action. Mirrors
                # memory_evaluator.py so a learned reroute gate cannot fire in one
                # pipeline and be inert in the other -- the mirrored-executor drift this
                # file warns about elsewhere.
                #
                # Ordered BEFORE the reprompt family: substituting acts, reprompting only
                # asks. A failed synthesis falls through so the write is never dropped.
                # FLAG-GUARDED, like every other _try_core_full_reroute call site. The
                # synthesizer is the opt-in reroute remedy; a learned gate selects WHEN it
                # fires, never whether the remedy itself is enabled. Without this, installing
                # the G3 gate would dispatch substitutions even with reroute switched off.
                _rr_injected = False
                for _rr_spec in reroute_specs(load_anchoropt_policy(), test_category):
                    if count >= MAXIMUM_STEP_LIMIT:
                        break
                    if _rr_spec.once_per_turn and _rr_spec.key in _rr_fired:
                        continue
                    if not gate_match_any(_rr_spec, execution_results):
                        continue
                    _rr_out = None
                    # The flag name appears IN the test so the guard is verifiable by the
                    # AST invariant, not just by a variable that happens to hold it.
                    if remedy_enabled(load_anchoropt_policy(), "enable_reroute"):
                        _rr_out = self._try_core_full_reroute(
                            decoded_calls, execution_results, involved_instances,
                            test_category, initial_config, involved_classes,
                        )
                    if _rr_out is None:
                        continue
                    _rr_fired.add(_rr_spec.key)
                    current_step_inference_log.append({
                        "role": "handler_log",
                        "content": (f"{_rr_spec.key}: recovery substitution dispatched "
                                    f"after its error contract."),
                    })
                    if isinstance(_rr_out, dict):
                        _rr_out = _rr_out.get("result", "")
                    execution_results = list(execution_results) + [_rr_out]
                    count += 1
                    _rr_injected = True
                    break
                if _rr_injected:
                    continue
                # === END GENERIC RECOVERY SUBSTITUTION ===

                # === REPROMPT FAMILY (was: G5 GATE, rec_sum-only) ===
                # Generic over the registry via reprompt_specs(), mirroring
                # memory_evaluator.py. Previously this hardcoded
                # "on_domain_error_too_long" at five points, so a newly learned reprompt
                # gate fired in the evaluator pipeline and was INERT here -- the
                # mirrored-executor drift this file warns about elsewhere.
                #
                # G5 semantics are unchanged: it is still in reprompt_specs(), still
                # rec_sum-scoped by its own backends/match_substrings, still once/turn,
                # and still resolves policy text -> seed_text -> gate_fallback in that
                # order. The latch is now per-spec so one gate firing cannot suppress
                # another's.
                # To remove: delete this block and the _rp_fired init above.
                _rp_injected = False
                for _rp_spec in reprompt_specs(load_anchoropt_policy(), test_category):
                    if count >= MAXIMUM_STEP_LIMIT:
                        break
                    if _rp_spec.once_per_turn and _rp_spec.key in _rp_fired:
                        continue
                    if not gate_match_any(_rp_spec, execution_results):
                        continue
                    _rp_text = anchoropt_template(
                        _rp_spec.key,
                        seed_text(_rp_spec.key) or gate_fallback(_rp_spec.key),
                    )
                    if _rp_text.strip():
                        _rp_fired.add(_rp_spec.key)
                        current_step_inference_log.append({
                            "role": "handler_log",
                            "content": (f"{_rp_spec.key}: reprompt gate fired on its "
                                        f"error contract — injecting retry redirect."),
                        })
                        inference_data = self._add_next_turn_user_message_prompting(
                            inference_data,
                            [{"role": "user", "content": _rp_text}],
                        )
                        count += 1
                        _rp_injected = True
                        break
                if _rp_injected:
                    continue
                # === END REPROMPT FAMILY ===

                # === BLOB PRESSURE GATE: proactive rec_sum compaction nudge ===
                # Fires once per turn when the live rec_sum blob is already close to
                # MAX_MEMORY_ENTRY_LENGTH, before any call has failed.
                # Phase 7: the trigger POINT is now policy-read via
                # blob_pressure_threshold() (default = BLOB_PRESSURE_THRESHOLD, so an
                # unconfigured policy keeps the historical 8000), and the registry now
                # ships real fallback text, so this gate is no longer an inert no-op.
                # To remove: delete this block and the _blob_pressure_fired init above.
                if (
                    gate_applies(REGISTRY_BY_KEY["on_blob_pressure"], test_category)
                    and gate_enabled(load_anchoropt_policy(), "on_blob_pressure")
                    and not _blob_pressure_fired
                    and count < MAXIMUM_STEP_LIMIT
                ):
                    _mem_inst = list(involved_instances.values())[0]
                    _blob_len = len(getattr(_mem_inst, "memory", "") or "")
                    if _blob_len >= blob_pressure_threshold(load_anchoropt_policy()):
                        _bp_text = anchoropt_template(
                            "on_blob_pressure",
                            gate_fallback("on_blob_pressure"),
                        )
                        if _bp_text.strip():
                            _blob_pressure_fired = True
                            current_step_inference_log.append({
                                "role": "handler_log",
                                "content": (
                                    f"Blob-pressure gate: rec_sum blob at {_blob_len} chars — "
                                    "injecting proactive compaction nudge."
                                ),
                            })
                            inference_data = self._add_next_turn_user_message_prompting(
                                inference_data,
                                [{"role": "user", "content": _bp_text}],
                            )
                            count += 1
                            continue
                # === END BLOB PRESSURE GATE ===

                # === LOOP GATE: repeated-successful-call nudge (all categories) ===
                # Phase 7, first wiring of on_loop into this pipeline. Mirrors
                # memory_evaluator.py's `_loop_signal`: the newest non-errored (tool, args)
                # signature has already been seen loop_repeat_threshold() times inside the
                # last RECENT_CALL_SIGNATURE_WINDOW non-errored calls, AND this step's own
                # execution did not error (a failed step is on_domain_error's / G3-G5's
                # business, not a loop). Threshold default 1 == the historical "fires on the
                # 2nd occurrence" trigger point. once_per_turn=False in the registry, so
                # there is no per-turn latch here either — a second repeat nudges again,
                # matching the evaluator's soft ladder, and MAXIMUM_STEP_LIMIT bounds it.
                # `_step_errored`/`_recent_call_signatures` are read here, not computed here
                # — see the LOOP/IDK STATE UPDATE block above, which captures them
                # unconditionally right after execution so an earlier structural gate's
                # `continue` (G1/D3/G3/G4/G5/blob-pressure) can never skip this step's
                # contribution to the deque.
                # Placed LAST (among the INJECTION decisions, not the state capture) so a
                # structural memory gate that already consumed this step wins, and because
                # every one of those triggers on an errored result, which suppresses this
                # gate anyway.
                # To remove: delete this block and the LOOP GATE INIT above.
                #
                # Known residual divergence from memory_evaluator.py (accepted, not fixed):
                # at count==0 of a turn, the evaluator's loop_signal is read BEFORE this
                # turn's own first call is appended to the signature deque (so it reflects
                # only cross-turn carryover) and, if it fires, injects ephemerally into the
                # turn-opening message (stripped right after that one query — see
                # memory_evaluator.py's step_count==0 branch). Here, `_step_errored` and the
                # signature append in the LOOP/IDK STATE UPDATE block above both already
                # reflect count==0's OWN call (that block runs post-execution, before this
                # gate), and a fired injection is always a permanent trailing message. The
                # two pipelines therefore agree for every step_count>=1 (both gate on "the
                # step that just completed," confirmed by inspection), but can diverge in
                # the rare case where a loop signal is carried over from the tail of the
                # PREVIOUS turn into this turn's very first step. Fixing this exactly would
                # require hoisting the loop check to before the turn's first query and
                # adding persistent cross-turn last_result state — deferred as low-value
                # relative to the rest of Phase 2.
                #
                # Phase 3: excludes a repeated retrieval call that keeps coming back
                # empty/failed from this signal — mirrors memory_evaluator.py's
                # `_loop_signal` fix exactly. Without this, a repeated list_keys()
                # returning "keys": [] tripped on_loop (top ladder priority) instead
                # of on_idk_fallback, telling the model "this already succeeded, don't
                # repeat it" on the one step where it should broaden its search.
                # `_idk_last_result`/`_idk_last_call_str` are this step's OWN values
                # (set in the LOOP/IDK STATE UPDATE block above, same iteration), so
                # this reads "was the call that was just repeated itself a failed
                # search" — the same semantic the evaluator's carried-over
                # last_result/last_call_str check applies one step later.
                #
                # Phase 4: on_loop is the one soft key with a registry gate_fallback()
                # injected regardless of policy text, so it gets an explicit off switch
                # via MASKABLE_GATE_KEYS/gate_enabled — mirrors gating loop_signal
                # itself in memory_evaluator.py, which covers this pipeline's single
                # injection surface for the key.
                if (
                    gate_applies(REGISTRY_BY_KEY["on_loop"], test_category)
                    and gate_enabled(load_anchoropt_policy(), "on_loop")
                    and not _step_errored
                    and not is_empty_or_failed_retrieval(_idk_last_result, _idk_last_call_str)
                    and count < MAXIMUM_STEP_LIMIT
                    and loop_repeat_detected(
                        _recent_call_signatures,
                        loop_repeat_threshold(load_anchoropt_policy()),
                    )
                ):
                    _loop_text = anchoropt_template("on_loop", gate_fallback("on_loop"))
                    # Transfer-contract fix: memory_evaluator.py renders this key's policy
                    # text through render_template (slot substitution), so a trained/edited
                    # on_loop template referencing {last_result}/{error} must resolve the
                    # same way here — otherwise the literal placeholder text would reach the
                    # model. gate_fallback("on_loop") itself has no slots, so this is a no-op
                    # on the untrained/null path.
                    _loop_last_result = execution_results[-1] if execution_results else ""
                    _loop_last_result = (_loop_last_result or "")[:200] + (
                        "…" if len(_loop_last_result or "") > 200 else ""
                    )
                    _loop_text = render_anchoropt_slots(
                        _loop_text,
                        last_result=_loop_last_result,
                        error=_loop_last_result,
                    )
                    if _loop_text.strip():
                        current_step_inference_log.append({
                            "role": "handler_log",
                            "content": (
                                "Loop gate: identical successful call repeated "
                                f"{loop_repeat_threshold(load_anchoropt_policy())}x — "
                                "injecting break-the-loop nudge."
                            ),
                        })
                        inference_data = self._add_next_turn_user_message_prompting(
                            inference_data,
                            [{"role": "user", "content": _loop_text}],
                        )
                        count += 1
                        continue
                # === END LOOP GATE ===

                # === IDK FALLBACK GATE: broaden-the-search nudge (memory KV/Vector only) ===
                # Phase 2: this key previously existed only in the registry and the
                # training evaluator — the registry comment used to read "Evaluator-only
                # guard; base_handler never fires this key." Wires it in for real so an
                # accepted edit to on_idk_fallback is not training-only. Mirrors
                # memory_evaluator.py's `idk_fallback_signal = failed_search_streak >=
                # _IDK_FALLBACK_THRESHOLD`: `_failed_search_streak` is captured in the
                # LOOP/IDK STATE UPDATE block above (unconditional, same placement
                # rationale as the loop gate's signatures). fallback="" in the registry,
                # so on the untrained/null policy this key stays silent — the seed library
                # (on-enable text) is what gives it working text, matching G1/G3/G4's
                # gate_fallback() pattern with an empty-by-default fallback instead.
                # Placed AFTER on_loop. Historically this ordering decided which signal won
                # when both fired on the same step (a repeated empty list_keys() tripped
                # loop_repeat_detected AND the failed-search streak at once); Phase 3's
                # exclusion above (the loop gate backs off on
                # is_empty_or_failed_retrieval) means that case can no longer occur — the
                # two signals are now mutually exclusive on any given step. Ordering kept
                # anyway since it still matches template_engine.py's loop-before-idk
                # priority, in case a future signal change reintroduces overlap.
                # To remove: delete this block and the IDK FALLBACK GATE INIT above.
                if (
                    gate_applies(REGISTRY_BY_KEY["on_idk_fallback"], test_category)
                    and count < MAXIMUM_STEP_LIMIT
                    and _failed_search_streak >= IDK_FALLBACK_THRESHOLD
                ):
                    _idk_text = anchoropt_template("on_idk_fallback", gate_fallback("on_idk_fallback"))
                    _idk_text = render_anchoropt_slots(
                        _idk_text,
                        failed_search_streak=str(_failed_search_streak),
                    )
                    if _idk_text.strip():
                        current_step_inference_log.append({
                            "role": "handler_log",
                            "content": (
                                f"IDK fallback gate: {_failed_search_streak} consecutive "
                                "failed/empty searches — injecting broaden-the-search nudge."
                            ),
                        })
                        inference_data = self._add_next_turn_user_message_prompting(
                            inference_data,
                            [{"role": "user", "content": _idk_text}],
                        )
                        count += 1
                        continue
                # === END IDK FALLBACK GATE ===

                count += 1
                # Force quit after too many steps
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )
                    break

            # Add to the total list
            all_model_response.append(current_turn_response)
            all_reasoning_content.append(current_turn_reasoning_content)
            all_inference_log.append(current_turn_inference_log)
            total_input_token_count.append(current_turn_input_token_count)
            total_output_token_count.append(current_turn_output_token_count)
            total_latency.append(current_turn_latency)

            if not exclude_state_log:
                state_log = []
                for class_name, class_instance in involved_instances.items():
                    if (
                        class_name in STATELESS_CLASSES
                        or class_name in OMIT_STATE_INFO_CLASSES
                    ):
                        continue
                    # Avoid modification in future turns
                    class_instance = deepcopy(class_instance)
                    state_log.append(
                        {
                            "role": "state_info",
                            "class_name": class_name,
                            "content": {
                                key: value
                                for key, value in vars(class_instance).items()
                                if not key.startswith("_")
                            },
                        }
                    )
                if len(state_log) > 0:
                    all_inference_log.append(state_log)

            if force_quit:
                break

        # Special handling for the memory category
        # Need to flush the memory to local file at the end of the conversation
        if is_memory_prereq(test_entry_id):
            assert (
                len(involved_instances) == 1
            ), "Memory category should only involve one class."
            memory_instance: "MemoryAPI" = list(involved_instances.values())[0]
            memory_instance._flush_memory_to_local_file()

        metadata = {
            "input_token_count": total_input_token_count,
            "output_token_count": total_output_token_count,
            "latency": total_latency,
            "inference_log": all_inference_log,
        }
        # We only include reasoning content if it exists and is not empty
        if not all(
            all(content == "" for content in single_turn_reasoning_content)
            for single_turn_reasoning_content in all_reasoning_content
        ):
            metadata["reasoning_content"] = all_reasoning_content

        return all_model_response, metadata

    def _try_core_full_reroute(
        self,
        decoded_model_responses: list,
        execution_results: list,
        involved_instances: dict,
        test_category: str,
        initial_config: dict,
        involved_classes: list,
        test_entry_id: str,
        long_context: bool,
    ) -> "Optional[str]":
        """Phase 3 (opt-in via enable_reroute): synthesize a corrected
        `archival_memory_add(...)` call from the failing `core_memory_add(...)` call
        that just returned a core/archival-full error, and re-dispatch it through the
        SAME executor used for model-issued calls so it mutates the same cached live
        instance and is flushed identically at end-of-conversation.

        Returns the (stringified) execution result of the re-dispatched call on
        success, or None if the reroute could not be synthesized or the re-dispatch
        itself failed — callers must fall back to the existing G3 reprompt text in
        that case (the fact is never silently dropped).
        """
        failing_call = find_failing_core_full_call(decoded_model_responses, execution_results)
        if failing_call is None:
            return None

        backend = memory_backend(test_category)
        mem_inst = list(involved_instances.values())[0]
        synth_call = synthesize_core_full_reroute(failing_call, mem_inst, backend)
        if synth_call is None:
            return None

        reroute_results, _ = execute_multi_turn_func_call(
            [synth_call],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=long_context,
            is_evaL_run=False,
        )
        reroute_result = reroute_results[0] if reroute_results else ""
        if not reroute_result or is_error_result(reroute_result):
            return None
        return reroute_result

    def _try_forced_archival_retrieval(
        self,
        test_category: str,
        initial_config: dict,
        involved_classes: list,
        test_entry_id: str,
        long_context: bool,
    ) -> "Optional[str]":
        """Phase 6 (opt-in via enable_forced_retrieval): deterministically execute an
        archival dump (list_keys for kv, retrieve_all for vector) on the model's
        behalf when it's about to give up without ever having searched archival this
        turn. Unlike the reroute remedy, no failing call to parse and no live-instance
        capacity introspection is needed — both target calls are static and zero-arg.

        Returns the stringified result, or None if the backend is unsupported or the
        dispatch failed — callers must fall back to the existing on_premature_idk text
        in that case.
        """
        backend = memory_backend(test_category)
        synth_call = synthesize_forced_retrieval_call(backend)
        if synth_call is None:
            return None

        results, _ = execute_multi_turn_func_call(
            [synth_call],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=long_context,
            is_evaL_run=False,
        )
        result = results[0] if results else ""
        if not result or is_error_result(result):
            return None
        return result

    def _try_forced_key_search(
        self,
        test_category: str,
        initial_config: dict,
        involved_classes: list,
        test_entry_id: str,
        long_context: bool,
    ) -> "Optional[str]":
        """Phase 7 (opt-in via enable_forced_key_search): after a KV retrieve failed with
        "Key not found", execute archival_memory_list_keys() on the model's behalf so it
        can see which keys actually exist, instead of only being told to look them up.

        Structurally identical to _try_forced_archival_retrieval (static zero-arg call,
        no failing call to parse, no capacity introspection) — kv only, matching G4's
        backend scope. Returns the stringified result, or None if the backend is
        unsupported or the dispatch failed, in which case callers must fall back to the
        existing on_domain_error_key_not_found text.
        """
        synth_call = synthesize_forced_key_search_call(memory_backend(test_category))
        if synth_call is None:
            return None

        results, _ = execute_multi_turn_func_call(
            [synth_call],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            test_entry_id,
            long_context=long_context,
            is_evaL_run=False,
        )
        result = results[0] if results else ""
        if not result or is_error_result(result):
            return None
        return result

    @final
    def inference_single_turn_FC(
        self, test_entry: dict, include_input_log: bool
    ) -> tuple[any, dict]:
        inference_data: dict = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)
        inference_data = self.add_first_turn_message_FC(
            inference_data, test_entry["question"][0]
        )

        api_response, query_latency = self._query_FC(inference_data)

        # Try parsing the model response
        model_response_data = self._parse_query_response_FC(api_response)

        # Process the metadata
        metadata = {}
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": inference_data.get("inference_input_log", ""),
                }
            ]
        metadata["input_token_count"] = model_response_data["input_token"]
        metadata["output_token_count"] = model_response_data["output_token"]
        metadata["latency"] = query_latency

        if (
            "reasoning_content" in model_response_data
            and model_response_data["reasoning_content"] != ""
        ):
            metadata["reasoning_content"] = model_response_data["reasoning_content"]

        return model_response_data["model_responses"], metadata

    @final
    def inference_single_turn_prompting(
        self, test_entry: dict, include_input_log: bool
    ) -> tuple[any, dict]:
        inference_data: dict = self._pre_query_processing_prompting(test_entry)
        inference_data = self.add_first_turn_message_prompting(
            inference_data, test_entry["question"][0]
        )

        api_response, query_latency = self._query_prompting(inference_data)

        # Try parsing the model response
        model_response_data = self._parse_query_response_prompting(api_response)

        # Process the metadata
        metadata = {}
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": inference_data.get("inference_input_log", ""),
                }
            ]
        metadata["input_token_count"] = model_response_data["input_token"]
        metadata["output_token_count"] = model_response_data["output_token"]
        metadata["latency"] = query_latency

        if (
            "reasoning_content" in model_response_data
            and model_response_data["reasoning_content"] != ""
        ):
            metadata["reasoning_content"] = model_response_data["reasoning_content"]

        return model_response_data["model_responses"], metadata

    def decode_ast(self, result, language: ReturnFormat, has_tool_call_tag: bool):
        """
        This method takes raw model output (from `_parse_query_response_xxx`) and convert it to standard AST checker input.
        """
        raise NotImplementedError

    def decode_execute(self, result, has_tool_call_tag: bool):
        """
        This method takes raw model output (from `_parse_query_response_xxx`) and convert it to standard execute checker input.
        """
        raise NotImplementedError

    @final
    def write(self, result, result_dir, update_mode=False):
        # Use the internal registry name to decide the result directory to avoid
        # collisions between different variants that share the same API model name.
        model_result_dir = result_dir / self.registry_dir_name

        if isinstance(result, dict):
            result = [result]

        # Collect and format each entry for JSON compatibility
        entries_to_write = [make_json_serializable(entry) for entry in result]

        # Group entries by their `test_category` for efficient file handling
        file_entries = {}
        for entry in entries_to_write:
            test_category = extract_test_category_from_id(entry["id"])
            # Determine the high-level grouping folder (non_live, live, etc.)
            group_dir_name = get_directory_structure_by_id(entry["id"])
            group_dir_path = model_result_dir / group_dir_name
            group_dir_path.mkdir(parents=True, exist_ok=True)

            file_path = group_dir_path / f"{VERSION_PREFIX}_{test_category}_result.json"
            file_entries.setdefault(file_path, []).append(entry)

        for file_path, entries in file_entries.items():
            if update_mode:
                # Load existing entries from the file
                existing_entries = {}
                if file_path.exists():
                    existing_entries = {
                        entry["id"]: entry for entry in load_file(file_path)
                    }

                # Update existing entries with new data
                for entry in entries:
                    existing_entries[entry["id"]] = entry

                # Sort entries by `id` and write them back to ensure order consistency
                sorted_entries = sorted(existing_entries.values(), key=sort_key)
                with open(file_path, "w") as f:
                    for entry in sorted_entries:
                        content = json.dumps(entry) + "\n"
                        f.write(content)
                        f.flush()

            else:
                # Normal mode: Append to the end of the file
                # Note: We will sort all the entries at the end of the generation pipeline to ensure the order is consistent
                entries.sort(key=sort_key)
                with open(file_path, "a") as f:
                    for entry in entries:
                        content = json.dumps(entry) + "\n"
                        f.write(content)
                        f.flush()

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        """
        Call the model API in FC mode to get the response.
        Return the response object that can be used to feed into the `_parse_query_response_FC` method.
        """
        raise NotImplementedError

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This might includes transforming the input user message into the format expected by the model, extract out the system prompt (if any), and any other necessary preprocessing steps. Those steps can also be done in the `add_first_turn_message_FC` and `_add_next_turn_user_message_FC` methods, but it's usually cleaner to do it here.
        The inference_data dict is updated in place and returned.

        Note: This method has different signature from its Prompting version.
        """
        raise NotImplementedError

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """
        [Only for FC mode]
        This method is used to prepare/compile the tools from the test entry and add them to the inference data to use for model query in FC mode.
        Function docs usually need to be transformed to the format expected by the model, done through the `convert_to_tool` function from `model_handler/utils.py`.
        The inference_data dict is updated in place and returned.
        """
        raise NotImplementedError

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - tool_call_ids (list[str]): The IDs of the tool calls that are generated by the model. Optional.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history, in the format that the model expects.

        Args:
            inference_data (dict): The inference data from previous processing steps.
            first_turn_message (list[dict]): The first turn message from the test entry. It has variable length. It might contain one or more of the following roles:
                - "system": The system message. This role will only appear at most once, at the beginning of the first turn. For most entry, this role will not appear.
                - "user": The user message.
                - "assistant": The assistant message. For most entry, this role will not appear.

        Returns:
            inference_data (dict): The updated inference data that will be send to `_query_FC` to call the model API.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is guaranteed to be a `user` role message.
        """
        raise NotImplementedError

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_FC(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        Some models may need to add additional information to the chat history, such as tool call IDs.
        """
        raise NotImplementedError

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        """
        Call the model API in prompting mode to get the response.
        Return the response object that can be used to feed into the decode method.
        """
        raise NotImplementedError

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Preprocess the testset entry before sending it to the model.
        This might includes transforming the input user message into the format expected by the model, extract out the system prompt (if any), and any other necessary preprocessing steps. Those steps can also be done in the `add_first_turn_message_prompting` and `_add_next_turn_user_message_prompting` methods, but it's usually cleaner to do it here.
        The function docs are usually supplied to the prompting models as part of the system prompt, done via the `system_prompt_pre_processing_chat_model` function from `model_handler/utils.py`, unless the model has a different way of handling it.
        Returns a dict that contains all the necessary information for the query method.
        Things like `system_prompt` and `chat_history` are optional, specific to the model.

        Note: This method has different signature from its FC version.
        """
        raise NotImplementedError

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        Parses the raw response from the model API to extract the result, input token count, and output token count.

        Args:
            api_response (any): The raw response from the model API.

        Returns:
            A dict containing the following elements:
                - model_responses (any): The parsed result that can be directly used as input to the decode method.
                - input_token (int): The number of tokens used in the input to the model.
                - output_token (int): The number of tokens generated by the model as output.
                - Any other metadata that is specific to the model.
        """
        raise NotImplementedError

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """
        Add the first turn message to the chat history, in the format that the model expects.

        Args:
            inference_data (dict): The inference data from previous processing steps.
            first_turn_message (list[dict]): The first turn message from the test entry. It has variable length. It might contain one or more of the following roles:
                - "system": The system message. This role will only appear at most once, at the beginning of the first turn.
                - "user": The user message.
                - "assistant": The assistant message. For most entry, this role will not appear.

        Returns:
            inference_data (dict): The updated inference data that will be send to `_query_prompting` to call the model API.
        """
        raise NotImplementedError

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """
        [Only for multi-turn]
        Add next turn user message to the chat history for query.
        user_message is a list of 1 element, which is guaranteed to be a `user` role message.
        """
        raise NotImplementedError

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """
        Add assistant message to the chat history.
        """
        raise NotImplementedError

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """
        Add the execution results to the chat history to prepare for the next turn of query.
        By default, execution results are added back as a `user` role message, as most models don't support the `tool` role in prompting mode.
        """
        raise NotImplementedError
