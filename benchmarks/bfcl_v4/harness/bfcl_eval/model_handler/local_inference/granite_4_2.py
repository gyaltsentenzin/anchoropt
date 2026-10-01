import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override

# granite-4.2-8b-prerelease-r260622a ships a ChatML-style chat template
# (<|im_start|>/<|im_end|>, eos=<|im_end|>) with an XML-tag tool-call format:
#   <tool_call>
#   <function=NAME>
#   <parameter=ARG>
#   value
#   </parameter>
#   </function>
#   </tool_call>
# instead of Granite 4.1's <|start_of_role|> markers and JSON-object tool calls
# (see granite_4.py). This is a from-scratch port of the checkpoint's own
# chat_template.jinja, not a variant of Granite4PromptHandler.

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>")
_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_TOOLS_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE "
    "the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning from model output.

    Ported from bfcl_v4_eval/eval/granite_xml_tool_parser.py's _strip_thinking(),
    minus the vLLM-plugin plumbing. Handles three cases: a complete <think>...</think>
    block; a bare trailing "...</think>" with no opening tag (the chat template puts
    <think> in the generation-prompt prefix, so the raw completion begins mid-thought);
    and an unclosed trailing <think> (partial generation).
    """
    text = _THINK_BLOCK_RE.sub("", text)
    if THINK_CLOSE in text and THINK_OPEN not in text.split(THINK_CLOSE, 1)[0]:
        text = text.split(THINK_CLOSE, 1)[1]
    if THINK_OPEN in text and THINK_CLOSE not in text[text.rfind(THINK_OPEN):]:
        text = text[: text.rfind(THINK_OPEN)]
    return text


def _split_reasoning(content: str) -> tuple:
    """Split "[<think>]reasoning</think>rest" into (reasoning, rest).

    No-op (returns ("", content)) if there's no </think> at all, which also covers
    the bare-prefix case where the generation prompt already opened <think> for us.
    """
    if THINK_CLOSE not in content:
        return "", content
    reasoning, _, rest = content.partition(THINK_CLOSE)
    if THINK_OPEN in reasoning:
        reasoning = reasoning.split(THINK_OPEN, 1)[-1]
    return reasoning.strip("\n"), rest.lstrip("\n")


def _coerce_param_value(pval: str):
    """Best-effort type coercion for a raw <parameter> string value.

    No tool schema is available at this call site (BFCL's decode_ast/decode_execute
    signature doesn't thread it through), so this only implements the "unknown type"
    fallback branch of granite_xml_tool_parser.py's _coerce_param_value: try
    json.loads, else a few literal mappings, else keep the raw string. This can
    mis-coerce a numeric-looking string parameter (e.g. a zip code) into an int —
    a known, pre-existing limitation of this output format without schema access,
    not something fixable here without changing the abstract handler interface.
    """
    try:
        return json.loads(pval)
    except (json.JSONDecodeError, ValueError):
        if pval == "True":
            return True
        if pval == "False":
            return False
        if pval in ("None", "null", "none"):
            return None
        return pval


def _extract_tool_calls(text: str) -> list:
    """Parse <tool_call><function=NAME><parameter=ARG>value</parameter>...</function></tool_call>
    blocks (outside any <think> block) into [{"name": ..., "arguments": {...}}, ...].
    """
    text = _strip_thinking(text)
    results = []
    for block in _TOOL_CALL_BLOCK_RE.findall(text):
        func_match = _FUNCTION_RE.search(block)
        if not func_match:
            continue
        name = func_match.group(1).strip()
        arguments = {}
        for pname, pval in _PARAMETER_RE.findall(block):
            arguments[pname.strip()] = _coerce_param_value(pval.strip())
        results.append({"name": name, "arguments": arguments})
    return results


def _serialize_tool_call(tool_call: dict) -> str:
    """Render a {"name": ..., "arguments": {...}} (or OpenAI-style {"function": {...}})
    dict back into the model's own <tool_call><function=...>... text, for replaying
    a prior assistant turn's tool calls back into the prompt on later turns.
    """
    if "function" in tool_call:
        tool_call = tool_call["function"]
    arguments = tool_call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            arguments = {}
    body = f"<tool_call>\n<function={tool_call['name']}>\n"
    for arg_name, arg_value in arguments.items():
        value_str = (
            json.dumps(arg_value) if isinstance(arg_value, (dict, list)) else str(arg_value)
        )
        body += f"<parameter={arg_name}>\n{value_str}\n</parameter>\n"
    body += "</function>\n</tool_call>\n"
    return body


class Granite42PromptHandler(OSSHandler):
    """
    Handler for granite-4.2-8b-prerelease-r260622a in prompting mode.

    enable_thinking only controls the generation-prompt suffix for the *next*
    completion (open "<think>\\n" vs. closed "<think></think>") — matching the
    checkpoint's own chat_template.jinja, where historical assistant turns always
    get wrapped in <think>...</think> (empty if there was no real reasoning) when
    replayed, regardless of this flag. truncate_history_thinking (also true to the
    template's own default) drops any real reasoning content from turns other than
    the current one, so multi-turn conversations don't accumulate unbounded
    chain-of-thought tokens.
    """

    enable_thinking = False
    truncate_history_thinking = True

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.is_fc_model = False

    def _render_assistant_content(self, message: dict, idx: int, last_user_idx: int) -> str:
        reasoning_content = message.get("reasoning_content", "") or ""
        content = message.get("content") or ""
        if not reasoning_content and THINK_CLOSE in content:
            reasoning_content, content = _split_reasoning(content)

        content = content.strip(chr(10))
        # The separator after </think> is "\n" iff real reasoning_content was present
        # at construction time (matches the checkpoint's own template: it bakes this
        # separator in before any historical-truncation logic runs, so truncating a
        # historical turn's reasoning text still leaves the separator behind).
        sep = "\n" if reasoning_content else ""

        is_historical = self.truncate_history_thinking and idx < last_user_idx
        if reasoning_content and not is_historical:
            return f"<think>\n{reasoning_content}\n</think>\n{content}"
        return f"<think></think>{sep}{content}"

    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = ""

        has_system = bool(messages) and messages[0]["role"] == "system"
        system_content = messages[0]["content"] if has_system else ""
        loop_messages = messages[1:] if has_system else messages

        if system_content or function:
            formatted_prompt += "<|im_start|>system\n" + system_content
            if function:
                if system_content:
                    formatted_prompt += "\n\n"
                formatted_prompt += (
                    "# Tools\n\nYou have access to the following functions:\n\n<tools>"
                )
                for tool in function:
                    tool = tool.get("function", tool)
                    formatted_prompt += "\n" + json.dumps(tool)
                formatted_prompt += "\n</tools>" + _TOOLS_INSTRUCTIONS
            formatted_prompt += "<|im_end|>\n"

        last_user_idx = -1
        for i, message in enumerate(loop_messages):
            if message["role"] == "user":
                last_user_idx = i

        idx = 0
        n = len(loop_messages)
        while idx < n:
            message = loop_messages[idx]
            role = message["role"]

            if role in ("user", "system"):
                formatted_prompt += f"<|im_start|>{role}\n{message['content']}<|im_end|>\n"
                idx += 1
                continue

            if role == "tool":
                formatted_prompt += "<|im_start|>user\n"
                while idx < n and loop_messages[idx]["role"] == "tool":
                    formatted_prompt += (
                        f"<tool_response>\n{loop_messages[idx]['content']}\n</tool_response>\n"
                    )
                    idx += 1
                formatted_prompt += "<|im_end|>\n"
                continue

            if role == "assistant":
                content = self._render_assistant_content(message, idx, last_user_idx)
                content = content.rstrip("\n")
                formatted_prompt += f"<|im_start|>assistant\n{content}"

                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    if content.strip():
                        formatted_prompt += "\n"
                    for tool_call in tool_calls:
                        formatted_prompt += _serialize_tool_call(tool_call)

                formatted_prompt += "<|im_end|>\n"
                idx += 1
                continue

            idx += 1  # defensive: unknown role, skip

        if self.enable_thinking:
            formatted_prompt += "<|im_start|>assistant\n<think>\n"
        else:
            formatted_prompt += "<|im_start|>assistant\n<think></think>"

        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        reasoning_content, cleaned_response = _split_reasoning(model_response)
        extracted_tool_calls = _extract_tool_calls(cleaned_response)

        if extracted_tool_calls:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
                "reasoning_content": reasoning_content,
            }
        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
                "reasoning_content": reasoning_content,
            }

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = _extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = _extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = [{item["name"]: item["arguments"]} for item in tool_calls]
        return convert_to_function_call(decoded_result)


class Granite42ThinkingPromptHandler(Granite42PromptHandler):
    """
    Same handler, thinking enabled: forces a fresh <think>...\\n</think> block on
    every new completion (enable_thinking=True), while still truncating reasoning
    out of replayed history turns via the inherited truncate_history_thinking=True.
    """

    enable_thinking = True
