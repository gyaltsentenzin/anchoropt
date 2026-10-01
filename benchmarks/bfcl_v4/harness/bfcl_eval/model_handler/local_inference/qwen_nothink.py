from overrides import override
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

_GENERATION_PREFIX = "<|im_start|>assistant\n"


class QwenNoThinkHandler(QwenFCHandler):
    """QwenFCHandler with thinking disabled via an empty <think> block.

    Inherits QwenFCHandler so tool definitions are injected as <tools> XML
    (the format Qwen3 was trained on) and responses are parsed as <tool_call>
    XML rather than Python AST syntax.
    """

    @override
    def _format_prompt(self, messages, function):
        prompt = super()._format_prompt(messages, function)
        # Parent always ends with '<|im_start|>assistant\n'.
        # Replace with the no-think variant that forces the model to skip reasoning.
        if prompt.endswith(_GENERATION_PREFIX):
            prompt = prompt[: -len(_GENERATION_PREFIX)]
            prompt += _GENERATION_PREFIX + "<think>\n\n</think>\n\n"
        return prompt
