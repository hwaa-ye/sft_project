"""Shared formatting helpers for Qwen3 math SFT data."""


def format_math_sft(example):
    instruction = example["instruction"]
    reasoning = example.get("reasoning", "")
    answer = example.get("answer", "")
    response = f"<think>\n{reasoning}\n</think>\n\n<answer>\n{answer}\n</answer>"
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def format_generic_sft(example):
    instruction = example.get("instruction") or example.get("input", "")
    target = example.get("target", "")
    response = target if "<think>" in target else f"<think>\n{target}\n</think>"
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def format_sft(example):
    if "reasoning" in example and "answer" in example:
        return format_math_sft(example)
    return format_generic_sft(example)


def instruction_prefix(example):
    instruction = example.get("instruction", example.get("input", ""))
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
