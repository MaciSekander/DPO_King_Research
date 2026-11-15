"""The ONE prompt format used everywhere: SFT, DPO, GRPO, and evaluation.

We write the ChatML string ourselves instead of calling tokenizer.apply_chat_template,
because Qwen3's built-in template has extra logic (<think> blocks, tool calls) that we
don't want silently changing the format between stages.

Qwen3 tokens used:
  <|im_start|>, <|im_end|>  - turn markers (ChatML)
  <|im_end|>                - end of the model's answer; generation must STOP on this
"""

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_prompt(problem, system=SYSTEM_PROMPT):
    """Everything the model sees BEFORE it starts writing (no loss on this part)."""
    return (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{problem.strip()}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def build_completion(solution):
    """What the model should write (loss is computed on this part), ending with <|im_end|>."""
    return f"{solution.strip()}{IM_END}"


def to_prompt_completion(example):
    """datasets.map() helper: {problem, solution, ...} -> adds {prompt, completion}."""
    return {
        "prompt": build_prompt(example["problem"]),
        "completion": build_completion(example["solution"]),
    }


STOP_STRINGS = [IM_END, "<|endoftext|>"]   # pass to vLLM / generate() as stop sequences