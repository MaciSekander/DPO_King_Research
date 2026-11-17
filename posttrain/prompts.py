"""The ONE prompt format used everywhere: SFT, DPO, GRPO, and evaluation.

We write the ChatML string ourselves instead of calling tokenizer.apply_chat_template,
because Qwen3's built-in template has extra logic (<think> blocks, tool calls) that we
don't want silently changing the format between stages.

Qwen3 tokens used:
  <|im_start|>, <|im_end|>  - turn markers (ChatML)
  <|im_end|>                - end of the model's answer; generation must STOP on this
"""
import random

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOS = "<|endoftext|>"


# ---------------------------------------------------------------- chat format (all arms)
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


STOP_STRINGS = [IM_END, EOS]   # kept for reference; prefer stop_token_ids() with vLLM


def stop_token_ids(tokenizer):
    """Token IDs to stop generation on. Use these (not strings) with vLLM, because
    special tokens are stripped from the output text and a string match can miss them."""
    return [tokenizer.convert_tokens_to_ids(t) for t in (IM_END, EOS)]


# ---------------------------------------------------------------- few-shot (base model only)
FEWSHOT_STOP_STRINGS = ["\n\nProblem:"]   # model starts inventing the next problem -> stop


def select_fewshot_examples(pool, k=4, seed=0, max_chars=600):
    """Pick k short MATH-train examples from distinct subjects, deterministically.

    pool: the SFT dataset (needs columns source, subject, problem, solution).
    Only source == "MATH_train" rows are used (human-written solutions, never MATH-500).
    """
    idxs = list(range(len(pool)))
    random.Random(seed).shuffle(idxs)
    chosen, used_subjects = [], set()
    for i in idxs:
        ex = pool[i]
        if ex["source"] != "MATH_train":
            continue
        if ex["subject"] in used_subjects:
            continue
        if len(ex["solution"]) > max_chars or "\\boxed" not in ex["solution"]:
            continue
        chosen.append({"idx": i, "subject": ex["subject"], "level": ex["level"],
                       "problem": ex["problem"], "solution": ex["solution"]})
        used_subjects.add(ex["subject"])
        if len(chosen) == k:
            break
    return chosen


def build_fewshot_prompt(problem, examples):
    """Plain-text few-shot prompt (no chat tokens) - the fair way to test a BASE model."""
    parts = [f"Problem:\n{ex['problem'].strip()}\n\nSolution:\n{ex['solution'].strip()}"
             for ex in examples]
    parts.append(f"Problem:\n{problem.strip()}\n\nSolution:\n")
    return "\n\n".join(parts)