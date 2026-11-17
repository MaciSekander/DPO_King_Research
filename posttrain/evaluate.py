"""Generation + grading + summaries + saving, shared by every eval run (all arms).

Typical use in a notebook:
    outputs, secs = run_generation(llm, prompts, params)
    records = grade(math500, prompts, outputs)
    summary = summarize(records, seconds=secs)
    save_run(records, summary, "arm0_zeroshot", hub_repo="Maci10/dpo-king-eval-results")
"""
import json
import os
import time
from collections import defaultdict

from .extract import get_last_boxed
from .grader import mv_grade


def run_generation(llm, prompts, sampling_params):
    """Generate one completion per prompt with vLLM. Returns (outputs, wall-clock seconds)."""
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    return outputs, time.time() - t0


def grade(dataset, prompts, outputs, answer_key="answer"):
    """One record per problem: response, extracted answer, correctness, length, stop reason."""
    records = []
    for i, (ex, prompt, out) in enumerate(zip(dataset, prompts, outputs)):
        comp = out.outputs[0]
        text = comp.text
        records.append({
            "idx": i,
            "unique_id": ex.get("unique_id", str(i)),
            "subject": ex.get("subject"),
            "level": ex.get("level"),
            "problem": ex["problem"],
            "gold": ex[answer_key],
            "prompt": prompt,
            "response": text,
            "extracted": get_last_boxed(text),
            "correct": bool(mv_grade(ex[answer_key], text)),
            "n_tokens": len(comp.token_ids),
            "finish_reason": comp.finish_reason,   # "stop" = ended itself, "length" = hit max_tokens
        })
    return records


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def _by(records, key):
    groups = defaultdict(list)
    for r in records:
        groups[str(r[key])].append(r["correct"])
    return {k: {"acc": _mean(v), "n": len(v)} for k, v in sorted(groups.items())}


def summarize(records, seconds=None):
    """Headline numbers for one run."""
    total_tokens = sum(r["n_tokens"] for r in records)
    summary = {
        "n": len(records),
        "accuracy": _mean(r["correct"] for r in records),
        "has_box_rate": _mean(r["extracted"] is not None for r in records),
        "hit_max_tokens_rate": _mean(r["finish_reason"] == "length" for r in records),
        "mean_tokens": _mean(r["n_tokens"] for r in records),
        "total_output_tokens": total_tokens,
        "by_level": _by(records, "level"),
        "by_subject": _by(records, "subject"),
    }
    if seconds:
        summary["seconds"] = round(seconds, 1)
        summary["output_tokens_per_sec"] = round(total_tokens / seconds, 1)
    return summary


def print_summary(summary, name=""):
    print(f"=== {name} ===")
    print(f"accuracy {summary['accuracy']:.1%}  |  has \\boxed {summary['has_box_rate']:.1%}  |  "
          f"hit max tokens {summary['hit_max_tokens_rate']:.1%}  |  mean tokens {summary['mean_tokens']:.0f}")
    if "seconds" in summary:
        print(f"time {summary['seconds']:.0f}s  |  {summary['output_tokens_per_sec']:.0f} output tok/s")
    print("by level:  " + "  ".join(f"L{k}: {v['acc']:.0%} (n={v['n']})" for k, v in summary["by_level"].items()))


def save_run(records, summary, run_name, out_dir="outputs", hub_repo=None, meta=None):
    """Save records (.jsonl) + summary (.json) locally; optionally push both to a private Hub repo.

    On the Hub, every run becomes its own config of ONE dataset repo:
        load_dataset(hub_repo, run_name)   # -> that run's records
    """
    os.makedirs(out_dir, exist_ok=True)
    full_summary = {"run_name": run_name, **(meta or {}), **summary}

    rec_path = os.path.join(out_dir, f"{run_name}.jsonl")
    sum_path = os.path.join(out_dir, f"{run_name}_summary.json")
    with open(rec_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(sum_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print("saved", rec_path, "and", sum_path)

    if hub_repo:
        from datasets import Dataset
        from huggingface_hub import HfApi

        Dataset.from_list(records).push_to_hub(hub_repo, config_name=run_name, private=True)
        HfApi().upload_file(path_or_fileobj=sum_path, path_in_repo=f"summaries/{run_name}.json",
                            repo_id=hub_repo, repo_type="dataset")
        print("pushed to", hub_repo, "config", run_name)
    return full_summary