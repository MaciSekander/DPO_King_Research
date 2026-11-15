"""Load the project datasets: local copy if present, otherwise the HF Hub.

Usage:
    from posttrain.data import load_split
    sft_all = load_split("sft_all")     # also: "math_val", "math500"
"""
import os

from datasets import load_dataset, load_from_disk

HF_USER = "Maci10"     # account the datasets were pushed to (check huggingface.co/Maci10)
PREFIX = "dpo-king"    # repos are named f"{HF_USER}/{PREFIX}-{name}"
SPLITS = ("sft_all", "math_val", "math500")


def load_split(name, local_dir="data"):
    """Load a dataset from data/<name> if it exists, else from the (private) Hub repo."""
    if name not in SPLITS:
        raise ValueError(f"Unknown split {name!r}; expected one of {SPLITS}")
    path = os.path.join(local_dir, name)
    if os.path.exists(path):
        return load_from_disk(path)
    return load_dataset(f"{HF_USER}/{PREFIX}-{name}", split="train")