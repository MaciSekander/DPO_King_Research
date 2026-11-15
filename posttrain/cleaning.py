"""Text helpers used in data cleaning (notebook 01).

loose_norm is for DATA FILTERING ONLY (checking a dataset agrees with itself).
Never use it to grade model outputs -- use posttrain.grader.mv_grade.
"""
import re

# Trailing "The answer is: ..." line at the end of a MetaMathQA response
ANS_RE = re.compile(r"\n?The answer is:\s*(.*)\s*$")


def norm(s):
    """Whitespace normalization for exact text matching (leak filter, dedup)."""
    return " ".join(s.split())


def loose_norm(s):
    """Strip formatting ($, degrees, %, \\text{}, spacing, dfrac) for agreement checks."""
    if s is None:
        return None
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = s.replace("\\!", "").replace("\\,", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    return "".join(s.split())


def split_response(resp):
    """Split a MetaMathQA response into (body without the label line, label)."""
    m = ANS_RE.search(resp)
    tail = m.group(1).strip() if m else None
    body = (resp[:m.start()] if m else resp).rstrip()
    return body, tail


def ngrams(text, n=13):
    """Word n-grams for decontamination checks."""
    words = re.sub(r"\s+", " ", text.lower()).split(" ")
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}