"""Answer extraction: pull the final \\boxed{...} answer out of a solution or model output."""


def get_last_boxed(text):
    """Return the contents of the last \\boxed{...} or \\fbox{...} in text, or None.

    - Handles nested braces: \\boxed{\\frac{1}{2}} -> "\\frac{1}{2}"
    - Handles brace-less form: "\\boxed 5." -> "5"
    - Empty box or unbalanced braces (truncated output) -> None
    """
    if text is None:
        return None

    # Find whichever of \boxed / \fbox appears last
    idx_boxed = text.rfind(r"\boxed")
    idx_fbox = text.rfind(r"\fbox")
    if idx_boxed == -1 and idx_fbox == -1:
        return None
    if idx_boxed >= idx_fbox:
        current_idx = idx_boxed + len(r"\boxed")
    else:
        current_idx = idx_fbox + len(r"\fbox")

    # Skip whitespace after the command
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1
    if current_idx >= len(text):
        return None

    # No brace, e.g. "\boxed 5." -> take the next token
    if text[current_idx] != "{":
        token = text[current_idx:].split()[0].rstrip("$.")
        return token or None

    # Walk forward tracking brace depth
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx
    while current_idx < len(text) and brace_depth > 0:
        if text[current_idx] == "{":
            brace_depth += 1
        elif text[current_idx] == "}":
            brace_depth -= 1
        current_idx += 1

    if brace_depth != 0:          # unbalanced = truncated
        return None

    answer = text[content_start_idx:current_idx - 1].strip()
    return answer or None


def run_tests(verbose=True):
    """Unit tests for get_last_boxed. Raises AssertionError on any failure."""
    tests = {
        r"so $a+b=\boxed{0}$.": "0",
        r"\boxed{\frac{1}{2}}": r"\frac{1}{2}",
        r"\boxed{\{1,2\}}": r"\{1,2\}",
        r"first \boxed{1} then \boxed{2}": "2",
        r"answer is \fbox{7}": "7",
        r"\boxed 5.": "5",
        r"\boxed{}": None,
        r"\boxed{\frac{1}{2": None,
        "no answer here": None,
    }
    fails = 0
    for text, expected in tests.items():
        got = get_last_boxed(text)
        ok = got == expected
        fails += not ok
        if verbose:
            print(f"{'OK  ' if ok else 'FAIL'} {text!r} -> {got!r}")
    assert fails == 0, f"{fails} extraction tests failed"
    return True