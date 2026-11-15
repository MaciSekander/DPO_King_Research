"""The single grader for eval, DPO labeling and GRPO reward.

Rule: the response must contain a \\boxed{} answer (no box = no credit). The boxed answer
is compared to the gold answer with Hugging Face Math-Verify.
Requires: pip install "math-verify[antlr4_13_2]"
"""
from math_verify import parse, verify

from .extract import get_last_boxed


def mv_grade(gold_answer, response):
    """True if the last \\boxed{} in response matches gold_answer."""
    boxed = get_last_boxed(response)
    if boxed is None:
        return False                      # format rule
    try:
        # Math-Verify only extracts LaTeX inside a math environment -> wrap in $...$
        gold = parse("$" + gold_answer + "$", parsing_timeout=5)
        pred = parse("$" + boxed + "$", parsing_timeout=5)
        if not gold or not pred:
            return False
        return bool(verify(gold, pred, timeout_seconds=5))   # gold FIRST (asymmetric)
    except Exception:
        return False                      # never crash a training run


def reward(response, gold_answer):
    """GRPO / DPO-labeling reward: 1.0 if correct, else 0.0."""
    return 1.0 if mv_grade(gold_answer, response) else 0.0


TEST_CASES = [  # (gold, model response, expected)
    (r"\frac{1}{2}", r"so $\boxed{0.5}$", True),
    (r"\frac{3}{4}", r"\boxed{\frac34}", True),
    (r"\frac{\sqrt{2}}{2}", r"\boxed{\sqrt{\frac{1}{2}}}", True),
    (r"2\pi", r"\boxed{2 \pi}", True),
    (r"\pi", r"\boxed{3}", False),
    (r"5\sqrt{2}", r"\boxed{\sqrt{50}}", True),
    (r"120^\circ", r"\boxed{120}", True),
    (r"5", r"\boxed{x=5}", True),
    (r"10\%", r"\boxed{10}", True),
    (r"5", r"\boxed{5 \text{ cm}}", True),
    (r"(x+1)^2", r"\boxed{x^2+2x+1}", True),
    (r"(0,1)", r"\boxed{[0,1]}", False),
    (r"\{1,2,3\}", r"\boxed{\{3,2,1\}}", True),
    (r"-2, 3", r"\boxed{3, -2}", True),
    (r"\begin{pmatrix} 1 \\ 2 \end{pmatrix}", r"\boxed{\begin{pmatrix} 2 \\ 1 \end{pmatrix}}", False),
    (r"\frac{7}{2}", r"\boxed{3}", False),
    (r"3", r"the answer is 3", False),
    (r"3", r"\boxed{\frac{3", False),
]


def run_tests(verbose=True):
    """Run the grader test suite. Raises AssertionError on any failure."""
    fails = 0
    for gold, resp, expected in TEST_CASES:
        got = mv_grade(gold, resp)
        ok = got == expected
        fails += not ok
        if verbose:
            print(f"{'OK  ' if ok else 'FAIL'} {gold!r:40} | {resp!r:42} -> {got}")
    assert fails == 0, f"{fails} grader tests failed"
    return True