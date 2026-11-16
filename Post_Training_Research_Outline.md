# Post-Training Research Plan (working draft)

## Working title
"Is DPO Still King? A Compute-Accounted Look Inside Offline vs On-Policy Post-Training" (draft)

## Main research question (draft)
Under a single-GPU budget, can DPO on self-generated, verifier-labeled pairs match GRPO on MATH-500 per GPU-hour — and where it falls short or wins, what do its training dynamics and internals reveal about why?

"King" is judged on four explicit criteria: accuracy, compute efficiency, stability, transparency.

### Sub-questions
- **RQ1 Performance:** How do SFT, SFT→DPO, SFT→GRPO, SFT→DPO→GRPO compare on MATH-500 (pass@1, pass@k, per level/subject)?
- **RQ2 Efficiency:** At matched GPU-hours (iterative DPO rounds vs GRPO), does the gap close?
- **RQ3 Mechanism:** What does DPO change (likelihood displacement, implicit reward, token-level credit, weight updates) and how does that differ from GRPO?
- **RQ4 Complementarity:** Is DPO a useful warm start for GRPO (arm D vs C)?

### Hypotheses (commit before running; DPO must be able to lose)
- H1: Single-round DPO trails GRPO on pass@1 but is far cheaper.
- H2: Iterative (on-policy-ish) DPO closes most of the gap at matched compute.
- H3: DPO shows likelihood displacement (chosen log-prob falls) that GRPO does not; size of displacement predicts final accuracy.
- H4: DPO's token-level implicit reward concentrates on the first erroneous step more often than chance (checked against PRM800K labels).

## Closest prior work and positioning (added 2026-11-16)
- **Lanchantin et al., "Bridging Offline and Online Reinforcement Learning for LLMs" (FAIR at Meta / NYU, arXiv 2506.21495).** Llama-3.1-8B-Instruct; offline vs semi-online (sync every s = 5/10/100 steps) vs online DPO vs GRPO; Math-Verify rewards on NuminaMath; eval incl. MATH-500. Findings: offline DPO lags (MATH-500 53.7 vs ~57–59); semi-online DPO ≈ online DPO; online DPO matches or slightly beats GRPO. Run on 32+8 H200s. **Does NOT** report GPU-hours / compute-matched comparisons, does NOT study model size (single 8B instruct seed), hyperparameters differ across settings, no mechanistic analysis.
  - → Effectively supports our H1 (offline lags) and H2 (iterative/semi-online closes the gap) at 8B with a big cluster. Our iterative-DPO arm = their "semi-online DPO"; adopt their terminology and sync-interval framing.
- **Verma & Ravindran, "How Much Online RL is Enough? Informative Rollouts for Offline Preference Optimization in RLVR" (arXiv 2605.21266, May 2026).** G2D: short GRPO warm-up → build static preference set with difficulty calibration → offline DPO. On Qwen2.5-7B, G2D reports 62.4% MATH-500 vs GRPO 51.6% at ~4x lower compute (their configuration). 7–8B models.
  - → Directly relevant to RQ4 (ordering of DPO/GRPO) — note they do GRPO→DPO, we do DPO→GRPO. Consider adding a G2D-style arm (GRPO→DPO) if budget allows.
- **"It Takes Two: Your GRPO Is Secretly DPO" (arXiv 2510.00977).** Shows GRPO and DPO gradients share a positive/negative contrastive structure. → Theory anchor for RQ3 (mechanism).
- **Our gap / contribution:** (1) explicit GPU-hour accounting on a single rented GPU; (2) small (0.6B) *base* model with our own SFT, vs 7–8B instruct seeds; (3) controlled, identical hyperparameter protocol across arms; (4) mechanistic / transparency analysis of DPO vs GRPO (RQ3), which none of the above provide; (5) sequential DPO→GRPO warm start (RQ4).
- Paper framing shifts from "nobody has compared these" to "replicate-and-extend at small scale with compute accounting, and explain *why* via training dynamics."

## Motivation / paper framing
- Academic post-training work leans heavily on offline methods (DPO and variants) because on-policy RL is seen as too expensive; careful on-policy RL studies mostly come from well-funded labs (e.g. Lanchantin et al. used 40 H200s).
- Pitch: with verifiable rewards (no reward model) and a small model, on-policy RL is affordable on one rented GPU. We run a **compute-accounted** comparison of offline, semi-online and on-policy methods, on the same prompts and the same reward signal — and open up DPO's internals, using GRPO as a contrast.
- Key controls: self-generated DPO pairs graded by the same verifier GRPO uses (isolates algorithm from data); compute-matched DPO (more iterative rounds) vs GRPO; GPU-hours reported for every arm; same hyperparameter selection protocol for every arm.

## DPO transparency thread (candidate analyses — pick 2–3 core)
1. **Training dynamics beyond loss** (cheap, logged during training): chosen/rejected log-probs, margin, implicit-reward accuracy, log-prob of held-out correct solutions and of the model's own greedy outputs, entropy, length. Watch for likelihood displacement. (Lanchantin et al. report entropy collapse and length collapse without reference syncing — worth tracking.)
2. **Implicit reward as a verifier:** score held-out correct/incorrect solutions with β·log(π/π_ref); measure how well DPO's implicit reward ranks correctness (AUROC), compare to the true verifier.
3. **Token-level credit assignment:** per-token log-ratio heatmaps over solutions. Does credit concentrate on the step where a wrong solution first goes wrong? Validate against human step-level labels (PRM800K, analysis only, never training).
4. **Where in the network DPO acts:** per-layer weight-update norms, effective rank / SVD of updates, sparsity; compare DPO vs GRPO vs SFT. Optional: linear probes for "this solution is correct" in hidden states before/after.
5. **Which pairs matter (data attribution):** controlled pair types (near-miss vs very different chosen/rejected), ablations, optional gradient-similarity attribution.
- Save frequent DPO checkpoints (dense early) to support dynamics analysis.
- Tools: plain PyTorch hooks / nnsight on HF Qwen3; 0.6B makes all of this cheap.

## Related work to cover
Lanchantin et al. 2025 (Bridging Offline and Online RL for LLMs); Verma & Ravindran 2026 (G2D); "It Takes Two: Your GRPO Is Secretly DPO" (2510.00977); Ivison et al. 2024 (Unpacking DPO and PPO); Xu et al. 2024 (Is DPO Superior to PPO?); Tulu 3 RLVR; DeepSeekMath/GRPO; Dr. GRPO / "Understanding R1-Zero-like Training"; SimpleRL-Zoo; Yue et al. 2025 (does RL incentivize reasoning beyond the base model?); iterative/online DPO papers (Pang et al. 2024, Xu et al. 2023). Transparency: Rafailov et al. 2024 ("From r to Q*", DPO as token-level Q-function); Razin et al. 2024 (likelihood displacement); Pal et al. 2024 (DPOP / chosen-likelihood decrease); Lee et al. 2024 (mechanistic study of DPO on toxicity); Mukherjee et al. 2025 (RL updates small subnetworks); Lightman et al. 2023 (PRM800K).

## Decisions so far
- **Model:** Qwen3-0.6B-Base (base, not instruct)
- **Task:** competition-style math (MATH distribution); all outputs end with `\boxed{answer}`
- **SFT data (clean by construction, everything traces back to MATH train):**
  - MATH train (Hendrycks, 7.5k of the 12.5k; the other 5k are test) minus ~500 held out for validation
  - MetaMathQA MATH-type subset, augmentations of MATH train only: **AnsAug + Rephrased only** (MATH_SV and MATH_FOBAR dropped)
  - Avoid large web-mixed sets (NuminaMath, OpenMathInstruct-2) to keep provenance simple
  - n-gram decontam check against MATH-500 run and reported
- **Prompt format (decided):** hand-written ChatML in `posttrain/prompts.py` (not `apply_chat_template`, to avoid Qwen3's hidden <think>/tool logic). System prompt: "Please reason step by step, and put your final answer within \boxed{}." Stored as separate `prompt` / `completion` fields (loss on completion only). Completion ends with `<|im_end|>` (base tokenizer's EOS = PAD = `<|endoftext|>`, so ending on EOS risks it being masked and the model never learning to stop). Generation stops on `<|im_end|>` / `<|endoftext|>` via stop_token_ids. Rebuild prompt/completion from problem/solution with `to_prompt_completion` at load time so training always matches the current prompts.py.
- **SFT max length:** 1024 tokens (prompt + completion). Over-length examples dropped, never truncated (truncation would cut off the `\boxed{}` answer).
- **DPO data:** self-generated pairs from the SFT model, graded by the verifier (central to the argument)
- **DPO/GRPO prompt pool:** MATH train problems (prompts only), filtered to ones the SFT model solves 10–90% of the time
- **Eval:** MATH-500 (500 problems from the MATH test split; main benchmark, never used for tuning; report per level and subject); held-out MATH-train validation split for checkpoint selection; secondary: GSM8K test, AMC23/AIME24; base model zero-shot (chat format) and few-shot (plain text) as arm 0
- **Arms:** A = SFT, B = SFT→DPO (offline), C = SFT→GRPO, D = SFT→DPO→GRPO, + semi-online/iterative DPO (compute-matched). Optional: GRPO→DPO (G2D-style) for contrast with Verma & Ravindran.
- **Compute (decided):** Hugging Face **JupyterLab Space** `Maci10/DPO_King` (L4 for setup, **A100 large 80 GB, $2.50/hr** for real runs, billed per minute); PRO plan. Storage ephemeral → checkpoints/results pushed to HF Hub; code via git. **Pause after every session.** Raise/disable auto-sleep before long runs.
- **Data hosting:** private HF Hub repos `Maci10/dpo-king-sft_all` (train), `Maci10/dpo-king-math_val` (train), `Maci10/dpo-king-math500` (**test** split). HF token only via `huggingface-cli login` / `login()`, never in notebook code.
- **Grader (decided):** Hugging Face **Math-Verify** (`math-verify[antlr4_13_2]`, v0.9.0 tested) is the single grader for eval, DPO labeling and GRPO reward. Strict format: `get_last_boxed()` extracts the last `\boxed{}` first (no box = 0 reward), then gold and boxed answer are each wrapped in `$...$`, parsed, and compared with `verify(gold, pred)` (gold first). 18/18 package test cases pass. Same grader as Lanchantin et al. (Math-Verify).
- **Workflow:** Jupyter notebooks, one per stage; shared code in `posttrain/` (extract.py, cleaning.py, grader.py, prompts.py, data.py); GitHub repo `MaciSekander/DPO_King_Research` (currently public); W&B logging from day one.

## Environment (Space, 2026-11-14)
Python 3.12, vLLM 0.33.0, PyTorch 2.14.0+cu130, driver 580.178.04 / CUDA 13.0, L4. FlashInfer JIT fails with the image's CUDA 12.x nvcc → set `VLLM_USE_FLASHINFER_SAMPLER=0` (and `VLLM_ENABLE_V1_MULTIPROCESSING=0`) before importing vLLM; proper fix: CUDA 13 base image. See `claude/setup_runbook.md`.

## Data cleaning log (notebook 01) — for the paper's data section
- Sources: `EleutherAI/hendrycks_math` (7 subject configs, train only), `HuggingFaceH4/MATH-500`, `meta-math/MetaMathQA` (155,000 MATH_* rows)
- MATH train: answer extracted from last `\boxed{}`; 2 / 7,500 failed (empty `\boxed{}` in source) → dropped
- Validation: stratified by (subject, level), seed 42 → 6,998 SFT / 502 val
- MetaMathQA leak filter: 100% of `original_question` match a MATH train problem; 10,047 rows derived from val problems dropped (144,953 remain)
- Answer consistency (last box vs "The answer is:" label): strict AnsAug 93.8% / Rephrased 94.1%; loose normalization AnsAug 98.8% / Rephrased 99.1% (most disagreements are formatting). Disagreeing rows dropped. Trailing "The answer is:" line stripped; unboxed responses get `\boxed{label}` appended
- MATH_SV (~14k rows) dropped: 99% unboxed, manual inspection found reasoning contradicting the label (e.g., concludes 6, labels 8); "unknown variable X" format doesn't match MATH-500
- MATH_FOBAR (~14k rows) dropped: only 64 / 13,984 rows boxed, so the agreement check covered <1% of rows (boxes on the rest were appended by our pipeline, not evidence of correctness); same backward "unknown variable X" format as SV; dropped for consistency
- MetaMathQA rows inherit subject/level from their original MATH problem; `source` column kept for traceability
- Grader self-check: 0 / 500 MATH-500 gold answers fail Math-Verify self-grading
- Token lengths (Qwen3 tokenizer, prompt + completion, n = 122,667): p50 201, p90 361, p95 445, p99.9 ≈ 1,185, max 2,747. Coverage: ≤512 96.88%, ≤768 99.20%, ≤1024 99.80%, ≤1536 99.97%, ≤2048 99.99%
- Length filter at 1024 tokens: 251 examples dropped → **final SFT set: 122,416 examples**
- TODO: decontam counts from summary table; math_val self-check count

## Early results
- Smoke test (2026-11-14, L4, greedy, max 1024 tokens, chat format, zero-shot): base Qwen3-0.6B-Base 2/5 correct on first 5 MATH-500 problems; the 3 failures hit the token limit (repetition loops / repeated symbols). ~550 output tok/s with enforce_eager.

## Known limitations (for the paper)
- Qwen3 pretraining data may overlap with MATH; out of our control. Shared by all arms, so relative comparisons hold; arm 0 measures the starting point.
- Single small model; conclusions may not hold at scale. Optional: repeat key arms on Qwen3-1.7B-Base if budget allows.

## Open questions
- Which 2–3 transparency analyses are core vs appendix
- LoRA vs full fine-tune (leaning full — also makes weight-update analysis cleaner)
- GRPO max completion length (likely 768–1024)
- Add a GRPO→DPO (G2D-style) arm?

## Notebook layout
00_setup · 01_data_cleaning · 02_eval_harness · 03_eval_baseline (arm 0, rerun per arm) · 04_sft · 05_dpo_data · 06_dpo · 07_grpo · 08_analysis · 09_dpo_transparency
