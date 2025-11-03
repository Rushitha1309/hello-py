# RL Task: Evaluating LLM Strategies under Extreme Class Imbalance

## 1. Introduction

### 1.1 About
An RL-style task designed for LLMs to tackle **extreme class imbalance**, equipping the agent with a suite of tools, including: 
- SMOTE
- class weighting
- focal loss
- tau thresholding (for minimum passing Test F1)
- training
- validation
- submission

The primary objective is to **maximize F1 score on a concealed test set**, all while operating under a strict step (action) budget. By design, the task aims for a 10–40% model pass rate, consistent with task evaluation guidelines.

### 1.2 Data
- **Synthetic Gaussian blobs**: Artificially generated datasets where two classes are drawn from Gaussian distributions. The minority fraction (class imbalance) is configurable, making this dataset useful for controlled experimentation and benchmarking strategies for extreme imbalance.
- **[Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) dataset** (`creditcardfraud`): A popular real-world dataset containing anonymized credit card transactions labeled as fraudulent or legitimate. It is extremely imbalanced, with fraudulent cases making up only a tiny fraction of the data. The project loads this dataset via `kagglehub`, and supports optional subsampling to speed up experiments.

> [!NOTE]
> The minimum F1 score set to achieve is 0.75 for the Kaggle imbalanced dataset.

### 1.3 Approach
The agent plans over multiple conversation turns, choosing from paid and free tools to maximize F1 under a budget. Core loop:

- **Initialize**: Call `imbalance_start(...)` with data source and budget (`--source`, `--sample-n`/`--minority-frac`, `--max-steps`, `--tau`, `--seed`). Data is split into train/val/test; only validation metrics are visible during planning.
- **Paid tools (consume budget)**:
  - `smote(ratio,k[,seed])`: synthesize minority examples.
  - `set_class_weights(beta)`: rebalance CE loss by effective number of samples.
  - `use_focal(alpha,gamma)`: switch to focal loss for hard examples.
  - `set_threshold(threshold)`: set decision threshold (often after sweeping).
- **Free tools (no budget cost)**:
  - `train(epochs,lr)`: explicit training; also triggered automatically after each paid action.
  - `eval_on_val()`: see current precision/recall/F1 on validation.
  - `sweep_thresholds([num_points])`: scan thresholds on validation; updates best threshold in-session.
  - `submit_answer(...)`: triggers grading on the hidden test set.
- **Auto-training**: After every effectful paid action we retrain for a few epochs (free) so evaluations reflect the latest configuration.
- **Budget reminders**: After each turn the runner injects a `get_budget` summary so the agent knows remaining paid actions and can plan the next steps.
- **Grading**: When the agent submits (or the runner auto-submits), we compute TEST F1 and compare to `tau`. The grader only checks the final outcome and that paid actions used ≤ `max_steps`.
- **Exploration**: Across sequential runs, we append a short history of previously tried paid-action sets to the next prompt to discourage repeating the same strategy while keeping runs independent.

**Flow**
```
User Prompt → imbalance_start → [Paid tools]* → (free) train/eval → sweep_thresholds (free) → set_threshold → submit_answer → Grader
```

### 1.4 Why use focal loss and SMOTE
- SMOTE synthesizes minority examples to mitigate skew and improve recall.
- Focal loss down-weights easy negatives and focuses learning on hard positives, which is critical under extreme imbalance.
Used together (plus threshold tuning), they often yield better F1 than class weighting alone.

## 2. Setup

1) Install deps (uses `uv`):
```bash
uv run --version  # verify uv is installed
```

2) Set Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

3) Run a quick smoke test (synthetic):
```bash
uv run main.py --runs 3 --sequential --conv-steps 3 --source synthetic --minority-frac 0.2 --max-steps 4 --tau 0.7 --model claude-haiku-4-5-20251001
```

## 3. Implementation

### 3.1 Example Input/Output
Command:
```bash
uv run main.py --runs 5 --conv-steps 5 \
  --source kaggle --sample-n 50000 --max-steps 4 --tau 0.75 --model claude-haiku-4-5-20251001
```

Terminal output (now only final results):
```text
============================================================
Final Results:
  Passed: 1/5
  Failed: 4/5
  Pass Rate: 20.0%
============================================================
Logs saved to: logs/run_YYYYMMDD_HHMMSS.log
```

All detailed verbose logs (assistant messages, tool inputs/outputs, auto-submission, per-run summaries) are written to the [logs](logs/) folder.

### 3.1.1 Context and Memory
- Per-turn (within a run): The task session persists; model parameters, thresholds, and counters carry over. After each turn the runner injects a budget reminder (via `get_budget`) like: “You may use up to <max_steps> paid actions; remaining: <remaining> (steps_used=<n>). Free tools: train, eval_on_val, sweep_thresholds.”
- Cross-run: Runs execute sequentially when multiple runs are requested so the next prompt can include a brief history of previously tried paid-action sets. This nudges exploration without sharing hidden state. Each run still starts with a fresh task session.

### 3.1.2 The Prompt
- Where it’s defined: `src/task/task.py::build_prompt` constructs the user prompt from CLI flags (`--source`, `--sample-n`, `--minority-frac`, `--max-steps`, `--tau`, `--seed`).
- What it contains:
  - Problem statement and success criteria (maximize TEST F1 ≥ `tau` within `max_steps` paid actions).
  - Tool catalog and rules, clearly distinguishing paid vs free tools.
  - Data-source clause (synthetic with `minority_frac` or Kaggle with `sample_n`).
  - Nudge to “train at least once”, then “sweep_thresholds” and submit.
- How it’s used: The prompt is sent alongside the tool schemas to the model. The runner may append a short history line between runs and a budget reminder between turns. Grading occurs only when the agent calls `submit_answer` (or the runner auto-submits), comparing TEST F1 vs `tau` and checking budget.

### 3.2 Parameters

| Flag | Description | Default |
|---|---|---|
| `--runs` | Number of trials to run | `10` |
| `--verbose` | Print detailed tool I/O and assistant messages | off |
| `--model` | Anthropic model ID | `claude-haiku-4-5-20251001` |
| `--source` | Data source: `synthetic` or `kaggle` | `synthetic` |
| `--sample-n` | Kaggle subsample size (for speed) | `None` |
| `--minority-frac` | Synthetic minority class fraction (only used with `synthetic` data) | `0.03` |
| `--max-steps` | Paid action budget available to agent | `5` |
| `--tau` | Minimum Test F1 score to pass | `0.3` |
| `--seed` | Seed used in prompt; diversified per run as `seed+i` | `42` |
| `--sequential` | Run trials sequentially for clean logs | off |
| `--conv-steps` | Max conversation turns (assistant-tool messages) | `10` |

Notes:
- Free tools do not consume budget: `train`, `eval_on_val`, `sweep_thresholds`.
- Paid tools consume budget: `smote`, `set_class_weights`, `use_focal`, `set_threshold`.
- `submit_answer` is always available; the runner auto-submits when needed.
- Terminal shows only “Final Results”; see the generated log file in `logs/` for complete per-run verbose output.
- Budget reminders: The runner appends a `get_budget` summary after each assistant/tool turn so the agent knows remaining paid actions.

### 3.3 Known errors
- `step_budget_exceeded`: Too many paid actions. Solution: raise `--max-steps`, avoid repeating identical paid actions (free no-op checks are implemented), or submit earlier.
- `anthropic.NotFoundError` on model: Use a model ID your key supports (e.g., `claude-haiku-4-5-20251001`).
- Rate limiting: Retry later; the runner ends gracefully when API errors occur.
- Short conversations ending without submit: The runner auto-submits at end or after `sweep_thresholds`.
- SMOTE ratio edge cases: Invalid ratios are rejected and no-oped safely.

## 4. Results
The task targets a 10–40% pass rate to balance difficulty and learning signal. Typical runs on Kaggle subsamples (e.g., `--sample-n 50k`, `--max-steps 4`, `--tau 0.75`) produce a mix of fails and occasional passes, with per-run test F1 and actions reported, and a final aggregated pass rate.

## 5. Scope
This task is intended for training and evaluating LLM planners on imbalanced classification. Multiple solution paths are valid (SMOTE-first vs. class weighting-first, focal toggles, threshold sweeps). The grader checks only final TEST F1 and budget compliance, allowing exploration.

## 6. Conclusion
This repository provides a concise RL-style environment where an agent learns to tackle class imbalance with practical tools (SMOTE, focal loss, thresholding). It is configurable via CLI, supports real-world data, and produces clear summaries suitable for RL training signals.
