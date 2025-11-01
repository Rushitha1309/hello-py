hello-py
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.

## Imbalance RL-style Task (separate runner)

To run the imbalance-handling prototype without modifying `main.py`:

```
uv run src/runner/runner.py
```

This launches an agent that must improve minority F1 on a hidden test set within an action budget using deterministic tools (SMOTE, class weights, focal loss, training, thresholding, validation eval, and final submission for grading).

### Using the Kaggle Credit Card Fraud dataset

Dependencies: `pandas`, `kagglehub` (already in pyproject).

- Optional: set `CREDITCARD_DATA_DIR` to a folder containing `creditcard.csv` to skip download.
- Otherwise, the agent can call `imbalance_start(..., source='kaggle', sample_n=20000)` to download via KaggleHub and subsample for speed.

Note: Download requires network access and may need Kaggle auth depending on your environment.

### Verbose logging and CLI flags

- Show detailed tool inputs/outputs and assistant messages:
```
uv run src/runner/runner.py --verbose
```

- Change number of trials, model, and conversation turns:
```
uv run src/runner/runner.py --runs 20 --model claude-haiku-4-5-20251001 --conv-steps 12
```

- Select data source and parameters:
```
# Synthetic with custom imbalance
uv run src/runner/runner.py --source synthetic --minority-frac 0.05 --max-steps 7 --tau 0.4 --conv-steps 12 --verbose

# Kaggle with subsampling for speed
uv run src/runner/runner.py --source kaggle --sample-n 20000 --max-steps 7 --tau 0.4 --conv-steps 12 --verbose
```

## Imbalance RL-style Task

The default run now executes an imbalance-handling task where the agent must improve minority F1 on a hidden test set by invoking deterministic tools within an action budget.

Tools exposed to the agent:

- imbalance_start(seed, max_steps, tau, minority_frac)
- smote(ratio, k, [seed])
- set_class_weights(beta)
- use_focal(alpha, gamma)
- train(epochs, lr)
- set_threshold(threshold)
- eval_on_val()
- submit_answer(any)  ← triggers grading; returns pass/fail

Pass condition: TEST F1 ≥ tau and actions used ≤ max_steps.
