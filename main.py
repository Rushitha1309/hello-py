from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Any, Callable

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

from src.task.task import build_prompt, build_tools
from src.tools.tools import reset_session  # ensure fresh state per trial


def _append_log(log_path: str | None, text: str) -> None:
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-3-5-haiku-latest",
    verbose: bool = False,
    log_path: str | None = None,
) -> Any | None:
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        _append_log(log_path, f"\n=== Step {step + 1}/{max_steps} ===")

        try:
            response = await client.messages.create(
                model=model, max_tokens=1000, tools=tools, messages=messages
            )
        except Exception as e:
            _append_log(log_path, f"API error: {e}. Ending conversation gracefully.")
            break

        has_tool_use = False
        tool_results = []
        submitted_payload: Any | None = None

        for content in response.content:
            if content.type == "text":
                _append_log(log_path, f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name
                if tool_name in tool_handlers:
                    handler = tool_handlers[tool_name]
                    tool_input = content.input
                    _append_log(log_path, "\nTool input:")
                    try:
                        _append_log(log_path, json.dumps(tool_input, indent=2))
                    except Exception:
                        _append_log(log_path, str(tool_input))
                    if isinstance(tool_input, dict):
                        result = handler(**tool_input)
                    else:
                        result = handler(tool_input)
                    _append_log(log_path, "Tool output:")
                    try:
                        _append_log(log_path, json.dumps(result, indent=2))
                    except Exception:
                        _append_log(log_path, str(result))
                    if tool_name == "submit_answer":
                        submitted_payload = result
                    # If threshold sweep just ran, auto-submit immediately to ensure grading within short convs
                    if tool_name == "sweep_thresholds" and "submit_answer" in tool_handlers:
                        try:
                            _append_log(log_path, "Auto-submitting after sweep_thresholds...")
                            auto_result = tool_handlers["submit_answer"]("auto_submit")
                            submitted_payload = auto_result
                        except Exception as e:
                            _append_log(log_path, f"Auto-submit after sweep failed: {e}")
                    tool_results.append({"type": "tool_result", "tool_use_id": content.id, "content": json.dumps(result)})

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            # Add a budget reminder to the next turn
            try:
                if "get_budget" in tool_handlers:
                    b = tool_handlers["get_budget"]()
                    reminder = f"Reminder: You may use up to {b.get('max_steps')} paid actions total; remaining: {b.get('remaining')} (steps_used={b.get('steps_used')}). Free tools: train, eval_on_val, sweep_thresholds."
                    messages.append({"role": "user", "content": reminder})
                    _append_log(log_path, reminder)
            except Exception:
                pass
            if submitted_payload is not None:
                ans = submitted_payload.get("answer") if isinstance(submitted_payload, dict) else submitted_payload
                _append_log(log_path, f"\nAgent submitted answer: {ans}")
                return submitted_payload
        else:
            _append_log(log_path, "\nNo tool use in response, ending loop.")
            break

    # Auto-submit whatever we have if conversation ends without explicit submission
    if "submit_answer" in tool_handlers:
        try:
            _append_log(log_path, "\nAuto-submitting current solution at conversation end...")
            auto_result = tool_handlers["submit_answer"]("auto_submit")
            submitted_answer = auto_result.get("answer")
            _append_log(log_path, f"Auto-submission result: {submitted_answer}")
            return auto_result
        except Exception as e:
            _append_log(log_path, f"Auto-submit failed: {e}")
    _append_log(log_path, f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_trials(
    num_runs: int = 10,
    verbose: bool = False,
    model: str = "claude-haiku-4-5-20251001",
    source: str = "synthetic",
    sample_n: int | None = None,
    minority_frac: float = 0.03,
    max_steps: int = 5,
    tau: float = 0.3,
    seed: int = 42,
    sequential: bool = False,
    conv_steps: int = 3,
):
    # Clamp conversation steps to a safe minimum
    conv_steps = max(1, int(conv_steps))
    # Enforce sequential runs to preserve cross-run context
    if not sequential and num_runs > 1:
        sequential = True
    # Prepare logs directory and file
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"run_{timestamp}.log")
    _append_log(log_path, f"Args: model={model}, source={source}, sample_n={sample_n}, minority_frac={minority_frac}, max_steps={max_steps}, tau={tau}, seed={seed}, runs={num_runs}, conv_steps={conv_steps}, sequential={sequential}")
    tools, handlers = build_tools()
    # Ensure submit_answer tool exists and always wrap its handler to include detailed fields
    tool_names = {t.get("name") for t in tools}
    if "submit_answer" not in tool_names:
        tools.append({
            "name": "submit_answer",
            "description": "Submit the final solution for grading (returns pass/fail).",
            "input_schema": {"type": "object", "properties": {"answer": {"description": "Ignored; grader computes pass/fail"}}, "required": ["answer"]},
        })
    try:
        from src.tools.tools import submit_and_grade  # type: ignore

        def _default_submit(answer: Any) -> dict[str, Any]:
            res = submit_and_grade()
            return {
                "answer": res.get("pass", False),
                "submitted": True,
                "test_f1": res.get("test_f1"),
                "steps_used": res.get("steps_used"),
                "max_steps": res.get("max_steps"),
                "actions_used": res.get("actions_used"),
            }

        handlers["submit_answer"] = _default_submit
    except Exception:
        handlers["submit_answer"] = lambda answer: {"answer": False, "submitted": True, "test_f1": None, "actions_used": []}
    results = []
    # Build per-trial prompts with varied seeds to diversify runs
    prompts: list[str] = []
    for i in range(num_runs):
        trial_seed = seed + i
        trial_prompt = build_prompt(
            source=source,
            sample_n=sample_n,
            minority_frac=minority_frac,
            max_steps=max_steps,
            tau=tau,
            seed=trial_seed,
        )
        prompts.append(trial_prompt)
    per_run: list[dict[str, Any]] = []
    if sequential:
        per_run = []
        tried_sets: list[list[str]] = []
        for idx in range(num_runs):
            # Build prompt with a small history hint of previous runs
            history_note = ""
            if tried_sets:
                history_note = ("\nHistory: Previously tried paid action sets: " + ", ".join(["{" + ", ".join(s) + "}" for s in tried_sets]) + ". Prefer a different combination this time.")
            trial_prompt = prompts[idx] + history_note
            # Fresh session per trial to diversify runs
            # Fresh session per trial to diversify runs
            try:
                reset_session()
            except Exception:
                pass
            _append_log(log_path, f"\n========== Run {idx + 1}/{num_runs} ==========")
            r = await run_agent_loop(prompt=trial_prompt, tools=tools, tool_handlers=handlers, max_steps=conv_steps, verbose=False, model=model, log_path=log_path)
            if isinstance(r, dict):
                per_run.append({"pass": bool(r.get("answer") or r.get("pass")), "test_f1": r.get("test_f1"), "actions_used": r.get("actions_used"), "steps_used": r.get("steps_used"), "max_steps": r.get("max_steps")})
                _append_log(log_path, f"Summary: pass={bool(r.get('answer') or r.get('pass'))}, test_f1={r.get('test_f1')}, actions={r.get('actions_used')}")
                # Record the set of paid action names used in this run
                actions = r.get("actions_used") or []
                names: list[str] = []
                for a in actions:
                    try:
                        name = str(a).split("(")[0]
                    except Exception:
                        name = str(a)
                    if name and name not in names:
                        names.append(name)
                tried_sets.append(names)
            else:
                per_run.append({"pass": bool(r), "test_f1": None, "actions_used": None, "steps_used": None, "max_steps": None})
                _append_log(log_path, f"Summary: pass={bool(r)}, test_f1=None, actions=None")
    else:
        # Concurrent mode: build and await tasks
        tasks = [
            run_agent_loop(prompt=prompts[i], tools=tools, tool_handlers=handlers, max_steps=conv_steps, verbose=False, model=model, log_path=log_path)
            for i in range(num_runs)
        ]
        for idx, coro in enumerate(asyncio.as_completed(tasks)):
            _append_log(log_path, f"\n========== Run {idx + 1}/{num_runs} (async) ==========")
            r = await coro
            if isinstance(r, dict):
                per_run.append({"pass": bool(r.get("answer") or r.get("pass")), "test_f1": r.get("test_f1"), "actions_used": r.get("actions_used"), "steps_used": r.get("steps_used"), "max_steps": r.get("max_steps")})
                _append_log(log_path, f"Summary: pass={bool(r.get('answer') or r.get('pass'))}, test_f1={r.get('test_f1')}, actions={r.get('actions_used')}")
            else:
                per_run.append({"pass": bool(r), "test_f1": None, "actions_used": None, "steps_used": None, "max_steps": None})
                _append_log(log_path, f"Summary: pass={bool(r)}, test_f1=None, actions=None")
    successes = sum(1 for r in per_run if r.get("pass"))
    pass_rate = (successes / num_runs) * 100
    print("\n" + "=" * 60)
    print("Final Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print("=" * 60)
    print(f"Logs saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run imbalance RL task trials")
    parser.add_argument("--runs", type=int, default=10, help="Number of trials to run")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001", help="Anthropic model name")
    parser.add_argument("--source", type=str, choices=["synthetic", "kaggle"], default="synthetic", help="Data source")
    parser.add_argument("--sample-n", type=int, default=None, help="Optional Kaggle subsample size")
    parser.add_argument("--minority-frac", type=float, default=0.03, help="Synthetic minority fraction")
    parser.add_argument("--max-steps", type=int, default=5, help="Action budget exposed to the agent")
    parser.add_argument("--tau", type=float, default=0.3, help="Target F1 threshold for pass")
    parser.add_argument("--seed", type=int, default=42, help="Seed used in the prompt")
    parser.add_argument("--sequential", action="store_true", help="Run trials sequentially for easier reading")
    parser.add_argument("--conv-steps", type=int, default=10, help="Max assistant-tool turns per trial")
    args = parser.parse_args()
    asyncio.run(
        run_trials(
            num_runs=args.runs,
            verbose=args.verbose,
            model=args.model,
            source=args.source,
            sample_n=args.sample_n,
            minority_frac=args.minority_frac,
            max_steps=args.max_steps,
            tau=args.tau,
            seed=args.seed,
            sequential=args.sequential,
            conv_steps=args.conv_steps,
        )
    )
