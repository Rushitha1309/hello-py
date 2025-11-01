from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Callable

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

from src.task.task import build_prompt, build_tools

async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-3-5-haiku-latest",
    verbose: bool = True,
) -> Any | None:
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=1000, tools=tools, messages=messages
        )

        has_tool_use = False
        tool_results = []
        submitted_answer = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name
                if tool_name in tool_handlers:
                    handler = tool_handlers[tool_name]
                    tool_input = content.input
                    if verbose:
                        print("\nTool input:")
                        try:
                            print(json.dumps(tool_input, indent=2))
                        except Exception:
                            print(tool_input)
                    if isinstance(tool_input, dict):
                        result = handler(**tool_input)
                    else:
                        result = handler(tool_input)
                    if verbose:
                        print("Tool output:")
                        try:
                            print(json.dumps(result, indent=2))
                        except Exception:
                            print(result)
                    if tool_name == "submit_answer":
                        submitted_answer = result.get("answer")
                    tool_results.append({"type": "tool_result", "tool_use_id": content.id, "content": json.dumps(result)})

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
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
    conv_steps: int = 10,
):
    prompt = build_prompt(
        source=source,
        sample_n=sample_n,
        minority_frac=minority_frac,
        max_steps=max_steps,
        tau=tau,
        seed=seed,
    )
    tools, handlers = build_tools()
    results = []
    tasks = [
        run_agent_loop(prompt=prompt, tools=tools, tool_handlers=handlers, max_steps=conv_steps, verbose=verbose, model=model)
        for _ in range(num_runs)
    ]
    if sequential:
        for t in tasks:
            r = await t
            results.append(r is True)
    else:
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r is True)
    successes = sum(results)
    pass_rate = (successes / num_runs) * 100
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print("=" * 60)


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


