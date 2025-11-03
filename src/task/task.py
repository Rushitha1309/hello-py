from __future__ import annotations

from typing import Any, Callable

from anthropic.types import ToolUnionParam

from src.tools.tools import (
    imbalance_start,
    smote,
    set_class_weights,
    use_focal,
    train,
    set_threshold,
    eval_on_val,
    submit_and_grade,
    sweep_thresholds,
)


def build_tools() -> tuple[list[ToolUnionParam], dict[str, Callable[..., Any]]]:
    tools: list[ToolUnionParam] = [
        {
            "name": "imbalance_start",
            "description": "Initialize the imbalance task. Must be called first.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "seed": {"type": "integer"},
                    "max_steps": {"type": "integer"},
                    "tau": {"type": "number"},
                    "minority_frac": {"type": "number"},
                    "source": {"type": "string", "enum": ["synthetic", "kaggle"]},
                    "sample_n": {"type": "integer", "description": "Optional subsample size for Kaggle dataset"},
                },
                "required": ["seed", "max_steps", "tau"],
            },
        },
        {
            "name": "smote",
            "description": "Apply SMOTE oversampling to the training set.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ratio": {"type": "number"},
                    "k": {"type": "integer"},
                    "seed": {"type": "integer"},
                },
                "required": ["ratio", "k"],
            },
        },
        {
            "name": "set_class_weights",
            "description": "Set class weights using effective number of samples.",
            "input_schema": {"type": "object", "properties": {"beta": {"type": "number"}}, "required": ["beta"]},
        },
        {
            "name": "use_focal",
            "description": "Use focal loss with alpha and gamma.",
            "input_schema": {"type": "object", "properties": {"alpha": {"type": "number"}, "gamma": {"type": "number"}}, "required": ["alpha", "gamma"]},
        },
        {
            "name": "train",
            "description": "Train logistic regression for a few epochs.",
            "input_schema": {"type": "object", "properties": {"epochs": {"type": "integer"}, "lr": {"type": "number"}}, "required": ["epochs", "lr"]},
        },
        {
            "name": "set_threshold",
            "description": "Set decision threshold for the positive class.",
            "input_schema": {"type": "object", "properties": {"threshold": {"type": "number"}}, "required": ["threshold"]},
        },
        {
            "name": "eval_on_val",
            "description": "Evaluate current model on validation set.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "sweep_thresholds",
            "description": "Free helper: sweep thresholds on validation and set the best F1 threshold.",
            "input_schema": {"type": "object", "properties": {"num_points": {"type": "integer"}}},
        },
        {
            "name": "submit_answer",
            "description": "Submit the final solution for grading (returns pass/fail).",
            "input_schema": {"type": "object", "properties": {"answer": {"description": "Ignored; grader computes pass/fail"}}, "required": ["answer"]},
        },
    ]

    tool_handlers: dict[str, Callable[..., Any]] = {
        "imbalance_start": imbalance_start,
        "smote": smote,
        "set_class_weights": set_class_weights,
        "use_focal": use_focal,
        "train": train,
        "set_threshold": set_threshold,
        "eval_on_val": eval_on_val,
        "sweep_thresholds": sweep_thresholds,
        "submit_answer": lambda answer: {"answer": submit_and_grade()["pass"], "submitted": True},
    }

    return tools, tool_handlers


def build_prompt(
    *,
    source: str = "synthetic",
    sample_n: int | None = None,
    minority_frac: float = 0.03,
    max_steps: int = 5,
    tau: float = 0.6,
    seed: int = 42,
) -> str:
    if source == "kaggle":
        src_line = (
            f"Call imbalance_start first with seed={seed}, max_steps={max_steps}, tau={tau}, source='kaggle'"
            + (f", sample_n={sample_n}" if sample_n is not None else "")
            + "."
        )
    else:
        src_line = (
            f"Call imbalance_start first with seed={seed}, max_steps={max_steps}, tau={tau}, source='synthetic', minority_frac={minority_frac}."
        )
    return (
        f"You must improve minority F1 on a hidden TEST set in at most {max_steps} actions.\n"
        "Tools: imbalance_start(seed,max_steps,tau,minority_frac,source[,sample_n]), smote(ratio,k[,seed]), set_class_weights(beta), use_focal(alpha,gamma), "
        "train(epochs,lr), set_threshold(threshold), eval_on_val(), sweep_thresholds([num_points]), submit_answer(...).\n"
        f"Rules: (1) {src_line} (2) Only eval_on_val reveals metrics; TEST labels are hidden until submit. "
        "(3) Actions that count toward the budget: smote, set_class_weights, use_focal, set_threshold. train, eval_on_val and sweep_thresholds are free. "
        f"(4) Pass if TEST F1 >= tau ({tau}) and you stay within the action budget. "
        "(5) You must train at least once before submitting. After training, run sweep_thresholds and then submit_answer('submit')."
    )


