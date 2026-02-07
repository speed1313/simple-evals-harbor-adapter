import argparse
import json
import os
from datetime import datetime

import pandas as pd

from . import common
from .browsecomp_eval import BrowseCompEval
from .config import EvalConfig, load_config
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .healthbench_eval import HealthBenchEval
from .healthbench_meta_eval import HealthBenchMetaEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .humaneval_eval import HumanEval
from .output import JobOutputManager
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from .sampler.claude_code_sampler import ClaudeCodeSampler
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.responses_sampler import ResponsesSampler
from .custom_types import Eval as EvalBase
from .simpleqa_eval import SimpleQAEval


def _build_eval(
    eval_name: str,
    debug_mode: bool,
    num_examples_override: int | None,
    n_repeats: int | None,
    n_threads: int | None,
    grading_sampler: ChatCompletionSampler,
    equality_checker: ChatCompletionSampler,
):
    """Build an eval object by name. Standalone so both CLI and config paths can use it."""
    num_examples = (
        num_examples_override
        if num_examples_override is not None
        else (5 if debug_mode else None)
    )
    match eval_name:
        case "mmlu":
            return MMLUEval(num_examples=1 if debug_mode else num_examples)
        case "math":
            return MathEval(
                equality_checker=equality_checker,
                num_examples=num_examples,
                n_repeats=1 if debug_mode else n_repeats or 10,
            )
        case "gpqa":
            return GPQAEval(
                n_repeats=1 if debug_mode else n_repeats or 10,
                num_examples=num_examples,
            )
        case "mgsm":
            return MGSMEval(
                num_examples_per_lang=10 if debug_mode else num_examples or 250
            )
        case "drop":
            return DropEval(
                num_examples=10 if debug_mode else num_examples,
                train_samples_per_prompt=3,
            )
        case "humaneval":
            return HumanEval(num_examples=10 if debug_mode else num_examples)
        case "simpleqa":
            return SimpleQAEval(
                grader_model=grading_sampler,
                num_examples=10 if debug_mode else num_examples,
                n_threads=n_threads,
            )
        case "browsecomp":
            return BrowseCompEval(
                grader_model=grading_sampler,
                num_examples=10 if debug_mode else num_examples,
            )
        case "healthbench":
            return HealthBenchEval(
                grader_model=grading_sampler,
                num_examples=10 if debug_mode else num_examples,
                n_repeats=n_repeats or 1,
                n_threads=n_threads or 120,
                subset_name=None,
            )
        case "healthbench_hard":
            return HealthBenchEval(
                grader_model=grading_sampler,
                num_examples=10 if debug_mode else num_examples,
                n_repeats=n_repeats or 1,
                n_threads=n_threads or 120,
                subset_name="hard",
            )
        case "healthbench_consensus":
            return HealthBenchEval(
                grader_model=grading_sampler,
                num_examples=10 if debug_mode else num_examples,
                n_repeats=n_repeats or 1,
                n_threads=n_threads or 120,
                subset_name="consensus",
            )
        case "healthbench_meta":
            return HealthBenchMetaEval(
                grader_model=grading_sampler,
                num_examples=10 if debug_mode else num_examples,
                n_repeats=n_repeats or 1,
                n_threads=n_threads or 120,
            )
        case _:
            raise Exception(f"Unrecognized eval type: {eval_name}")


def run_from_config(config: EvalConfig, args: argparse.Namespace):
    """Run evaluations from a YAML config file (Harbor-style)."""

    # Warn if --model or --eval are specified alongside --config
    if args.model:
        print("Warning: --model is ignored when using -c/--config (agents are defined in config)")
    if args.eval:
        print("Warning: --eval is ignored when using -c/--config (datasets are defined in config)")

    # Grading/checker models needed by evals
    grading_sampler = ChatCompletionSampler(
        model="gpt-4o-mini",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")

    # Resolve environment variables from config
    extra_env = config.environment.get_resolved_env()

    # Build samplers from config agents
    samplers: dict[str, ClaudeCodeSampler | ClaudeCompletionSampler] = {}
    for agent in config.agents:
        timeout = int(agent.override_timeout_sec * config.timeout_multiplier)

        # Filter out kwargs that are informational only (e.g. version)
        sampler_kwargs = {k: v for k, v in agent.kwargs.items() if k != "version"}
        if "version" in agent.kwargs:
            print(f"Info: agent '{agent.name}' specifies version={agent.kwargs['version']} (noted, not enforced)")

        if agent.name in ("claude-code", "claude-code-local"):
            use_docker = config.environment.use_docker
            sampler = ClaudeCodeSampler(
                model=agent.model_name,
                timeout=timeout,
                use_docker=use_docker,
                build_image=config.environment.force_build,
                extra_env=extra_env,
                **sampler_kwargs,
            )
            samplers[agent.name] = sampler
        else:
            sampler = ClaudeCompletionSampler(
                model=agent.model_name,
                system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
                **sampler_kwargs,
            )
            samplers[agent.name] = sampler

    # Build evals from config datasets
    debug_mode = args.debug
    num_examples_override = args.examples
    n_repeats = args.n_repeats
    n_threads = args.n_threads or config.orchestrator.n_concurrent_trials

    evals: dict[str, EvalBase] = {}
    for dataset in config.datasets:
        eval_name = dataset.eval_name
        evals[eval_name] = _build_eval(
            eval_name=eval_name,
            debug_mode=debug_mode,
            num_examples_override=num_examples_override,
            n_repeats=n_repeats,
            n_threads=n_threads,
            grading_sampler=grading_sampler,
            equality_checker=equality_checker,
        )

    # Setup output manager
    output_mgr = JobOutputManager(jobs_dir=config.jobs_dir, job_name=config.job_name)

    print(f"Config: job_name={config.job_name}, jobs_dir={config.jobs_dir}")
    print(f"Agents: {list(samplers.keys())}")
    print(f"Datasets: {list(evals.keys())}")

    # Run each agent x dataset combination
    started_at = datetime.now()
    all_results: list[dict[str, str | float | None]] = []
    # eval_task_info: maps "{agent}__{eval}" -> list of (dir_name, score) for job result.json
    eval_task_info: dict[str, list[tuple[str, float]]] = {}
    task_id_counter = 0

    for agent_name, sampler in samplers.items():
        for eval_name, eval_obj in evals.items():
            eval_key = f"{agent_name}__{eval_name}"

            # Point sampler trajectory to the job directory
            if hasattr(sampler, "trajectory_dir"):
                sampler.trajectory_dir = output_mgr.base_dir

            print(f"\nRunning {eval_name} with {agent_name}")
            result = eval_obj(sampler)

            # Get per-example scores
            individual_scores = (result.metadata or {}).get("individual_scores", [])

            # Create per-task directories and write per-task outputs
            task_entries: list[tuple[str, float]] = []
            for i, score in enumerate(individual_scores):
                task_dir, dir_name = output_mgr.create_task_dir(eval_name, task_id_counter + i)
                output_mgr.write_reward(task_dir, score)

                # Write per-task grading details
                grading_details = {"score": score}
                if result.convos and i < len(result.convos):
                    grading_details["convo"] = result.convos[i]
                output_mgr.write_grading_details(task_dir, grading_details)

                task_entries.append((dir_name, float(score)))

            task_id_counter += len(individual_scores)
            eval_task_info[eval_key] = task_entries

            metrics = (result.metrics or {}) | {"score": result.score}
            metrics = dict(sorted(metrics.items()))

            print(f"  Score: {result.score}")
            print(f"  Metrics: {metrics}")
            print(f"  Tasks: {len(individual_scores)}")

            all_results.append({
                "agent": agent_name,
                "eval": eval_name,
                "score": result.score,
            })

    # Write job-level result.json
    finished_at = datetime.now()
    if eval_task_info:
        result_path = output_mgr.write_job_result(started_at, finished_at, eval_task_info)
        print(f"\nJob result: {result_path}")

    # Print summary table
    if all_results:
        print("\n=== Summary ===")
        df = pd.DataFrame(all_results)
        if len(df) > 1:
            pivot = df.pivot(index="agent", columns="eval", values="score")
            print(pivot.to_markdown())
        else:
            for r in all_results:
                print(f"  {r['agent']} / {r['eval']}: score={r['score']}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Select a model by name. Also accepts a comma-separated list of models.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Select an eval by name. Also accepts a comma-separated list of evals.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeats to run. Only supported for certain evals.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Number of threads to run. Supported for SimpleQA, HealthBench, and HealthBenchMeta. Default: CPU count for SimpleQA, 120 for HealthBench.",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--claude-code-model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use with claude-code sampler (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to YAML config file (Harbor-style evaluation)",
    )

    args = parser.parse_args()

    # Dispatch to config-based runner if -c is provided
    if args.config:
        config = load_config(args.config)
        return run_from_config(config, args)

    # List of available model names (defined separately to avoid initializing samplers for --list-models)
    MODEL_NAMES = [
        # Reasoning Models
        "o3", "o3-temp-1", "o3_high", "o3_low",
        "o4-mini", "o4-mini_high", "o4-mini_low",
        "o1-pro", "o1", "o1_high", "o1_low", "o1-preview", "o1-mini",
        # GPT-4.1 models
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        # GPT-4o models
        "gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-08-06-temp-1",
        "gpt-4o-2024-05-13", "gpt-4o-mini",
        # GPT-4.5 model
        "gpt-4.5-preview",
        # GPT-4-turbo model
        "gpt-4-turbo-2024-04-09",
        # GPT-4 model
        "gpt-4-0613",
        # GPT-3.5 Turbo model
        "gpt-3.5-turbo-0125", "gpt-3.5-turbo-0125-temp-1",
        # Chatgpt models
        "chatgpt-4o-latest", "gpt-4-turbo-2024-04-09_chatgpt",
        # Claude models
        "claude-3-opus-20240229_empty", "claude-3-7-sonnet-20250219", "claude-3-haiku-20240307",
        # Claude Code (agent-based evaluation)
        "claude-code", "claude-code-local",
    ]

    if args.list_models:
        print("Available models:")
        for model_name in MODEL_NAMES:
            print(f" - {model_name}")
        return

    models = {
        # Reasoning Models
        "o3": ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
        ),
        "o3-temp-1": ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
            temperature=1.0,
        ),
        "o3_high": ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
            reasoning_effort="high",
        ),
        "o3_low": ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
            reasoning_effort="low",
        ),
        # Default == Medium
        "o4-mini": ResponsesSampler(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
        ),
        "o4-mini_high": ResponsesSampler(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
            reasoning_effort="high",
        ),
        "o4-mini_low": ResponsesSampler(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
            reasoning_effort="low",
        ),
        "o1-pro": ResponsesSampler(
            model="o1-pro",
            reasoning_model=True,
        ),
        "o1": OChatCompletionSampler(
            model="o1",
        ),
        "o1_high": OChatCompletionSampler(
            model="o1",
            reasoning_effort="high",
        ),
        "o1_low": OChatCompletionSampler(
            model="o1",
            reasoning_effort="low",
        ),
        "o1-preview": OChatCompletionSampler(
            model="o1-preview",
        ),
        "o1-mini": OChatCompletionSampler(
            model="o1-mini",
        ),
        # Default == Medium
        "o3-mini": OChatCompletionSampler(
            model="o3-mini",
        ),
        "o3-mini_high": OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="high",
        ),
        "o3-mini_low": OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="low",
        ),
        # GPT-4.1 models
        "gpt-4.1": ChatCompletionSampler(
            model="gpt-4.1-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4.1-temp-1": ChatCompletionSampler(
            model="gpt-4.1-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            temperature=1.0,
        ),
        "gpt-4.1-mini": ChatCompletionSampler(
            model="gpt-4.1-mini-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4.1-nano": ChatCompletionSampler(
            model="gpt-4.1-nano-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4o models
        "gpt-4o": ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-11-20": ChatCompletionSampler(
            model="gpt-4o-2024-11-20",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-08-06": ChatCompletionSampler(
            model="gpt-4o-2024-08-06",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-08-06-temp-1": ChatCompletionSampler(
            model="gpt-4o-2024-08-06",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            temperature=1.0,
        ),
        "gpt-4o-2024-05-13": ChatCompletionSampler(
            model="gpt-4o-2024-05-13",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-mini": ChatCompletionSampler(
            model="gpt-4o-mini-2024-07-18",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4.5 model
        "gpt-4.5-preview": ChatCompletionSampler(
            model="gpt-4.5-preview-2025-02-27",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4-turbo model
        "gpt-4-turbo-2024-04-09": ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        # GPT-4 model
        "gpt-4-0613": ChatCompletionSampler(
            model="gpt-4-0613",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        # GPT-3.5 Turbo model
        "gpt-3.5-turbo-0125": ChatCompletionSampler(
            model="gpt-3.5-turbo-0125",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        "gpt-3.5-turbo-0125-temp-1": ChatCompletionSampler(
            model="gpt-3.5-turbo-0125",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1.0,
        ),
        # Chatgpt models:
        "chatgpt-4o-latest": ChatCompletionSampler(
            model="chatgpt-4o-latest",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        ),
        "gpt-4-turbo-2024-04-09_chatgpt": ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        ),
        # Claude models:
        "claude-3-opus-20240229_empty": ClaudeCompletionSampler(
            model="claude-3-opus-20240229",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-3-7-sonnet-20250219": ClaudeCompletionSampler(
            model="claude-3-7-sonnet-20250219",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-3-haiku-20240307": ClaudeCompletionSampler(
            model="claude-3-haiku-20240307",
        ),
        # Claude Code (agent-based evaluation using Docker)
        "claude-code": ClaudeCodeSampler(
            model=args.claude_code_model,
            use_docker=True,
        ),
        "claude-code-local": ClaudeCodeSampler(
            model=args.claude_code_model,
            use_docker=False,
        ),
    }

    if args.model:
        models_chosen = args.model.split(",")
        for model_name in models_chosen:
            if model_name not in models:
                print(f"Error: Model '{model_name}' not found.")
                return
        models = {model_name: models[model_name] for model_name in models_chosen}

    print(f"Running with args {args}")

    grading_sampler = ChatCompletionSampler(
        model="gpt-4o-mini",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        return _build_eval(
            eval_name=eval_name,
            debug_mode=debug_mode,
            num_examples_override=args.examples,
            n_repeats=args.n_repeats,
            n_threads=args.n_threads,
            grading_sampler=grading_sampler,
            equality_checker=equality_checker,
        )

    if args.eval:
        evals_list = args.eval.split(",")
        evals = {}
        for eval_name in evals_list:
            try:
                evals[eval_name] = get_evals(eval_name, args.debug)
            except Exception:
                print(f"Error: eval '{eval_name}' not found.")
                return
    else:
        evals = {
            eval_name: get_evals(eval_name, args.debug)
            for eval_name in [
                "mmlu",
                "math",
                "gpqa",
                "mgsm",
                "drop",
                "humaneval",
                "simpleqa",
                "browsecomp",
                "healthbench",
                "healthbench_hard",
                "healthbench_consensus",
                "healthbench_meta",
            ]
        }

    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    print(f"Running the following evals: {list(evals.keys())}")
    print(f"Running evals for the following models: {list(models.keys())}")

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name}"
            # file stem should also include the year, month, day, and time in hours and minutes
            file_stem += f"_{date_str}"
            report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            # Sort metrics by key
            metrics = dict(sorted(metrics.items()))
            print(metrics)
            result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")

            full_result_filename = f"/tmp/{file_stem}{debug_suffix}_allresults.json"
            with open(full_result_filename, "w") as f:
                result_dict = {
                    "score": result.score,
                    "metrics": result.metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {full_result_filename}")

            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
