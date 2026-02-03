# SimpleQA Parity Experiments

This repository evaluates language models on the SimpleQA benchmark, including agent-based evaluation using Claude Code following [Harbor's](https://github.com/laude-institute/harbor) approach.

## Quick Start

```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```


```bash
# Run SimpleQA with Claude Code (agent-based, Docker)
uv run python main.py --model claude-code --claude-code-model claude-opus-4-20250514 --eval simpleqa --examples 5
```

## Experimental Results

### SimpleQA Benchmark (n=50)

| Model | Type | Correct | Incorrect | Not Attempted | Accuracy (Given Attempted) | F1 Score |
|-------|------|---------|-----------|---------------|---------------------------|----------|
| claude-opus-4-20250514 | Agent (Claude Code) | 84% | 10% | 6% | 89.4% | 0.866 |

**Notes:**
- `claude-code` runs the specified model via Claude Code CLI inside a Docker container following Harbor's experimental setup
- The agent must write answers to `/workspace/answer.txt` - if the file is not created, the task is marked as failed
- Grading is performed by gpt-4o-mini using the same criteria as LLM-based evaluation

### Evaluation Details

- **Models:** claude-opus-4-20250514 (via Claude Code agent)
- **Timeout:** 300 seconds per task
- **Grader:** gpt-4o-mini

## Agent Trajectories

When using `claude-code` or `claude-code-local`, agent trajectories are saved to `/tmp/claude-code-trajectories/`:

```
/tmp/claude-code-trajectories/
├── task_0001/
│   ├── claude-code-output.txt    # Raw stream-json output
│   ├── logs/agent/               # Execution logs
│   ├── workspace/answer.txt      # Agent's answer
│   └── sessions/                 # Claude Code session JSONL (full trajectory)
├── task_0002/
│   └── ...

```
