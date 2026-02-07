# SimpleQA Parity Experiments

This repository evaluates language models on the SimpleQA benchmark, including agent-based evaluation using Claude Code following [Harbor's](https://github.com/laude-institute/harbor) approach.

## Quick Start

```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```


```bash
# Run SimpleQA with Claude Code (agent-based, Docker)
uv run python main.py -c simpleqa_parity_claude_opus4_6.yaml --examples 50
```

## Experimental Results

### SimpleQA Benchmark (n=50)

| Model | Type | Accuracy | Notes |
|-------|------|---------|-----------|
| claude-opus-4-6 | Agent (Claude Code) | 0.967 ± 0.025 | 50 examples over 3 runs |

**Notes:**
- `claude-code` runs the specified model via Claude Code CLI inside a Docker container following Harbor's experimental setup
- The agent must write answers to `/workspace/answer.txt` - if the file is not created, the task is marked as failed
- Grading is performed by gpt-4o-mini using the same criteria as LLM-based evaluation

### Evaluation Details

- **Models:** claude-opus-4-6 (via Claude Code agent)
- **Timeout:** 3000 seconds per task
- **Grader:** gpt-4o-mini
