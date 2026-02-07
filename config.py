"""
YAML config file parsing for Harbor-style evaluation runs.

Supports configs like:

    job_name: test-simpleqa
    jobs_dir: jobs
    n_attempts: 1
    timeout_multiplier: 1.0
    orchestrator:
      type: local
      n_concurrent_trials: 4
      quiet: false
    environment:
      type: docker
      force_build: true
      delete: true
      env:
        - OPENAI_API_KEY=${OPENAI_API_KEY}
    agents:
      - name: claude-code
        model_name: claude-opus-4-6
        kwargs:
          version: "2.1.32"
        override_timeout_sec: 3000
    datasets:
      - path: datasets/simpleqa
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OrchestratorConfig:
    type: str = "local"
    n_concurrent_trials: int = 4
    quiet: bool = False


@dataclass
class EnvironmentConfig:
    type: str = "docker"
    force_build: bool = True
    delete: bool = True
    env: list[str] = field(default_factory=list)

    @property
    def use_docker(self) -> bool:
        return self.type == "docker"

    def get_resolved_env(self) -> dict[str, str]:
        """Resolve env entries like 'KEY=${VAR_NAME}' using os.environ."""
        resolved: dict[str, str] = {}
        for entry in self.env:
            if "=" not in entry:
                continue
            key, value = entry.split("=", 1)
            resolved[key] = _interpolate_env_vars(value)
        return resolved


@dataclass
class AgentConfig:
    name: str
    model_name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    override_timeout_sec: int = 3000


@dataclass
class DatasetConfig:
    path: str

    @property
    def eval_name(self) -> str:
        """Extract eval name from path basename, e.g. 'datasets/simpleqa' -> 'simpleqa'."""
        return Path(self.path).name


@dataclass
class EvalConfig:
    job_name: str
    jobs_dir: str = "jobs"
    n_attempts: int = 1
    timeout_multiplier: float = 1.0
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agents: list[AgentConfig] = field(default_factory=list)
    datasets: list[DatasetConfig] = field(default_factory=list)


def _interpolate_env_vars(value: str) -> str:
    """Resolve ${VAR_NAME} patterns from os.environ."""
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        env_val = os.environ.get(var_name)
        if env_val is None:
            raise ValueError(
                f"Environment variable '{var_name}' not set (referenced in config)"
            )
        return env_val

    return re.sub(r"\$\{([^}]+)\}", replacer, value)


def load_config(path: str) -> EvalConfig:
    """Read a YAML config file and return a typed EvalConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw).__name__}")

    orchestrator = OrchestratorConfig(**raw.get("orchestrator", {}))
    environment = EnvironmentConfig(**raw.get("environment", {}))

    agents = [AgentConfig(**a) for a in raw.get("agents", [])]
    datasets = [DatasetConfig(**d) for d in raw.get("datasets", [])]

    return EvalConfig(
        job_name=raw["job_name"],
        jobs_dir=raw.get("jobs_dir", "jobs"),
        n_attempts=raw.get("n_attempts", 1),
        timeout_multiplier=raw.get("timeout_multiplier", 1.0),
        orchestrator=orchestrator,
        environment=environment,
        agents=agents,
        datasets=datasets,
    )
