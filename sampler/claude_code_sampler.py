"""
Claude Code CLI sampler for agent-based evaluation.

This sampler runs Claude Code CLI (the agent) inside a Docker container,
following Harbor's approach: https://github.com/laude-institute/harbor

The sampler:
1. Builds/uses a Docker image with Claude Code installed
2. Runs the agent inside the container with the prompt
3. Instructs the agent to write the answer to a file
4. Parses the answer from that file
"""

import os
import shlex
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ..custom_types import MessageList, SamplerBase, SamplerResponse


# Tools allowed for Claude Code (following Harbor's configuration)
ALLOWED_TOOLS = [
    "Bash",
    "Edit",
    "Write",
    "Read",
    "Glob",
    "Grep",
    "LS",
    "WebFetch",
    "NotebookEdit",
    "NotebookRead",
    "TodoRead",
    "TodoWrite",
    "Agent",
    "Skill",
    "SlashCommand",
    "Task",
    "WebSearch",
]

# Docker image name for Claude Code
CLAUDE_CODE_IMAGE = "claude-code-eval"

# Installation script for Claude Code (following Harbor's install-claude-code.sh.j2)
INSTALL_SCRIPT = """#!/bin/bash
set -euo pipefail

# Install dependencies based on package manager
if command -v apk &> /dev/null; then
    apk add --no-cache curl bash git
elif command -v apt-get &> /dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y curl bash git ca-certificates
fi

# Install Claude Code using the official installer
curl -fsSL https://claude.ai/install.sh | bash

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
claude --version
"""

# Dockerfile for Claude Code environment
DOCKERFILE = """
FROM ubuntu:22.04

# Install base dependencies
RUN apt-get update && \\
    DEBIAN_FRONTEND=noninteractive apt-get install -y \\
    curl \\
    bash \\
    git \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Create directories (following Harbor's structure)
RUN mkdir -p /workspace /logs/agent /output

WORKDIR /workspace

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash && \\
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Set PATH
ENV PATH="/root/.local/bin:$PATH"

# Create Claude config directories
RUN mkdir -p /root/.claude/debug \\
    /root/.claude/projects/-workspace \\
    /root/.claude/shell-snapshots \\
    /root/.claude/statsig \\
    /root/.claude/todos

# Default command
CMD ["bash"]
"""


class ClaudeCodeSampler(SamplerBase):
    """
    Sampler that runs Claude Code CLI inside a Docker container for agent-based evaluation.

    This follows Harbor's approach:
    1. Uses a Docker container with Claude Code installed
    2. Passes the prompt via the -p flag
    3. Instructs the agent to write the answer to /workspace/answer.txt
    4. Parses the answer from that file
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_thinking_tokens: int | None = None,
        timeout: int = 300,  # 5 minutes default timeout
        solution_path: str = "/workspace/answer.txt",
        allowed_tools: list[str] | None = None,
        docker_image: str | None = None,
        use_docker: bool = True,
        build_image: bool = True,
        trajectory_dir: str | None = None,
    ):
        """
        Initialize the Claude Code sampler.

        Args:
            model: The Claude model to use (e.g., "claude-sonnet-4-20250514")
            max_thinking_tokens: Maximum thinking tokens for extended thinking
            timeout: Timeout in seconds for each evaluation
            solution_path: Path inside container where agent writes the answer
            allowed_tools: List of tools to allow (defaults to Harbor's list)
            docker_image: Custom Docker image name (defaults to "claude-code-eval")
            use_docker: Whether to use Docker (set False for local execution)
            build_image: Whether to build the Docker image if it doesn't exist
            trajectory_dir: Directory to save agent trajectories (default: /tmp/claude-code-trajectories)
        """
        self.model = model
        self.max_thinking_tokens = max_thinking_tokens
        self.timeout = timeout
        self.solution_path = solution_path
        self.allowed_tools = allowed_tools or ALLOWED_TOOLS
        self.docker_image = docker_image or CLAUDE_CODE_IMAGE
        self.use_docker = use_docker
        self.image_format = "base64"
        self.trajectory_dir = trajectory_dir or "/tmp/claude-code-trajectories"
        self._task_counter = 0

        # Create trajectory directory
        os.makedirs(self.trajectory_dir, exist_ok=True)

        # Verify Docker is available if using Docker
        if self.use_docker:
            self._verify_docker()
            if build_image:
                self._ensure_docker_image()
        else:
            self._verify_claude_code_local()

    def _verify_docker(self) -> None:
        """Verify Docker is installed and running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Docker is not running: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError("Docker is not installed. Please install Docker first.")

    def _verify_claude_code_local(self) -> None:
        """Verify Claude Code is installed locally (for non-Docker mode)."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Claude Code CLI check failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: curl -fsSL https://claude.ai/install.sh | bash"
            )

    def _ensure_docker_image(self) -> None:
        """Build the Docker image if it doesn't exist."""
        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", self.docker_image],
            capture_output=True,
            text=True,
        )

        if result.stdout.strip():
            print(f"Docker image '{self.docker_image}' already exists")
            return

        print(f"Building Docker image '{self.docker_image}'...")

        # Create a temporary directory for the build context
        with tempfile.TemporaryDirectory() as build_dir:
            dockerfile_path = Path(build_dir) / "Dockerfile"
            dockerfile_path.write_text(DOCKERFILE)

            result = subprocess.run(
                ["docker", "build", "-t", self.docker_image, build_dir],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for build
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to build Docker image: {result.stderr}")

            print(f"Successfully built Docker image '{self.docker_image}'")

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ) -> dict[str, Any]:
        # Note: fovea parameter is unused but kept for interface compatibility
        _ = fovea
        return {
            "type": "image",
            "source": {
                "type": encoding,
                "media_type": f"image/{format}",
                "data": image,
            },
        }

    def _handle_text(self, text: str) -> dict[str, Any]:
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _build_agent_instruction(self, message_list: MessageList) -> str:
        """
        Build the instruction for Claude Code from the message list.

        Uses Harbor-style task instruction format for agent evaluation.
        """
        # Extract the question from the message list
        question = ""
        for msg in message_list:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Handle multi-part content
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_val = item.get("text", "")
                        if isinstance(text_val, str):
                            text_parts.append(text_val)
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = "\n".join(text_parts)

            if role == "user":
                question = content

        # Harbor-style task instruction template
        instruction = f"""Answer the following factual question accurately and concisely.

**Question:** {question}

Write your answer to `{self.solution_path}`. Your answer should be:
- A direct, factual response to the question
- As concise as possible while being complete
- Without unnecessary hedging or qualifications

**Important:**
- You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.
- Focus on accuracy - if you're uncertain, it's better to acknowledge that than to guess incorrectly.
- The answer file should contain your response as plain text."""

        return instruction

    def _run_claude_code_docker(
        self, instruction: str, output_dir: str
    ) -> tuple[str, dict[str, Any]]:
        """
        Run Claude Code CLI inside a Docker container.

        Args:
            instruction: The prompt/instruction for Claude Code
            output_dir: Host directory to mount for output

        Returns:
            Tuple of (stdout output, metadata dict)
        """
        escaped_instruction = shlex.quote(instruction)
        container_name = f"claude-code-eval-{uuid.uuid4().hex[:8]}"

        # Build environment variables
        env_args: list[str] = []

        # Pass API key
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if api_key:
            env_args.extend(["-e", f"ANTHROPIC_API_KEY={api_key}"])

        # Set model
        if self.model:
            env_args.extend(["-e", f"ANTHROPIC_MODEL={self.model}"])

        # Set max thinking tokens if specified
        if self.max_thinking_tokens is not None:
            env_args.extend(["-e", f"MAX_THINKING_TOKENS={self.max_thinking_tokens}"])

        # Disable telemetry
        env_args.extend(["-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"])

        # Build the claude command (following Harbor's approach exactly)
        # - 2>&1: combine stderr with stdout
        # - </dev/null: provide empty stdin (no interactive input)
        # - tee: log output to file
        # - After execution, copy session files to /sessions for trajectory capture
        tool_list = " ".join(self.allowed_tools)
        claude_cmd = (
            f"claude --verbose --output-format stream-json "
            f"-p {escaped_instruction} "
            f"--allowedTools {tool_list} 2>&1 </dev/null | tee /logs/agent/claude-code.txt; "
            f"cp -r /root/.claude/sessions /sessions 2>/dev/null || true"
        )

        # Build Docker run command
        # Mount /workspace to output_dir/workspace so answer.txt is accessible from host
        # Mount /sessions to capture Claude session JSONL files for trajectory
        docker_cmd = [
            "docker", "run",
            "--rm",
            "--name", container_name,
            "-v", f"{output_dir}/workspace:/workspace",
            "-v", f"{output_dir}/logs:/logs",
            "-v", f"{output_dir}/sessions:/sessions",
            "-w", "/workspace",
        ] + env_args + [
            self.docker_image,
            "bash", "-c", claude_cmd,
        ]

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            metadata: dict[str, Any] = {
                "returncode": result.returncode,
                "stderr": result.stderr,
                "container_name": container_name,
                "docker_cmd": " ".join(docker_cmd),
            }

            return result.stdout, metadata

        except subprocess.TimeoutExpired:
            # Kill the container if it's still running
            subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True,
            )
            return "", {"error": "timeout", "timeout_seconds": self.timeout}
        except Exception as e:
            return "", {"error": str(e)}

    def _run_claude_code_local(self, instruction: str) -> tuple[str, dict[str, Any]]:
        """
        Run Claude Code CLI locally (without Docker).

        Returns:
            Tuple of (stdout output, metadata dict)
        """
        escaped_instruction = shlex.quote(instruction)

        # Build environment variables
        env = os.environ.copy()

        if self.model:
            env["ANTHROPIC_MODEL"] = self.model

        if self.max_thinking_tokens is not None:
            env["MAX_THINKING_TOKENS"] = str(self.max_thinking_tokens)

        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

        # Create config directory
        with tempfile.TemporaryDirectory() as config_dir:
            env["CLAUDE_CONFIG_DIR"] = config_dir

            for subdir in ["debug", "projects/-app", "shell-snapshots", "statsig", "todos"]:
                os.makedirs(os.path.join(config_dir, subdir), exist_ok=True)

            tool_list = " ".join(self.allowed_tools)
            command = (
                f"claude --verbose --output-format stream-json "
                f"-p {escaped_instruction} --allowedTools {tool_list}"
            )

            try:
                # Create workspace directory for local execution
                workspace_dir = "/tmp/workspace"
                os.makedirs(workspace_dir, exist_ok=True)

                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                    cwd=workspace_dir,
                )

                metadata: dict[str, Any] = {
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                    "command": command,
                }

                return result.stdout, metadata

            except subprocess.TimeoutExpired:
                return "", {"error": "timeout", "timeout_seconds": self.timeout}
            except Exception as e:
                return "", {"error": str(e)}

    def _parse_solution_from_dir(self, output_dir: str) -> str:
        """
        Parse the solution from the output directory.

        Looks for the solution file in the mounted output directory.
        The solution path inside container is /workspace/answer.txt,
        which maps to output_dir/workspace/answer.txt on the host.
        """
        solution_filename = Path(self.solution_path).name  # "answer.txt"

        # Check various possible locations
        possible_paths = [
            Path(output_dir) / solution_filename,
            Path(output_dir) / "workspace" / solution_filename,
            Path(output_dir) / self.solution_path.lstrip("/"),
        ]

        for path in possible_paths:
            if path.exists():
                try:
                    return path.read_text().strip()
                except Exception:
                    continue

        return ""

    def _parse_solution_local(self) -> str:
        """Parse the solution from local filesystem."""
        solution_file = Path(self.solution_path)
        if solution_file.exists():
            try:
                return solution_file.read_text().strip()
            except Exception:
                pass
        return ""

    def _extract_answer(self, output_dir: str | None = None) -> str:
        """
        Extract the answer from the solution file.

        If the solution file is not created, return empty string (treated as failure).
        The agent must write to the solution file - no fallback parsing.
        """
        if output_dir:
            return self._parse_solution_from_dir(output_dir)
        else:
            return self._parse_solution_local()

    def _save_trajectory(self, output_dir: str, task_id: int, stdout: str) -> str:
        """
        Save the agent trajectory to the trajectory directory.

        Returns the path to the saved trajectory directory.
        """
        import shutil

        task_trajectory_dir = os.path.join(self.trajectory_dir, f"task_{task_id:04d}")
        os.makedirs(task_trajectory_dir, exist_ok=True)

        # Save the raw stdout (stream-json output)
        stdout_path = os.path.join(task_trajectory_dir, "claude-code-output.txt")
        with open(stdout_path, "w") as f:
            f.write(stdout)

        # Copy logs if they exist
        logs_src = os.path.join(output_dir, "logs")
        if os.path.exists(logs_src):
            logs_dst = os.path.join(task_trajectory_dir, "logs")
            if os.path.exists(logs_dst):
                shutil.rmtree(logs_dst)
            shutil.copytree(logs_src, logs_dst)

        # Copy workspace (contains answer.txt)
        workspace_src = os.path.join(output_dir, "workspace")
        if os.path.exists(workspace_src):
            workspace_dst = os.path.join(task_trajectory_dir, "workspace")
            if os.path.exists(workspace_dst):
                shutil.rmtree(workspace_dst)
            shutil.copytree(workspace_src, workspace_dst)

        # Copy Claude session files if they exist
        sessions_src = os.path.join(output_dir, "sessions")
        if os.path.exists(sessions_src):
            sessions_dst = os.path.join(task_trajectory_dir, "sessions")
            if os.path.exists(sessions_dst):
                shutil.rmtree(sessions_dst)
            shutil.copytree(sessions_src, sessions_dst)

        return task_trajectory_dir

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """
        Run Claude Code CLI with the given messages and return the response.
        """
        # Build instruction from message list
        instruction = self._build_agent_instruction(message_list)

        # Increment task counter for trajectory naming
        self._task_counter += 1
        task_id = self._task_counter

        trial = 0
        max_retries = 1

        while trial < max_retries:
            try:
                if self.use_docker:
                    # Create temporary directory for output
                    with tempfile.TemporaryDirectory() as output_dir:
                        # Create subdirectories
                        os.makedirs(os.path.join(output_dir, "logs", "agent"), exist_ok=True)
                        os.makedirs(os.path.join(output_dir, "workspace"), exist_ok=True)
                        os.makedirs(os.path.join(output_dir, "sessions"), exist_ok=True)

                        # Run Claude Code in Docker
                        stdout, metadata = self._run_claude_code_docker(instruction, output_dir)

                        # Save trajectory before the temp directory is deleted
                        trajectory_path = self._save_trajectory(output_dir, task_id, stdout)
                        metadata["trajectory_path"] = trajectory_path

                        if metadata.get("error"):
                            if trial < max_retries - 1:
                                wait_time = 2 ** trial
                                print(f"Claude Code error, retrying in {wait_time}s: {metadata.get('error')}")
                                time.sleep(wait_time)
                                trial += 1
                                continue
                            else:
                                return SamplerResponse(
                                    response_text=f"Error: {metadata.get('error')}",
                                    response_metadata=metadata,
                                    actual_queried_message_list=message_list,
                                )

                        # Extract response
                        response_text = self._extract_answer(output_dir)
                else:
                    # Clean up any previous solution file
                    solution_file = Path(self.solution_path)
                    if solution_file.exists():
                        solution_file.unlink()

                    # Run Claude Code locally
                    stdout, metadata = self._run_claude_code_local(instruction)

                    # Save trajectory for local execution
                    task_trajectory_dir = os.path.join(self.trajectory_dir, f"task_{task_id:04d}")
                    os.makedirs(task_trajectory_dir, exist_ok=True)
                    with open(os.path.join(task_trajectory_dir, "claude-code-output.txt"), "w") as f:
                        f.write(stdout)
                    # Copy the answer file if it exists
                    if solution_file.exists():
                        import shutil
                        workspace_dst = os.path.join(task_trajectory_dir, "workspace")
                        os.makedirs(workspace_dst, exist_ok=True)
                        shutil.copy(solution_file, os.path.join(workspace_dst, solution_file.name))
                    metadata["trajectory_path"] = task_trajectory_dir

                    if metadata.get("error"):
                        if trial < max_retries - 1:
                            wait_time = 2 ** trial
                            print(f"Claude Code error, retrying in {wait_time}s: {metadata.get('error')}")
                            time.sleep(wait_time)
                            trial += 1
                            continue
                        else:
                            return SamplerResponse(
                                response_text=f"Error: {metadata.get('error')}",
                                response_metadata=metadata,
                                actual_queried_message_list=message_list,
                            )

                    # Extract response
                    response_text = self._extract_answer()

                return SamplerResponse(
                    response_text=response_text,
                    response_metadata=metadata,
                    actual_queried_message_list=message_list,
                )

            except Exception as e:
                if trial < max_retries - 1:
                    wait_time = 2 ** trial
                    print(f"Exception in Claude Code sampler, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    trial += 1
                else:
                    return SamplerResponse(
                        response_text=f"Error: {e}",
                        response_metadata={"error": str(e)},
                        actual_queried_message_list=message_list,
                    )

        # Should not reach here, but just in case
        return SamplerResponse(
            response_text="",
            response_metadata={"error": "max retries exceeded"},
            actual_queried_message_list=message_list,
        )
