from __future__ import annotations

import argparse
import contextlib
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, ConfigDict, Field

from sweagent import CONFIG_DIR

# ============================================================================
# Request/Response Models
# ============================================================================


class DeploymentConfigRequest(BaseModel):
    """Docker or Modal deployment configuration."""

    type: Literal["docker", "modal"] = Field(default="docker", description="Deployment type.")
    image: str = Field(default="python:3.11", description="Docker image to use.")
    python_standalone_dir: str = Field(default="/root", description="Python standalone directory.")

    model_config = ConfigDict(extra="allow")


class RepoConfigRequest(BaseModel):
    """Repository configuration."""

    github_url: str | None = Field(default=None, description="GitHub repository URL.")
    path: str | None = Field(default=None, description="Local path to the repository.")

    model_config = ConfigDict(extra="allow")


class EnvironmentConfigRequest(BaseModel):
    """Environment configuration for SWE-agent."""

    deployment: DeploymentConfigRequest = Field(
        default_factory=DeploymentConfigRequest, description="Deployment options."
    )
    repo: RepoConfigRequest | None = Field(default=None, description="Repository options.")
    post_startup_commands: list[str] = Field(default_factory=list, description="Commands to execute after startup.")
    post_startup_command_timeout: int = Field(default=500, description="Timeout for post-startup commands.")

    model_config = ConfigDict(extra="forbid")


class ModelConfigRequest(BaseModel):
    """Model configuration."""

    name: str = Field(description="Model name (e.g., 'gpt-4o', 'claude-3-opus').")
    per_instance_cost_limit: float | None = Field(default=None, description="Cost limit per instance.")
    
    temperature: float | None = Field(default=None, description="Sampling temperature (e.g., 1.0 for GPT-5).")

    model_config = ConfigDict(extra="allow")


class ToolConfigRequest(BaseModel):
    """Tool configuration."""

    bundles: list[str] = Field(default_factory=list, description="Tool bundle paths or names.")

    model_config = ConfigDict(extra="allow")


class TemplateConfigRequest(BaseModel):
    """Template configuration."""

    system_template: str = Field(default="", description="System message template.")
    instance_template: str = Field(default="", description="Instance message template.")

    model_config = ConfigDict(extra="allow")


class HistoryProcessorConfigRequest(BaseModel):
    """History processor configuration."""

    type: str = Field(description="History processor type (e.g., 'default', 'last_n_observations', 'cache_control').")

    model_config = ConfigDict(extra="allow")


class AgentConfigRequest(BaseModel):
    """Agent configuration."""

    model: ModelConfigRequest = Field(description="Model options.")
    tools: ToolConfigRequest | None = Field(default=None, description="Tool options.")
    templates: TemplateConfigRequest | None = Field(default=None, description="Template options.")
    history_processors: list[HistoryProcessorConfigRequest] | None = Field(
        default=None, description="History processors configuration."
    )
    max_requeries: int = Field(default=3, description="Maximum number of retries after errors.")
    type: Literal["default", "retry", "shell"] = Field(default="default", description="Agent type.")

    model_config = ConfigDict(extra="allow")


class ProblemStatementRequest(BaseModel):
    """Problem statement configuration.

    Provide one of: text, path, or github_url.
    """

    type: Literal["text", "text_file", "github", "empty"] = Field(
        default="empty", description="Problem statement type."
    )
    text: str | None = Field(default=None, description="Direct text problem statement.")
    path: str | None = Field(default=None, description="Path to problem statement file.")
    github_url: str | None = Field(default=None, description="GitHub issue URL.")
    id: str | None = Field(default=None, description="Problem statement ID.")
    extra_fields: dict[str, Any] = Field(default_factory=dict, description="Additional fields for templates.")

    model_config = ConfigDict(extra="forbid")


class ActionConfigRequest(BaseModel):
    """Actions to perform after solving the issue."""

    open_pr: bool = Field(default=False, description="Open a PR with the patch.")
    apply_patch_locally: bool = Field(default=False, description="Apply patch to local repository.")

    model_config = ConfigDict(extra="forbid")


class RunSingleRequest(BaseModel):
    """Request body for running SWE-agent on a single instance."""

    env: EnvironmentConfigRequest = Field(
        default_factory=EnvironmentConfigRequest, description="Environment configuration."
    )
    agent: AgentConfigRequest = Field(description="Agent configuration.")
    problem_statement: ProblemStatementRequest = Field(
        default_factory=ProblemStatementRequest, description="Problem statement configuration."
    )
    output_dir: str = Field(default="DEFAULT", description="Output directory for results.")
    actions: ActionConfigRequest = Field(default_factory=ActionConfigRequest, description="Post-run actions.")
    env_var_path: str | None = Field(default=None, description="Path to a .env file.")
    env_vars: dict[str, str] | None = Field(
        default=None,
        description="Environment variable overrides applied during execution.",
    )
    config_file: str | None = Field(
        default=None,
        description="Path to a YAML config file to use as base configuration. "
        "If not specified, 'config/default.yaml' is loaded automatically. "
        "Set to empty string to disable default config loading.",
    )

    model_config = ConfigDict(extra="forbid")


class RunSingleResponse(BaseModel):
    """Response from running SWE-agent on a single instance."""

    success: bool = Field(description="Whether the run completed successfully.")
    instance_id: str = Field(description="ID of the problem instance.")
    output_dir: str = Field(description="Directory where results are saved.")
    patch: str | None = Field(default=None, description="Generated patch if available.")
    exit_status: str | None = Field(default=None, description="Exit status of the agent.")
    error: str | None = Field(default=None, description="Error message if run failed.")

    model_config = ConfigDict(extra="forbid")


class JobSubmissionResponse(BaseModel):
    """Response for job submission."""

    job_id: str = Field(description="Unique identifier for the submitted job.")
    status: str = Field(description="Initial job status (e.g., 'pending').")
    message: str = Field(description="Confirmation message.")

    model_config = ConfigDict(extra="forbid")


class JobStatusResponse(BaseModel):
    """Response for job status query."""

    job_id: str = Field(description="Unique identifier for the job.")
    status: str = Field(description="Current job status: 'pending', 'running', 'completed', 'error'.")
    created_at: str = Field(description="Timestamp when job was created.")
    completed_at: str | None = Field(default=None, description="Timestamp when job completed.")
    result: RunSingleResponse | None = Field(default=None, description="Job result if completed.")

    model_config = ConfigDict(extra="forbid")


# ============================================================================
# Helper Functions
# ============================================================================


@contextlib.contextmanager
def _temporary_environ(overrides: dict[str, str] | None):
    """Temporarily set environment variables, then restore original values."""
    if not overrides:
        yield
        return

    previous: dict[str, str | None] = {k: os.environ.get(k) for k in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries. Values in override take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_default_config(config_file: str | None) -> dict[str, Any]:
    """Load base configuration from YAML file.

    Args:
        config_file: Path to config file. If None, loads config/default.yaml.
                     If empty string, returns empty dict (no default config).
    """
    if config_file == "":
        # Explicitly disabled
        return {}

    if config_file is not None:
        config_path = Path(config_file)
    else:
        config_path = CONFIG_DIR / "default.yaml"

    if not config_path.exists():
        return {}

    txt = config_path.read_text()
    if not txt.strip():
        return {}

    return yaml.safe_load(txt) or {}


def _build_run_single_config(req: RunSingleRequest):
    """Convert API request to RunSingleConfig.

    This function loads the base configuration from default.yaml (or specified config_file)
    and deep merges the API request values on top of it.
    """
    from sweagent.agent.problem_statement import (
        EmptyProblemStatement,
        FileProblemStatement,
        GithubIssue,
        TextProblemStatement,
    )
    from sweagent.run.run_single import RunSingleActionConfig, RunSingleConfig

    # Load base configuration from YAML file
    base_config = _load_default_config(req.config_file)
    base_agent_config = base_config.get("agent", {})

    # Build problem statement config
    ps_req = req.problem_statement
    if ps_req.type == "text" and ps_req.text:
        kwargs = {"text": ps_req.text, "extra_fields": ps_req.extra_fields}
        if ps_req.id is not None:
            kwargs["id"] = ps_req.id
        problem_statement = TextProblemStatement(**kwargs)  # type: ignore[arg-type]
    elif ps_req.type == "text_file" and ps_req.path:
        kwargs = {"path": Path(ps_req.path), "extra_fields": ps_req.extra_fields}
        if ps_req.id is not None:
            kwargs["id"] = ps_req.id
        problem_statement = FileProblemStatement(**kwargs)  # type: ignore[arg-type]
    elif ps_req.type == "github" and ps_req.github_url:
        kwargs = {"github_url": ps_req.github_url, "extra_fields": ps_req.extra_fields}
        if ps_req.id is not None:
            kwargs["id"] = ps_req.id
        problem_statement = GithubIssue(**kwargs)  # type: ignore[arg-type]
    else:
        problem_statement = EmptyProblemStatement(id=ps_req.id) if ps_req.id else EmptyProblemStatement()

    # Build environment config dict
    env_dict: dict[str, Any] = {
        "deployment": req.env.deployment.model_dump(exclude_none=True),
        "post_startup_commands": req.env.post_startup_commands,
        "post_startup_command_timeout": req.env.post_startup_command_timeout,
    }
    if req.env.repo:
        repo_dict = req.env.repo.model_dump(exclude_none=True)
        if repo_dict:
            env_dict["repo"] = repo_dict

    # Build agent config dict from API request
    request_agent_dict: dict[str, Any] = {
        "model": req.agent.model.model_dump(exclude_none=True),
        "max_requeries": req.agent.max_requeries,
        "type": req.agent.type,
    }
    if req.agent.tools:
        request_agent_dict["tools"] = req.agent.tools.model_dump(exclude_none=True)
    if req.agent.templates:
        request_agent_dict["templates"] = req.agent.templates.model_dump(exclude_none=True)
    if req.agent.history_processors:
        request_agent_dict["history_processors"] = [
            hp.model_dump(exclude_none=True) for hp in req.agent.history_processors
        ]

    # Deep merge: base config from YAML + API request overrides
    agent_dict = _deep_merge(base_agent_config, request_agent_dict)

    # Build actions config
    actions = RunSingleActionConfig(
        open_pr=req.actions.open_pr,
        apply_patch_locally=req.actions.apply_patch_locally,
    )

    # Create and return the RunSingleConfig
    return RunSingleConfig(
        env=env_dict,  # type: ignore[arg-type]
        agent=agent_dict,  # type: ignore[arg-type]
        problem_statement=problem_statement,
        output_dir=Path(req.output_dir),
        actions=actions,
        env_var_path=Path(req.env_var_path) if req.env_var_path else None,
    )


def _run_single(req: RunSingleRequest) -> RunSingleResponse:
    """Execute RunSingle with the given configuration."""
    from sweagent.run.run_single import RunSingle

    try:
        with _temporary_environ(req.env_vars):
            config = _build_run_single_config(req)
            runner = RunSingle.from_config(config)
            runner.run()

            # Extract results
            instance_id = runner.problem_statement.id
            output_dir = str(runner.output_dir)

            # Try to read the patch if it exists
            patch = None
            patch_file = runner.output_dir / instance_id / "patch.diff"
            if patch_file.exists():
                patch = patch_file.read_text()

            return RunSingleResponse(
                success=True,
                instance_id=instance_id,
                output_dir=output_dir,
                patch=patch,
                exit_status="completed",
            )

    except Exception as e:
        # Return error response instead of raising
        return RunSingleResponse(
            success=False,
            instance_id=req.problem_statement.id or "unknown",
            output_dir=req.output_dir,
            error=f"{type(e).__name__}: {e}",
            exit_status="error",
        )




# ============================================================================
# Job Tracking
# ============================================================================

# In-memory job storage (use Redis/database for production)
jobs: dict[str, dict[str, Any]] = {}


def _run_single_background(job_id: str, req: RunSingleRequest) -> None:
    """Background task to run SWE-agent job."""
    jobs[job_id]["status"] = "running"
    
    try:
        result = _run_single(req)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["result"] = RunSingleResponse(
            success=False,
            instance_id="unknown",
            output_dir="",
            error=f"{type(e).__name__}: {e}",
            exit_status="error",
        )


# ============================================================================
# FastAPI Application
# ============================================================================


def create_app() -> FastAPI:
    app = FastAPI(
        title="SWE-agent API",
        version="0.1",
        description="REST API for running SWE-agent on single problem instances.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.post("/run", response_model=JobSubmissionResponse)
    async def run_single(req: RunSingleRequest, background_tasks: BackgroundTasks) -> JobSubmissionResponse:
        """Submit a SWE-agent job to run in the background.

        This endpoint accepts a structured configuration and queues a SWE-agent
        job to solve the specified problem. Returns immediately with a job_id
        that can be used to check the status and retrieve results.

        Example request body:
        ```json
        {
            "agent": {
                "model": {"name": "gpt-4o"}
            },
            "problem_statement": {
                "type": "github",
                "github_url": "https://github.com/owner/repo/issues/1"
            },
            "env": {
                "repo": {"github_url": "https://github.com/owner/repo"}
            }
        }
        ```
        """
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "result": None,
        }
        
        background_tasks.add_task(_run_single_background, job_id, req)
        
        return JobSubmissionResponse(
            job_id=job_id,
            status="pending",
            message="Job submitted successfully. Use /job/{job_id} to check status.",
        )

    @app.get("/job/{job_id}", response_model=JobStatusResponse)
    def get_job_status(job_id: str) -> JobStatusResponse:
        """Get the status and result of a submitted job.

        Args:
            job_id: The unique identifier returned when the job was submitted.

        Returns:
            Job status and results if completed.
        """
        if job_id not in jobs:
            return JobStatusResponse(
                job_id=job_id,
                status="not_found",
                created_at="",
                completed_at=None,
                result=None,
            )
        
        job = jobs[job_id]
        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
            result=job.get("result"),
        )

    return app


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    log_level: str


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SWE-agent FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug", "trace"])
    return parser


def run_from_cli(args: list[str] | None = None) -> None:
    # Deferred import so `sweagent` CLI stays fast unless API is used.
    import uvicorn

    parser = _build_arg_parser()
    parsed = parser.parse_args(args)
    config = ApiConfig(host=parsed.host, port=parsed.port, log_level=parsed.log_level)

    uvicorn.run(
        "sweagent.api.server:create_app",
        factory=True,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )
