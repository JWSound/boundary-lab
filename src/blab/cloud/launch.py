"""Cloud worker launchers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EcsTaskLaunch:
    task_arn: str
    cluster: str
    task_definition: str
    command: list[str]
    raw_response: dict = field(default_factory=dict)


class EcsFargateTaskLauncher:
    """Submit one Boundary Lab worker task to ECS/Fargate."""

    def __init__(
        self,
        *,
        cluster: str,
        task_definition: str,
        subnets: list[str],
        security_groups: list[str] | None = None,
        container_name: str | None = None,
        assign_public_ip: str = "DISABLED",
        worker_environment: dict[str, str] | None = None,
        client=None,
    ):
        self.cluster = cluster
        self.task_definition = task_definition
        self.subnets = subnets
        self.security_groups = security_groups or []
        self.container_name = container_name
        self.assign_public_ip = assign_public_ip
        self.worker_environment = worker_environment or {}
        self._client = client

    @classmethod
    def from_env(cls):
        cluster = _required_env("BLAB_ECS_CLUSTER")
        task_definition = _required_env("BLAB_ECS_TASK_DEFINITION")
        subnets = _split_env("BLAB_ECS_SUBNETS")
        if not subnets:
            raise SystemExit("BLAB_ECS_SUBNETS is required when BLAB_JOB_LAUNCHER=ecs.")
        return cls(
            cluster=cluster,
            task_definition=task_definition,
            subnets=subnets,
            security_groups=_split_env("BLAB_ECS_SECURITY_GROUPS"),
            container_name=_required_env("BLAB_ECS_CONTAINER_NAME"),
            assign_public_ip=os.environ.get("BLAB_ECS_ASSIGN_PUBLIC_IP", "DISABLED"),
            worker_environment=_worker_environment_from_env(),
        )

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - optional AWS extra
                raise RuntimeError('Install AWS dependencies with: python -m pip install -e ".[aws]"') from exc
            self._client = boto3.client("ecs")
        return self._client

    def launch_worker(self, *, job_id: str, s3_bucket: str, s3_key: str) -> EcsTaskLaunch:
        command = [
            "cloud-worker",
            "--job-id",
            job_id,
            "--s3-bucket",
            s3_bucket,
            "--s3-key",
            s3_key,
        ]
        kwargs = {
            "cluster": self.cluster,
            "taskDefinition": self.task_definition,
            "launchType": "FARGATE",
            "networkConfiguration": {
                "awsvpcConfiguration": {
                    "subnets": self.subnets,
                    "assignPublicIp": self.assign_public_ip,
                }
            },
        }
        if self.security_groups:
            kwargs["networkConfiguration"]["awsvpcConfiguration"]["securityGroups"] = self.security_groups
        if self.container_name:
            container_override = {
                "name": self.container_name,
                "command": command,
            }
            if self.worker_environment:
                container_override["environment"] = [
                    {"name": key, "value": value}
                    for key, value in sorted(self.worker_environment.items())
                ]
            kwargs["overrides"] = {
                "containerOverrides": [container_override]
            }

        response = self.client.run_task(**kwargs)
        failures = response.get("failures", [])
        if failures:
            raise RuntimeError(f"ECS RunTask failed: {failures}")
        tasks = response.get("tasks", [])
        if not tasks:
            raise RuntimeError("ECS RunTask returned no tasks.")
        return EcsTaskLaunch(
            task_arn=str(tasks[0]["taskArn"]),
            cluster=self.cluster,
            task_definition=self.task_definition,
            command=command,
            raw_response=response,
        )


def _required_env(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        raise SystemExit(f"{key} is required.")
    return value


def _split_env(key: str) -> list[str]:
    value = os.environ.get(key, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _worker_environment_from_env() -> dict[str, str]:
    keys = (
        "BLAB_EVENT_STORE",
        "BLAB_DYNAMODB_EVENTS_TABLE",
        "BLAB_LOCAL_EVENT_ROOT",
    )
    return {
        key: value
        for key in keys
        if (value := os.environ.get(key, "").strip())
    }
