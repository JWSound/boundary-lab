import pytest

from blab.cloud.launch import EcsFargateTaskLauncher


class FakeEcsClient:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or {
            "tasks": [{"taskArn": "arn:aws:ecs:us-east-1:123:task/cluster/task-id"}],
            "failures": [],
        }

    def run_task(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_ecs_fargate_launcher_submits_worker_command() -> None:
    client = FakeEcsClient()
    launcher = EcsFargateTaskLauncher(
        cluster="cluster",
        task_definition="taskdef:1",
        subnets=["subnet-1", "subnet-2"],
        security_groups=["sg-1"],
        container_name="worker",
        assign_public_ip="ENABLED",
        worker_environment={"BLAB_EVENT_STORE": "dynamodb", "BLAB_DYNAMODB_EVENTS_TABLE": "events"},
        client=client,
    )

    launch = launcher.launch_worker(
        job_id="job_test",
        s3_bucket="bucket",
        s3_key="jobs/job_test/input/solve.blabsolve.zip",
    )

    assert launch.task_arn == "arn:aws:ecs:us-east-1:123:task/cluster/task-id"
    call = client.calls[0]
    assert call["cluster"] == "cluster"
    assert call["taskDefinition"] == "taskdef:1"
    assert call["launchType"] == "FARGATE"
    assert call["networkConfiguration"]["awsvpcConfiguration"] == {
        "subnets": ["subnet-1", "subnet-2"],
        "assignPublicIp": "ENABLED",
        "securityGroups": ["sg-1"],
    }
    assert call["overrides"]["containerOverrides"] == [
        {
            "name": "worker",
            "command": [
                "cloud-worker",
                "--job-id",
                "job_test",
                "--s3-bucket",
                "bucket",
                "--s3-key",
                "jobs/job_test/input/solve.blabsolve.zip",
            ],
            "environment": [
                {"name": "BLAB_DYNAMODB_EVENTS_TABLE", "value": "events"},
                {"name": "BLAB_EVENT_STORE", "value": "dynamodb"},
            ],
        }
    ]


def test_ecs_fargate_launcher_raises_on_failures() -> None:
    launcher = EcsFargateTaskLauncher(
        cluster="cluster",
        task_definition="taskdef:1",
        subnets=["subnet-1"],
        container_name="worker",
        client=FakeEcsClient(response={"tasks": [], "failures": [{"reason": "No capacity"}]}),
    )

    with pytest.raises(RuntimeError, match="ECS RunTask failed"):
        launcher.launch_worker(job_id="job_test", s3_bucket="bucket", s3_key="key")
