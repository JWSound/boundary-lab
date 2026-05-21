output "ecr_repository_url" {
  description = "ECR repository URL for Boundary Lab cloud image pushes."
  value       = aws_ecr_repository.cloud.repository_url
}

output "api_url" {
  description = "Prototype HTTP API URL."
  value       = "http://${aws_lb.api.dns_name}"
}

output "jobs_bucket" {
  description = "S3 bucket used for uploaded solve bundles and future artifacts."
  value       = aws_s3_bucket.jobs.bucket
}

output "events_table" {
  description = "DynamoDB table used for worker event rows."
  value       = aws_dynamodb_table.events.name
}

output "ecs_cluster" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.main.name
}

output "api_task_definition" {
  description = "API ECS task definition ARN."
  value       = aws_ecs_task_definition.api.arn
}

output "worker_task_definition" {
  description = "Worker ECS task definition ARN."
  value       = aws_ecs_task_definition.worker.arn
}

output "api_security_group_id" {
  description = "API task security group ID."
  value       = aws_security_group.api.id
}

output "worker_security_group_id" {
  description = "Worker task security group ID."
  value       = aws_security_group.worker.id
}
