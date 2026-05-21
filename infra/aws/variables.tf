variable "aws_region" {
  type        = string
  description = "AWS region for Boundary Lab cloud resources."
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Name prefix for created resources."
  default     = "boundary-lab"
}

variable "environment" {
  type        = string
  description = "Deployment environment name."
  default     = "dev"
}

variable "vpc_id" {
  type        = string
  description = "Existing VPC ID."
}

variable "public_subnet_ids" {
  type        = list(string)
  description = "Public subnets for the API load balancer."
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Subnets for API and worker Fargate tasks. Use private subnets with NAT/VPC endpoints for production; public subnets are acceptable for a first prototype when public IP assignment is enabled."
}

variable "api_assign_public_ip" {
  type        = bool
  description = "Whether the API ECS service task should receive a public IP. Enable only for public-subnet prototype deployments."
  default     = false
}

variable "allowed_http_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed to reach the prototype HTTP API."
  default     = ["0.0.0.0/0"]
}

variable "container_image" {
  type        = string
  description = "Boundary Lab cloud container image URI, usually ECR repository URL plus tag."
}

variable "api_desired_count" {
  type        = number
  description = "Desired API task count."
  default     = 1
}

variable "api_cpu" {
  type        = number
  description = "API Fargate CPU units."
  default     = 512
}

variable "api_memory" {
  type        = number
  description = "API Fargate memory MiB."
  default     = 1024
}

variable "worker_cpu" {
  type        = number
  description = "Worker Fargate CPU units."
  default     = 4096
}

variable "worker_memory" {
  type        = number
  description = "Worker Fargate memory MiB."
  default     = 8192
}

variable "worker_assign_public_ip" {
  type        = bool
  description = "Whether ECS worker tasks should receive public IPs."
  default     = false
}

variable "bundle_presign_expiration_seconds" {
  type        = number
  description = "Reserved for future API configuration; documents intended presigned upload lifetime."
  default     = 3600
}

variable "tags" {
  type        = map(string)
  description = "Additional tags for created resources."
  default     = {}
}
