# Boundary Lab AWS Prototype

This Terraform stack creates a first-pass Fargate deployment for Boundary Lab cloud solving.

It assumes you already have:

- an AWS account/profile configured locally
- an existing VPC
- at least two public subnets for the load balancer
- at least two private subnets for Fargate tasks, with NAT or VPC endpoints for ECR/S3/CloudWatch/DynamoDB
  - for a first default-VPC prototype, you may use public subnets for the tasks and enable public IP assignment
- Docker image built and pushed to ECR

## Resources

The stack creates:

- ECR repository
- S3 job artifact bucket
- DynamoDB event table
- ECS cluster
- API task definition and service
- worker task definition
- Application Load Balancer
- CloudWatch log groups
- IAM roles and policies
- security groups

## Deploy

Copy the example vars:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit:

```text
vpc_id
public_subnet_ids
private_subnet_ids
api_assign_public_ip
worker_assign_public_ip
container_image
allowed_http_cidr_blocks
```

If you are using the AWS default VPC and only have public subnets, set
`private_subnet_ids` to the same two subnet IDs used by `public_subnet_ids`,
then set:

```hcl
api_assign_public_ip    = true
worker_assign_public_ip = true
```

This is useful for an early smoke test. For production, prefer real private
subnets and leave both public IP flags disabled.

Initialize and apply:

```bash
terraform init
terraform plan
terraform apply
```

If you want Terraform to create the ECR repository first, do an initial apply
with any placeholder image URI, push your image to the `ecr_repository_url`
output, then update `container_image` and apply again.

## Image Push

Example:

```bash
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

docker build -t boundary-lab-cloud:latest ../..
docker tag boundary-lab-cloud:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/boundary-lab-dev-cloud:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/boundary-lab-dev-cloud:latest
```

## Current Limitations

- The ALB listener is HTTP only for the prototype. Add ACM/HTTPS before public use.
- Job metadata is still in the API process memory. Events are durable in DynamoDB, but job records should move to DynamoDB next.
- API desired count should stay at `1` until job metadata becomes durable.
- Auth, billing, and quotas are not implemented.
