variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment label (e.g. prod, staging)"
  type        = string
  default     = "prod"
}

variable "ecr_image_uri" {
  description = "Full ECR image URI including tag, e.g. 123456789012.dkr.ecr.us-east-1.amazonaws.com/nfl-pred-app:abc1234"
  type        = string
}

variable "desired_count" {
  description = "Number of ECS tasks to run. Keep at 1 while using SQLite on EFS."
  type        = number
  default     = 1
}

variable "task_cpu" {
  description = "ECS task CPU units (1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "ECS task memory in MB"
  type        = number
  default     = 1024
}

variable "alb_idle_timeout" {
  description = "ALB idle timeout in seconds. Set high for long-lived Streamlit WebSocket connections."
  type        = number
  default     = 300
}
