output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer. Open this in your browser after deploy."
  value       = aws_lb.main.dns_name
}

output "ecr_repository_url" {
  description = "ECR repository URL for pushing images"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.app.name
}

output "efs_filesystem_id" {
  description = "EFS filesystem ID — needed to seed the SQLite DB on first deploy"
  value       = aws_efs_file_system.data.id
}

output "target_group_arn" {
  description = "ALB target group ARN — use with describe-target-health to verify tasks are healthy"
  value       = aws_lb_target_group.app.arn
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for ECS container logs"
  value       = aws_cloudwatch_log_group.ecs.name
}

output "secrets_manager_arn" {
  description = "ARN of the Secrets Manager secret storing the SportsData API key"
  value       = aws_secretsmanager_secret.sportsdata_api_key.arn
}
