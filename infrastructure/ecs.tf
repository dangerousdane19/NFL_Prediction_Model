# ── CloudWatch Log Group ──────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${local.name_prefix}"
  retention_in_days = 30

  tags = { Name = "${local.name_prefix}-logs" }
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Name = "${local.name_prefix}-cluster" }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
  }
}

# ── Task Definition ───────────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "app" {
  family                   = "${local.name_prefix}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  # EFS volume — mounted at /app/data inside the container
  volume {
    name = "nfl-efs-data"

    efs_volume_configuration {
      file_system_id          = aws_efs_file_system.data.id
      root_directory          = "/"
      transit_encryption      = "ENABLED"
      transit_encryption_port = 2049

      authorization_config {
        access_point_id = aws_efs_access_point.data.id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = "${local.name_prefix}-container"
      image     = var.ecr_image_uri
      essential = true

      portMappings = [
        {
          containerPort = 8501
          protocol      = "tcp"
        }
      ]

      environment = [
        { name = "MODEL_DIR", value = "/app/models" },
        { name = "DB_PATH",   value = "/app/data/nfl_predictions.db" }
      ]

      # SPORTSDATA_API_KEY injected from Secrets Manager at task launch.
      # The JSON key selector (:SPORTSDATA_API_KEY::) extracts just the value
      # from the JSON object stored in the secret.
      secrets = [
        {
          name      = "SPORTSDATA_API_KEY"
          valueFrom = "${aws_secretsmanager_secret.sportsdata_api_key.arn}:SPORTSDATA_API_KEY::"
        }
      ]

      mountPoints = [
        {
          sourceVolume  = "nfl-efs-data"
          containerPath = "/app/data"
          readOnly      = false
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = { Name = "${local.name_prefix}-task" }
}

# ── ECS Service ───────────────────────────────────────────────────────────────
resource "aws_ecs_service" "app" {
  name            = "${local.name_prefix}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  # Enable ECS Exec for interactive debugging:
  #   aws ecs execute-command --cluster nfl-pred-cluster --task <arn> \
  #     --container nfl-pred-container --interactive --command "/bin/bash"
  enable_execute_command = true

  network_configuration {
    subnets          = [aws_subnet.public_a.id, aws_subnet.public_b.id]
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "${local.name_prefix}-container"
    container_port   = 8501
  }

  # Wait for ALB listener to exist before creating the service
  depends_on = [
    aws_lb_listener.http,
    aws_iam_role_policy_attachment.ecs_execution_managed,
  ]

  # Ignore task_definition changes — deployments are triggered by updating
  # var.ecr_image_uri and running terraform apply, not by in-place task def changes
  lifecycle {
    ignore_changes = [task_definition]
  }

  tags = { Name = "${local.name_prefix}-service" }
}
