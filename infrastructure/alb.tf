# ── Application Load Balancer ─────────────────────────────────────────────────
resource "aws_lb" "main" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]

  # Set high for long-lived Streamlit WebSocket connections.
  # Default 60s would terminate active user sessions mid-use.
  idle_timeout = var.alb_idle_timeout

  enable_deletion_protection = false

  tags = { Name = "${local.name_prefix}-alb" }
}

# ── Target Group ──────────────────────────────────────────────────────────────
resource "aws_lb_target_group" "app" {
  name        = "${local.name_prefix}-tg"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip" # required for Fargate awsvpc networking

  health_check {
    enabled             = true
    path                = "/_stcore/health"
    protocol            = "HTTP"
    port                = "traffic-port"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 10
    interval            = 30
    matcher             = "200"
  }

  # Sticky sessions: ensures a browser always reconnects to the same container.
  # Without this, a WebSocket reconnect may land on a different task and show
  # a blank app (Streamlit session state is in-memory per container).
  stickiness {
    type            = "lb_cookie"
    cookie_duration = 86400
    enabled         = true
  }

  tags = { Name = "${local.name_prefix}-tg" }
}

# ── Listener ──────────────────────────────────────────────────────────────────
# HTTP only for initial deployment (no domain/certificate required).
# To add HTTPS later: request an ACM certificate, add an aws_lb_listener "https"
# resource on port 443, and add an HTTP→HTTPS redirect here.
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}
