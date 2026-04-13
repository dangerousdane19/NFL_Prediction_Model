# Security groups are defined without cross-references first,
# then rules referencing other groups are added separately to break the cycle.

# ── ALB Security Group ────────────────────────────────────────────────────────
resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-sg-alb"
  description = "Internet-facing ALB"
  vpc_id      = aws_vpc.main.id
  tags        = { Name = "${local.name_prefix}-sg-alb" }
}

resource "aws_security_group_rule" "alb_ingress_https" {
  security_group_id = aws_security_group.alb.id
  type              = "ingress"
  description       = "HTTPS from internet"
  from_port         = 443
  to_port           = 443
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "aws_security_group_rule" "alb_ingress_http" {
  security_group_id = aws_security_group.alb.id
  type              = "ingress"
  description       = "HTTP from internet"
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "aws_security_group_rule" "alb_egress_ecs" {
  security_group_id        = aws_security_group.alb.id
  type                     = "egress"
  description              = "Forward to Streamlit containers"
  from_port                = 8501
  to_port                  = 8501
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.ecs.id
}

# ── ECS Security Group ────────────────────────────────────────────────────────
resource "aws_security_group" "ecs" {
  name        = "${local.name_prefix}-sg-ecs"
  description = "ECS Fargate tasks"
  vpc_id      = aws_vpc.main.id
  tags        = { Name = "${local.name_prefix}-sg-ecs" }
}

resource "aws_security_group_rule" "ecs_ingress_alb" {
  security_group_id        = aws_security_group.ecs.id
  type                     = "ingress"
  description              = "Streamlit from ALB only"
  from_port                = 8501
  to_port                  = 8501
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.alb.id
}

resource "aws_security_group_rule" "ecs_egress_https" {
  security_group_id = aws_security_group.ecs.id
  type              = "egress"
  description       = "HTTPS outbound: ECR, Secrets Manager, CloudWatch, APIs"
  from_port         = 443
  to_port           = 443
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "aws_security_group_rule" "ecs_egress_http" {
  security_group_id = aws_security_group.ecs.id
  type              = "egress"
  description       = "HTTP outbound: nflpenalties.com scraper"
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]
}

resource "aws_security_group_rule" "ecs_egress_efs" {
  security_group_id        = aws_security_group.ecs.id
  type                     = "egress"
  description              = "NFS to EFS mount targets"
  from_port                = 2049
  to_port                  = 2049
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.efs.id
}

# ── EFS Security Group ────────────────────────────────────────────────────────
resource "aws_security_group" "efs" {
  name        = "${local.name_prefix}-sg-efs"
  description = "EFS mount targets"
  vpc_id      = aws_vpc.main.id
  tags        = { Name = "${local.name_prefix}-sg-efs" }
}

resource "aws_security_group_rule" "efs_ingress_ecs" {
  security_group_id        = aws_security_group.efs.id
  type                     = "ingress"
  description              = "NFS from ECS tasks"
  from_port                = 2049
  to_port                  = 2049
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.ecs.id
}

resource "aws_security_group_rule" "efs_egress_ecs" {
  security_group_id        = aws_security_group.efs.id
  type                     = "egress"
  description              = "NFS responses to ECS tasks"
  from_port                = 2049
  to_port                  = 2049
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.ecs.id
}
