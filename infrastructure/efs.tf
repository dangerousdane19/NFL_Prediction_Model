resource "aws_efs_file_system" "data" {
  creation_token   = "${local.name_prefix}-efs"
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"
  encrypted        = true

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = { Name = "${local.name_prefix}-efs" }
}

# Mount targets — one per private subnet for multi-AZ availability
resource "aws_efs_mount_target" "private_a" {
  file_system_id  = aws_efs_file_system.data.id
  subnet_id       = aws_subnet.private_a.id
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_mount_target" "private_b" {
  file_system_id  = aws_efs_file_system.data.id
  subnet_id       = aws_subnet.private_b.id
  security_groups = [aws_security_group.efs.id]
}

# Access point — enforces /nfl-data directory with uid/gid 1000 (appuser in container)
resource "aws_efs_access_point" "data" {
  file_system_id = aws_efs_file_system.data.id

  posix_user {
    uid = 1000
    gid = 1000
  }

  root_directory {
    path = "/nfl-data"
    creation_info {
      owner_uid   = 1000
      owner_gid   = 1000
      permissions = "755"
    }
  }

  tags = { Name = "${local.name_prefix}-efs-ap" }
}
