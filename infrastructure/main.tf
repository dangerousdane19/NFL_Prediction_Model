terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state — create this S3 bucket manually before running terraform init
  # aws s3api create-bucket --bucket nfl-pred-terraform-state --region us-east-1
  # aws s3api put-bucket-versioning --bucket nfl-pred-terraform-state \
  #   --versioning-configuration Status=Enabled
  backend "s3" {
    bucket = "nfl-pred-terraform-state"
    key    = "nfl-pred/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "nfl-pred"
      ManagedBy   = "terraform"
      Environment = var.environment
    }
  }
}

locals {
  name_prefix = "nfl-pred"
}
