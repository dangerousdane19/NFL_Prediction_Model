resource "aws_secretsmanager_secret" "sportsdata_api_key" {
  name                    = "${local.name_prefix}/sportsdata-api-key"
  description             = "SportsData.io API key for NFL data ingestion"
  recovery_window_in_days = 7

  tags = { Name = "${local.name_prefix}-sportsdata-api-key" }
}

# The actual key value is set outside Terraform to avoid storing secrets in state.
# Run this once after terraform apply:
#
#   aws secretsmanager put-secret-value \
#     --secret-id nfl-pred/sportsdata-api-key \
#     --secret-string '{"SPORTSDATA_API_KEY":"YOUR_KEY_HERE"}'
#
# The ECS task definition references this secret by ARN with a JSON key selector,
# so the container receives SPORTSDATA_API_KEY as a plain environment variable.
