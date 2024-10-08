data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "bucket" {
  bucket = "${var.bucket_name}-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "${var.bucket_name}-${var.environment}-${data.aws_caller_identity.current.account_id}"
    Environment = var.environment
  }

  lifecycle {
    prevent_destroy = false
  }
}

