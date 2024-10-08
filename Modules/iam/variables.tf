variable "lambda_role_name" {
  description = "The name of the IAM role for Lambda"
  type        = string
}

variable "environment" {
  description = "The environment (e.g. dev, prod)"
  type        = string
}
