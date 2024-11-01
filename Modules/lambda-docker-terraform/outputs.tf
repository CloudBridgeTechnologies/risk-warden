# Outputs for convenience
output "lambda_function_name" {
  value = aws_lambda_function.lambda_function.function_name
}

output "ecr_repository_url" {
  value = aws_ecr_repository.lambda_ecr_repo.repository_url
}

output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.lambda_function.arn
}
