output "lambda_function_arn" {
  description = "The ARN of the Lambda function"
  value       = module.lambda_function.lambda_arn
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket used for storage"
  value       = module.s3_bucket.bucket_name
}

output "iam_role_arn" {
  description = "ARN of the IAM Role used by Lambda"
  value       = module.iam_role.role_arn
}
