function_name    = "my-lambda-function-prod"
handler          = "handler.lambda_handler"
runtime          = "python3.9"
role_arn         = "arn:aws:iam::123456789012:role/my-lambda-role-prod"
s3_bucket        = "my-prod-lambda-bucket"
environment      = "prod"
lambda_role_name = "my-lambda-role-prod"