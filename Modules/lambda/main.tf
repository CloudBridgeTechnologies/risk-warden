resource "aws_lambda_function" "lambda_function" {
  function_name = var.function_name      # Dynamic based on environment
  handler       = var.handler            # Dynamic handler
  runtime       = var.runtime            # Lambda runtime (Python 3.9)
  role          = var.role_arn           # IAM role ARN for Lambda execution
  memory_size = var.memory_size
  timeout   =   var.timeout
  architectures  = ["x86_64"]
  
  environment {
        variables = {
        ENV = var.environment    # Passing environment variable to Lambda
        }
    }


  s3_bucket = var.lambda_zip_bucket            # S3 bucket where Lambda zip is stored
  s3_key    = "lambda/lambda_function.zip"       # Path to the Lambda package in S3 (can be environment-specific)

  layers = [
    var.lambda_layer  # Attach the layer here
  ]

  tags = {
    Name        = var.function_name      # Tagging Lambda function with environment
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "invoke_bedrock" {
  name              = "/aws/lambda/${aws_lambda_function.lambda_function.function_name}"
  retention_in_days = 14
}

# Run the packaging script with local-exec
resource "null_resource" "package_lambda" {
  provisioner "local-exec" {
    command = "bash ../Modules/scripts/package_lambda.sh"
  }

  triggers = {
    always_run = "${timestamp()}"
  }
}

# # Upload the ZIP to S3
# resource "aws_s3_object" "lambda_zip" {
#   bucket = var.lambda_zip_bucket
#   key    = "layers/lambda_dependencies.zip"
#   source = "../lambda_code/layers/lambda_dependencies.zip"
#   depends_on = [null_resource.package_lambda]
# }

resource "aws_lambda_layer_version" "lambda_layer_1" {
  layer_name          = "lambda_dependencies_layer_1"
  s3_bucket           = var.lambda_zip_bucket
  s3_key              = "layers/lambda_dependencies_1.zip"
  # filename            = "../Modules/layers/lambda_dependencies.zip"
  compatible_runtimes = ["python3.12"]
  compatible_architectures  = ["x86_64"]
  depends_on = [null_resource.package_lambda]
}

# resource "aws_lambda_layer_version" "lambda_layer_2" {
#   layer_name          = "lambda_dependencies_layer_2"
#   s3_bucket           = var.lambda_zip_bucket
#   s3_key              = "layers/lambda_dependencies_2.zip"
#   # filename            = "../Modules/layers/lambda_dependencies.zip"
#   compatible_runtimes = ["python3.12"]
#   compatible_architectures  = ["x86_64"]
#   depends_on = [null_resource.package_lambda]
# }

resource "aws_lambda_permission" "s3" {
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lambda_function.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = var.s3_bucket_upload_source_arn
}

