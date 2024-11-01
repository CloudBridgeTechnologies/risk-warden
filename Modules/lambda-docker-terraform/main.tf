provider "aws" {
  region = "eu-central-1"  # Replace with your desired AWS region
}

# Create an ECR repository to store the Docker image
resource "aws_ecr_repository" "lambda_ecr_repo" {
  name                 = "lambda-ecr-repo"
  image_tag_mutability = "MUTABLE"
}

# IAM Role for Lambda Execution
resource "aws_iam_role" "lambda_exec_role" {
  name = var.lambda_role_name
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_policy" "lambda_policy" {
  name        = "${var.lambda_role_name}-policy"  # Environment-specific policy name
  description = "Policy for Lambda function access to S3 and Bedrock"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
        {
            "Sid": "SSMAccess",
            "Effect": "Allow",
            "Action": [
                "ssm:DescribeParameters",
                "ssm:GetParameters",
                "ssm:GetParameter"
            ],
            "Resource": "*"
        },
        {
            "Sid": "BedrockAll",
            "Effect": "Allow",
            "Action": [
                "bedrock:*"
            ],
            "Resource": "*"
        },
        {
            "Sid": "KMSDescribe",
            "Effect": "Allow",
            "Action": [
                "kms:DescribeKey"
            ],
            "Resource": "arn:*:kms:*:::*"
        },
        {
            "Sid": "APIsWithAllResourceAccess",
            "Effect": "Allow",
            "Action": [
                "iam:ListRoles",
                "ec2:DescribeVpcs",
                "ec2:DescribeSubnets",
                "ec2:DescribeSecurityGroups"
            ],
            "Resource": "*"
        },
        {
            "Sid": "PassRoleToBedrock",
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": "arn:aws:iam::*:role/*AmazonBedrock*",
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": [
                        "bedrock.amazonaws.com"
                    ]
                }
            }
        },
        {
            "Sid": "S3FullAccess",
            "Effect": "Allow",
            "Action": [
                "s3:*",
                "s3-object-lambda:*"
            ],
            "Resource": "*"
        },
        {
            "Sid": "LogsAccess",
            "Effect": "Allow",
            "Action": "logs:CreateLogGroup",
            "Resource": "arn:aws:logs:eu-central-1:868380198248:*"
        },
        {
            "Sid": "LogsStreamAccess",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": [
                "arn:aws:logs:eu-central-1:868380198248:log-group:/aws/lambda/lambda-docker-function:*"
            ]
        }
    ]
})


  tags = {
    Name        = "${var.lambda_role_name}-policy"  # Tagging IAM policy with environment
    Environment = var.environment
  }
}

# Attach the AWSLambdaBasicExecutionRole policy to the IAM role
resource "aws_iam_role_policy_attachment" "lambda_basic_exec_role_policy" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = aws_iam_policy.lambda_policy.arn
}

# Build and Push Docker Image Using Local-Exec
resource "null_resource" "build_and_push_image" {
  provisioner "local-exec" {
    command = <<EOT
      # Log in to ECR
      aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin ${aws_ecr_repository.lambda_ecr_repo.repository_url}

      # Build Docker image
      docker build -t lambda-ecr-repo .

      # Tag Docker image
      docker tag lambda-ecr-repo:latest ${aws_ecr_repository.lambda_ecr_repo.repository_url}:latest

      # Push Docker image to ECR
      docker push ${aws_ecr_repository.lambda_ecr_repo.repository_url}:latest
    EOT
  }

  depends_on = [aws_ecr_repository.lambda_ecr_repo]
}

# Lambda function definition
resource "aws_lambda_function" "lambda_function" {
  function_name = "lambda-docker-function"
  role          = aws_iam_role.lambda_exec_role.arn
  package_type  = "Image"

  # Use the image URI from ECR
  image_uri     = "${aws_ecr_repository.lambda_ecr_repo.repository_url}:latest"

  timeout       = 900  # Adjust as necessary
  memory_size   = 512  # Adjust as necessary

  # Ensure that the Docker image has been built and pushed before creating the Lambda function
  depends_on    = [null_resource.build_and_push_image]
}

resource "aws_cloudwatch_log_group" "invoke_bedrock" {
  name              = "/aws/lambda/${aws_lambda_function.lambda_function.function_name}"
  retention_in_days = 14
}

resource "aws_lambda_permission" "s3" {
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.lambda_function.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = var.s3_bucket_upload_source_arn
}