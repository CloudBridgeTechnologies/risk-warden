resource "aws_iam_role" "lambda_role" {
  name = var.lambda_role_name            # Environment-specific Lambda role name

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action    = "sts:AssumeRole",
      Effect    = "Allow",
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = {
    Name        = var.lambda_role_name    # Tagging IAM role with environment
    Environment = var.environment
  }
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
                "arn:aws:logs:eu-central-1:868380198248:log-group:/aws/lambda/rw-d1a-lmb-invoke-bedrock-dev:*"
            ]
        }
    ]
})


  tags = {
    Name        = "${var.lambda_role_name}-policy"  # Tagging IAM policy with environment
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attachment" {
  role       = aws_iam_role.lambda_role.name   # Attaching role and policy
  policy_arn = aws_iam_policy.lambda_policy.arn
}
