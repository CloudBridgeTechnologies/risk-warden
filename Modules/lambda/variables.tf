variable "function_name" {
  description = "The name of the Lambda function"
  type        = string
}

variable "handler" {
  description = "The Lambda function handler"
  type        = string
}

variable "runtime" {
  description = "The Lambda runtime"
  type        = string
}

variable "role_arn" {
  description = "The IAM Role ARN for Lambda"
  type        = string
}

variable "s3_bucket_upload_source_arn" {
  description = "arn of the s3 source bucket from which the file is uploaded"
  type        = string
}


variable "environment" {
  description = "Environment variables for the Lambda function"
  type        = string
}

variable "lambda_zip_bucket"{
  description = "Bucket used to hold the lambda zip"
  type  =   string
}

variable "timeout"{
    description =   "Enter in what time the lambda will timeout"
    type    =   string
}

variable "memory_size"{
    description = "Enter the value in MB needed to execute the lambda"
    type = string
}