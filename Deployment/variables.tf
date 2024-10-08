variable "s3_bucket_name" {
  description = "Name of the S3 bucket to store outputs"
  type        = string
  default     = ""
}

variable "function_name" {
  description = "Name of the Lambda function"
  type        = string
  default     = ""
}

variable "lambda_role_name" {
  description = "Name of the IAM role for Lambda"
  type        = string
  default     = ""
}

variable "handler" {
  description = "Lambda handler function"
  type        = string
  default     = "lambda_function.lambda_handler"
}

variable "runtime" {
  description = "Runtime environment for Lambda"
  type        = string
  default     = "python3.9"
}

variable "backend_bucket" {
  description = "The S3 bucket name where the Terraform state file will be stored"
  type        = string
}

variable "backend_dynamodb_table" {
  description = "The DynamoDB table for Terraform state locking"
  type        = string
}

variable "environment" {
  description = "The environment name (dev, prod, etc.)"
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

variable "s3_bucket_upload_source_arn" {
  description = "arn of the s3 source bucket from which the file is uploaded"
  type        = string
}