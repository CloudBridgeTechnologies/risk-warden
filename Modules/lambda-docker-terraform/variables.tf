variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "eu-central-1"
}

variable "lambda_memory_size" {
  description = "Memory size for Lambda function"
  type        = number
  default     = 512
}

variable "lambda_timeout" {
  description = "Timeout for Lambda function"
  type        = number
  default     = 30
}

variable "lambda_role_name"{
    type    =   string
    default =   "rw-d1a-s3-iam-invoke-bedrock-02"
}

variable "environment"{
    type    =   string
    default =   "dev"
}

variable "s3_bucket_upload_source_arn" {
  description = "arn of the s3 source bucket from which the file is uploaded"
  type        = string
  default   =   "arn:aws:s3:::rw-d2x-idp-s3-01"
}