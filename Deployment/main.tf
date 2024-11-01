module "s3_bucket" {
  source       = "../Modules/s3"
  bucket_name  = var.s3_bucket_name
  environment  = var.environment
}

module "iam_role" {
  source = "../Modules/iam"
  lambda_role_name = var.lambda_role_name
  environment   =   var.environment
}

module "lambda_function" {
  source         = "../Modules/lambda"
  function_name  = "${var.function_name}-${var.environment}"
#   s3_bucket      = module.s3_bucket.bucket_name
  role_arn       = module.iam_role.role_arn
  environment    = var.environment
  runtime        = var.runtime
  handler        = var.handler
  lambda_zip_bucket = var.lambda_zip_bucket
  memory_size       = var.memory_size
  timeout           =  var.memory_size
  s3_bucket_upload_source_arn = var.s3_bucket_upload_source_arn
  lambda_layer  = var.lambda_layer
}
