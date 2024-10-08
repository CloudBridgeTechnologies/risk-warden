#!/bin/bash

# Define paths
LAMBDA_CODE_DIR="../Modules/lambda_code"
DEPS_DIR="../Modules/lambda_code/Modules"
REQUIREMENTS_FILE="requirements.txt"
ZIP_FILE="lambda_function.zip"
S3_BUCKET="rw-d1a-s3-lmb-zip"  # Replace with your S3 bucket
S3_KEY="lambda/${ZIP_FILE}"
DEPS_ZIP="lambda_dependencies.zip"

# Ensure the dependencies directory exists
mkdir -p $DEPS_DIR

# Change to the Lambda code directory
cd $LAMBDA_CODE_DIR || { echo "Lambda code directory not found: $LAMBDA_CODE_DIR"; exit 1; }

# Check if requirements.txt exists, if not skip installing dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE"
    #pip3 install -r $REQUIREMENTS_FILE -t $DEPS_DIR || { echo "Failed to install dependencies"; exit 1; }
    pip3 install -r $REQUIREMENTS_FILE -t $DEPS_DIR || { echo "Failed to install dependencies"; exit 1; }
else
    echo "No requirements.txt found, skipping dependency installation."
fi

# Zip the Lambda code
echo "Zipping the Lambda function and dependencies into $ZIP_FILE"
zip -r $ZIP_FILE . -x "$ZIP_FILE" || { echo "Failed to create zip file"; exit 1; }

# # Upload the zip file to S3
echo "Uploading $ZIP_FILE to S3://$S3_BUCKET/$S3_KEY"
aws s3 cp $ZIP_FILE s3://$S3_BUCKET/$S3_KEY || { echo "Failed to upload zip file to S3"; exit 1; }

# Clean up the zip file (but not the whole dependencies folder)
echo "Cleaning up local zip file"
rm -f $ZIP_FILE || { echo "Failed to remove local zip file"; exit 1; }

echo "Packaging complete"

# # # Zip only the dependencies
# echo "Zipping dependencies into $DEPS_ZIP"
# cd $DEPS_DIR
# zip -r $DEPS_ZIP . -x "$DEPS_ZIP" || { echo "Failed to create dependencies zip file"; exit 1; }

# # # Upload dependencies zip to S3
# echo "Uploading $DEPS_ZIP to S3://$S3_BUCKET/layers/$DEPS_ZIP"
# aws s3 cp $DEPS_ZIP s3://$S3_BUCKET/layers/$DEPS_ZIP || { echo "Failed to upload dependencies to S3"; exit 1; }