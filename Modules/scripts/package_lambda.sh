#!/bin/bash

# Define paths
LAMBDA_CODE_DIR="../Modules/lambda_code"
DEPS_DIR_1="../python/lib/python3.12/site-packages"
DEPS_DIR_2="../layers/layer_2"
REQUIREMENTS_FILE_1="requirements_1.txt"
REQUIREMENTS_FILE_2="requirements_2.txt"
ZIP_FILE="lambda_function.zip"
S3_BUCKET="rw-d1a-s3-lmb-zip"  # Replace with your S3 bucket
S3_KEY="lambda/${ZIP_FILE}"
DEPS_ZIP_1="lambda_dependencies_1.zip"
DEPS_ZIP_2="lambda_dependencies_2.zip"

# Ensure the dependencies directory exists
# mkdir -p $DEPS_DIR_1

# Change to the Lambda code directory
cd $LAMBDA_CODE_DIR || { echo "Lambda code directory not found: $LAMBDA_CODE_DIR"; exit 1; }

# Check if requirements_1.txt exists, if not skip installing dependencies
if [ -f "$REQUIREMENTS_FILE_1" ]; then
    echo "Installing dependencies from $REQUIREMENTS_FILE_1"
    #pip3 install -r $REQUIREMENTS_FILE -t $DEPS_DIR || { echo "Failed to install dependencies"; exit 1; }
    pip3 install -r $REQUIREMENTS_FILE_1 -t $DEPS_DIR_1 --platform manylinux2014_x86_64 --only-binary=:all:|| { echo "Failed to install dependencies"; exit 1; } || { echo "Failed to install dependencies"; exit 1; }
else
    echo "No requirements.txt found, skipping dependency installation."
fi

# # Check if requirements_2.txt exists, if not skip installing dependencies
# if [ -f "$REQUIREMENTS_FILE_2" ]; then
#     echo "Installing dependencies from $REQUIREMENTS_FILE_2"
#     #pip3 install -r $REQUIREMENTS_FILE -t $DEPS_DIR || { echo "Failed to install dependencies"; exit 1; }
#     pip3 install -r $REQUIREMENTS_FILE_2 -t $DEPS_DIR_2 --platform manylinux2014_x86_64 --only-binary=:all:|| { echo "Failed to install dependencies"; exit 1; } || { echo "Failed to install dependencies"; exit 1; }
# else
#     echo "No requirements.txt found, skipping dependency installation."
# fi

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

# # Zip only the dependencies_1
echo "Zipping dependencies into $DEPS_ZIP_1"
pwd
cd ..
zip -r $DEPS_ZIP_1 python/lib/python3.12/site-packages/ -x "$DEPS_ZIP_1" || { echo "Failed to create dependencies zip file"; exit 1; }


# # Upload dependencies_1 zip to S3
echo "Uploading $DEPS_ZIP_1 to S3://$S3_BUCKET/layers/$DEPS_ZIP_1"
aws s3 cp $DEPS_ZIP_1 s3://$S3_BUCKET/layers/$DEPS_ZIP_1 || { echo "Failed to upload dependencies to S3"; exit 1; }
# rm -f $DEPS_ZIP_1 || { echo "Failed to remove local zip file"; exit 1; }

# # # Zip only the dependencies_2
# echo "Zipping dependencies into $DEPS_ZIP_2"
# cd ../layer_2
# zip -r $DEPS_ZIP_2 . -x "$DEPS_ZIP_2" || { echo "Failed to create dependencies zip file"; exit 1; }


# # # Upload dependencies_2 zip to S3
# echo "Uploading $DEPS_ZIP_2 to S3://$S3_BUCKET/layers/$DEPS_ZIP_2"
# aws s3 cp $DEPS_ZIP_2 s3://$S3_BUCKET/layers/$DEPS_ZIP_2 || { echo "Failed to upload dependencies to S3"; exit 1; }
# # rm -f $DEPS_ZIP_2