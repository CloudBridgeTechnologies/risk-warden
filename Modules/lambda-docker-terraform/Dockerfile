FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12 AS build_amd64

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the required Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the Lambda function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
RUN chmod 755 /var/task/lambda_function.py

# Set the command for Lambda to run when the container starts
CMD ["lambda_function.lambda_handler"]
