import boto3
import time
from urllib.parse import unquote_plus
from io import BytesIO
import json
from pypdf import PdfReader
from botocore.config import Config

def lambda_handler(event, context):
    REGION = 'eu-central-1'
    MAX_TEXT_LENGTH = 15000

    # Set up a custom configuration with an increased read timeout
    config = Config(
        region_name=REGION,
        read_timeout=1500
    )

    # Initialize the Bedrock client (Claude v2)
    bedrock_client = boto3.client('bedrock-runtime', config=config)

    # S3 client 
    s3_client = boto3.client('s3')

    # Load the document from S3
    file_obj = event["Records"][0]
    bucketname = str(file_obj["s3"]["bucket"]["name"])
    filename = unquote_plus(str(file_obj["s3"]["object"]["key"]))

    # Get S3 object 
    s3_object = s3_client.get_object(Bucket=bucketname, Key=filename)
    file_content = s3_object['Body'].read()

    # Extract text from PDF using pypdf
    pdf_stream = BytesIO(file_content)
    reader = PdfReader(pdf_stream)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Chunk the text if it exceeds the maximum limit
    text_chunks = [text[i:i + MAX_TEXT_LENGTH] for i in range(0, len(text), MAX_TEXT_LENGTH)]

    # Retrieve classification details from Parameter Store
    assessment_date_desc = get_ssm_parameter('assessment_date_desc')
    location_desc = get_ssm_parameter('location_desc')
    type_desc = get_ssm_parameter('type_desc')
    title_desc = get_ssm_parameter('title_desc')
    priority_desc = get_ssm_parameter('priority_desc')
    due_date_desc = get_ssm_parameter('due_date_desc')

    # Prepare the initial user message
    user_message = f"""You are an AI designed to analyze risk assessment documents and extract all relevant actions, tasks, or remediations. 
For each action, task, or remediation, please identify and output the following information in a valid JSON format: 
{{
    "assessment_date": "{assessment_date_desc}",
    "location": "{location_desc}",
    "type": "{type_desc}",
    "title": "{title_desc}",
    "priority": "{priority_desc}",
    "due_date": "{due_date_desc}"
}}
Ensure all output is valid JSON. Strip any special characters that could break the JSON format."""

    # Initialize result storage
    all_results = []

    # Process each chunk using Converse API
    for i, text_chunk in enumerate(text_chunks):
        # Construct the conversation list with proper dictionary structure
        conversation = [
            {
                'role': 'user',
                'content': [
                    {
                        "text": user_message
                    },
                    {
                        "text": f"Document Text (Part {i+1}): {text_chunk}"
                    }
                ]
            }
        ]

        # Configuration for the inference process
        inference_config = {
            "maxTokens": 4096,
            "stopSequences": ["\n\nHuman"],
            "temperature": 0.5,
            "topP": 0.5
        }

        # Run the Converse API for the first time
        is_truncated = True
        while is_truncated:
            # Run Converse API
            converse_response = bedrock_client.converse(
                modelId="anthropic.claude-v2",
                messages=conversation,
                inferenceConfig=inference_config,
                additionalModelRequestFields={},
                additionalModelResponseFieldPaths=['/stop_sequence']
            )

            # Get output and handle as a list
            output_content = converse_response['output']['message']['content']

            if isinstance(output_content, list):
                # If output_content is a list, concatenate all elements into a single string
                output_content = ''.join([str(item) for item in output_content])

            all_results.append(output_content)

            # Check if the output appears to be truncated
            if output_content.strip().endswith("}"):
                is_truncated = False  # Complete JSON output indicates it may not be truncated
            else:
                # Add a follow-up message asking the AI to continue
                follow_up_message = "Please continue from where the output ended."
                conversation[0]['content'].append({"text": follow_up_message})

    # Combine all results
    combined_results = "\n".join(all_results)

    # Save the combined results to S3
    s3filename = filename.split('/')[-1].replace('.pdf', '')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    json_filename = f"{s3filename.replace(' ', '-')}-{timestr}"
    with open("/tmp/output.json", "w") as json_file:
        json.dump(combined_results, json_file, indent=4)
    
    upload_to_s3("/tmp/output.json", bucketname, f"bedrock/raw_output/{json_filename}_raw.json")

def upload_to_s3(filename, bucket, key):
    s3 = boto3.client("s3")
    s3.upload_file(Filename=filename, Bucket=bucket, Key=key)

def get_ssm_parameter(parameter_name):
    ssm_client = boto3.client('ssm', region_name='eu-central-1')
    try:
        response = ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)
        return response['Parameter']['Value']
    except ssm_client.exceptions.ParameterNotFound:
        print(f"Parameter {parameter_name} not found.")
        return None
