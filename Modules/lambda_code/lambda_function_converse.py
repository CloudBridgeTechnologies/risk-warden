import boto3
import time
from urllib.parse import unquote_plus

def lambda_handler(event, context):
    # from langchain_community.document_loaders import AmazonTextractPDFLoader
    # from langchain_aws import BedrockLLM
    # from langchain_community.chat_models import BedrockChat
    # from langchain.prompts import PromptTemplate
    # from langchain.chains import LLMChain
    # from langchain.chains.summarize import load_summarize_chain
    from botocore.config import Config

    # # Initialize the textract client
    # textract_client = boto3.client('textract', region_name='eu-central-1')
    
    # Load the document from S3
    file_obj = event["Records"][0]
    bucketname = str(file_obj["s3"]["bucket"]["name"])
    filename = unquote_plus(str(file_obj["s3"]["object"]["key"]))
    s3uri = f"s3://{bucketname}/{filename}"
    s3filename = filename.split('/')[-1].replace('.pdf', '')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    REGION  =   'eu-central-1'
    # Set up a custom configuration with an increased read timeout
    config = Config(
        region_name=REGION,
        read_timeout=1500  # Increase read timeout to 1500 seconds
    )
    # Initialize the Bedrock client (Claude v2)
    bedrock_client = boto3.client('bedrock-runtime', config=config) 

    ## S3 client 
    s3_client = boto3.client('s3')

    ## get s3 object 
    object = s3_client.get_object(Bucket=bucketname, Key=filename)

    ## read file 
    file_content = object['Body'].read()

    # Retrieve classification details from Parameter Store
    assessment_date_desc = get_ssm_parameter('assessment_date_desc')
    location_desc = get_ssm_parameter('location_desc')
    type_desc = get_ssm_parameter('type_desc')
    title_desc = get_ssm_parameter('title_desc')
    priority_desc = get_ssm_parameter('priority_desc')
    due_date_desc = get_ssm_parameter('due_date_desc')


    
    # Start a conversation with the user message.
    model_id = "anthropic.claude-v2"
    user_message = f"""Using the attached document, you are an AI designed to analyze risk assessment documents and extract all relevant actions, tasks, or remediations. 
    For each action, task, or remediation, please identify and output the following information in a valid JSON format: 
    {{
        "assesment_date" : {assessment_date_desc},
        "location" :{location_desc},
        "type" : {type_desc},
        "title" : {title_desc},
        "priority" : {priority_desc},
        "due_date" : {due_date_desc}
    }}
    
    title, location, priority, due_date, type, and assessment_date. Output each result in the following JSON format, Ensure all output is valid JSON. Strip any special characters that could break the JSON format."""

    conversation = [
            {
                'role': 'user',
                'content': [
                    {
                        'document': {
                            'format': 'pdf',
                            'name': 'sample',
                            'source': {
                                'bytes': file_content
                            }
                        },

                    },
                    {
                        'text':user_message
                    }
                ]
            },
        ]



    inference_config = {
        "temperature": 0.1,
        "topP":0.9,
        "stopSequences":["\n\nHuman"],
        "maxTokens": 4096
    }
    additional_model_fields = {'top_k': 250}
    additional_model_response_field_paths = ['/stop_sequence']

    # Run a Converse API
    converse_response = bedrock_client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields,
        additionalModelResponseFieldPaths=additional_model_response_field_paths
    )

    print(converse_response['output']['message']['content'])
       

#     json_filename = f"{s3filename.replace(' ', '-')}-{timestr}"
#     with open("/tmp/output.json", "w") as json_file:
#         json.dump(parsed_data, json_file, indent=4)
    
#     return upload_to_s3("/tmp/output.json", bucketname, f"bedrock/raw_output/{json_filename}_raw.json")

# # Function to upload file to S3
# def upload_to_s3(filename, bucket, key):
#     s3 = boto3.client("s3")
#     s3.upload_file(Filename=filename, Bucket=bucket, Key=key)

def get_ssm_parameter(parameter_name):
    ssm_client = boto3.client('ssm', region_name='eu-central-1')
    try:
        response = ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)
        return response['Parameter']['Value']
    except ssm_client.exceptions.ParameterNotFound:
        print(f"Parameter {parameter_name} not found.")
        return None