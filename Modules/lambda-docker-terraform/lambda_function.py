import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
import json
from urllib.parse import unquote_plus
import os
import logging
from botocore.exceptions import BotoCoreError, ClientError
from botocore.config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter
import unittest
import time

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
REGION = "eu-central-1"
TEMP_FAISS_PATH = "/tmp/faiss_index"

# Set up a custom configuration with an increased read timeout
config = Config(
    region_name=REGION,
    read_timeout=800  # Increase read timeout to 800 seconds
)
## Bedrock Clients 
try:
    bedrock = boto3.client("bedrock-runtime", config=config)
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)
except (BotoCoreError, ClientError) as e:
    logger.error(f"Error initializing Bedrock client: {str(e)}")
    raise

def get_ssm_parameter(parameter_name):
    try:
        ssm = boto3.client('ssm')
        response = ssm.get_parameter(Name=parameter_name)
        return response['Parameter']['Value']
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Error retrieving SSM parameter {parameter_name}: {str(e)}")
        raise ValueError(f"Error retrieving SSM parameter {parameter_name}: {str(e)}")

def get_prompt_parameters():
    try:
        assessment_date_desc = get_ssm_parameter('assessment_date_desc')
        location_desc = get_ssm_parameter('location_desc')
        type_desc = get_ssm_parameter('type_desc')
        title_desc = get_ssm_parameter('title_desc')
        priority_desc = get_ssm_parameter('priority_desc')
        due_date_desc = get_ssm_parameter('due_date_desc')

        return assessment_date_desc, location_desc, type_desc, title_desc, priority_desc, due_date_desc
    except ValueError as e:
        logger.error(f"Failed to retrieve prompt parameters: {str(e)}")
        raise

def split_document_into_chunks(file_path):
    logger.info(f"Splitting document into chunks: {file_path}")
    loader = PyPDFLoader(file_path)
    try:
        documents = loader.load()
        if not documents:
            raise ValueError(f"No content found in the file: {file_path}")

        complete_text = " ".join([doc.page_content for doc in documents if doc.page_content.strip()])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=100)
        chunks = text_splitter.split_text(complete_text)

        logger.info(f"Document split into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting document: {str(e)}")
        raise ValueError(f"Error splitting document: {str(e)}")

def generate_and_store_embeddings(chunks):
    try:
        # Create embeddings for the chunks using Bedrock embeddings
        vectorstore = FAISS.from_texts(chunks, bedrock_embeddings,)
        
        # Save only the vectorstore data without the Bedrock client
        vectorstore_path = "/tmp/faiss_index"
        vectorstore.save_local(vectorstore_path)

        logger.info("Vector embeddings generated and stored successfully.")
    except Exception as e:
        logger.error(f"Error generating vector embeddings: {str(e)}")
        raise ValueError(f"Error generating vector embeddings: {str(e)}")


def combine_embeddings_for_bedrock(query):
    try:
        # Load the vectorstore from the local saved file
        vectorstore_path = "/tmp/faiss_index"
        vectorstore = FAISS.load_local(vectorstore_path, bedrock_embeddings,allow_dangerous_deserialization=True)
        
        # Perform a similarity search with the query to get relevant chunks
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(query)
        
        # Combine all retrieved document chunks into a single context string
        combined_context = " ".join([doc.page_content for doc in docs])
        
        logger.info("Successfully combined chunks for Bedrock.")
        return combined_context
    except Exception as e:
        logger.error(f"Error combining embeddings for Bedrock: {str(e)}")
        raise ValueError(f"Error combining embeddings for Bedrock: {str(e)}")


def get_claude_llm():    
    try:
        llm = Bedrock(
            model_id="anthropic.claude-v2", 
            client=bedrock, 
            model_kwargs={
                "stop_sequences": ["\n\nHuman:"], 
                "temperature": 0.5, 
                "top_p": 0.5, 
                "top_k": 250, 
                "max_tokens_to_sample": 18000
            }
        )
        logger.info("Claude LLM created successfully.")
        return llm
    except Exception as e:
        logger.error(f"Error creating Claude LLM: {str(e)}")
        raise ValueError(f"Error creating Claude LLM: {str(e)}")

def process_with_bedrock(llm, combined_context, query, PROMPT, retries=3, wait_seconds=10):
    for attempt in range(retries):
        try:
            formatted_input = f"Context: {combined_context}\nPrompt: {PROMPT}"
            logger.info(f"Input to Bedrock LLM: {formatted_input}")

            # Pass the formatted input to the LLM
            response = llm.generate([formatted_input])
            
            # Assuming `generations` is the correct attribute to access the generated text
            answer = response.generations[0][0].text
            logger.info("Response from Bedrock LLM received successfully.")
            return answer

        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                raise ValueError(f"Error getting response from Bedrock after {retries} attempts: {str(e)}")


def sanitize_filename(filename):
    filename = os.path.basename(filename)  # Ensure we get the file name and not a path
    base_name, extension = os.path.splitext(filename)  # Split filename and extension
    if not extension.lower() in ['.pdf', '.txt', '.docx', '.csv']:  # Add allowed extensions
        extension = ''  # Remove invalid extensions
    return f"{base_name}{extension}"

def process_s3_event(event):
    try:
        logger.info("Processing S3 event.")
        file_obj = event["Records"][0]
        bucketname = str(file_obj["s3"]["bucket"]["name"])
        filename = unquote_plus(str(file_obj["s3"]["object"]["key"]))
        
        sanitized_filename = sanitize_filename(filename)

        s3 = boto3.client('s3')
        download_path = f'/tmp/{sanitized_filename}'
        try:
            s3.download_file(bucketname, filename, download_path)
            logger.info(f"File downloaded from S3: {bucketname}/{filename}")
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Error downloading file from S3: {str(e)}")
            raise ValueError(f"Error downloading file from S3: {str(e)}")

        # Split the document into chunks
        chunks = split_document_into_chunks(download_path)
        
        # Generate vector embeddings and store them
        generate_and_store_embeddings(chunks)

        # Combine embeddings for Bedrock
        assessment_date_desc, location_desc, type_desc, title_desc, priority_desc, due_date_desc = get_prompt_parameters()
        query = """
                    Using the attached document, you are an AI designed to analyze risk assessment documents and extract relevant actions, tasks, or remediations. For each action, task, or remediation in the document, please identify and output the following information in a structured and valid JSON format, using RFC 8259 specification, with the following attributes: 
                    - title: Action or task title
                    - location: Location where the action is required
                    - priority: Priority of the action or task
                    - due_date: Due date, using YYYY-MM-DD format, or timeframe
                    - type: Type of hazard
                    - assessment_date: The date the site visit took place to conduct the risk assessment, format the date using YYYY-MM-DD

                    Use "null" as a placeholder if any information is missing or not explicitly provided. Ensure each action is extracted as a separate JSON object and provide the full list at the end. Remember to strip out special characters to ensure the JSON output matches the RFC 8259 specification. Only output the valid JSON response and nothing else.
                    """

        combined_context = combine_embeddings_for_bedrock(query)

        # Process with Bedrock
        prompt_template = f"""
        Human: 
        You are a helpful assistant. Your task is to analyze the attached risk assessment document and extract relevant actions, tasks, or remediations, providing the output in a structured JSON format that adheres to the RFC 8259 specification.
        
        <context>
            

            Instructions:
            Extract Information: Identify each action, task, or remediation from the document.
            Data Points: For each identified item, extract the following details:
            title: The name or short description of the action, task, remediation, or recommendation.
            location: The specific area where the action is required. If a "Location" field is not explicitly provided, infer it from the title page or surrounding context, including any reference numbers that help locate the area.
            priority: The priority level, which could be represented by words (e.g., Low, Medium, High) or letters/numbers (e.g., L, M, H, R, etc.).
            due_date: The deadline for completing the action in YYYY-MM-DD format. If no exact date is provided, extract the timeframe from the priority table and format it accordingly (e.g., "6 months"). Prioritize due dates over timeframes, and if a range is given (e.g., 1-3 weeks), use the lowest value.
            type: The type of hazard or risk associated with the action.
            assessment_date: The date of the site visit for the risk assessment, formatted as YYYY-MM-DD.
            Missing Data: Use "null" as a placeholder for any missing or unprovided information.
            Formatting: Ensure the JSON output is valid, stripping special characters where necessary. Each action must be represented as a separate JSON object.
            Exact Match: Extract the content exactly as it appears in the document without rewording or rephrasing. If multiple actions have the same title, list them as separate entries.
        
        </context>
        <question>
        Output each result in the following JSON format:
               
                "title": "Action or task title",
                "location": "Location where the action is required",
                "priority": "Priority of the action or task",
                "due_date": "Due date, using YYYY-MM-DD format, or timeframe",
                "type": "Type of hazard",
                "assessment_date": "The date the site visit took place to conduct the risk assessment, format the date using YYYY-MM-DD"
               
        </question>
        Assistant: """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "assessment_date", "location", "type", "title", "priority", "due_date"])

        llm = get_claude_llm()

        response = process_with_bedrock(llm, combined_context, query, PROMPT)
        logger.info(f"Response: {response}")

        return response, sanitized_filename

    except Exception as e:
        logger.error(f"Error processing S3 event: {str(e)}")
        raise

def upload_to_s3(content, target_bucket, target_key):
    s3 = boto3.client('s3')
    try:
        s3.put_object(
            Body=content,
            Bucket=target_bucket,
            Key=target_key
        )
        logger.info(f"Result uploaded to s3://{target_bucket}/{target_key}")
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Error uploading result to S3: {str(e)}")
        raise ValueError(f"Error uploading result to S3: {str(e)}")

def lambda_handler(event, context):
    try:
        response, sanitized_filename = process_s3_event(event)
        result_json = json.dumps(response)

        target_bucket = 'rw-d1a-s3-invoke-bedrock-output-dev-868380198248' 
        target_key = f"processed_results/{sanitized_filename}_result.json"

        upload_to_s3(result_json, target_bucket, target_key)

        return {
            'statusCode': 200,
            'body': json.dumps(f"Result uploaded to s3://{target_bucket}/{target_key}")
        }
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error processing request: {str(e)}")
        }

# Unit Tests
class TestLambdaFunctions(unittest.TestCase):
    def test_split_document_into_chunks(self):
        chunks = split_document_into_chunks("/path/to/sample.pdf")
        self.assertTrue(len(chunks) > 0)

    def test_generate_and_store_embeddings(self):
        chunks = ["This is a test chunk.", "Another test chunk."]
        generate_and_store_embeddings(chunks)
        self.assertTrue(os.path.exists(TEMP_FAISS_PATH))

    def test_combine_embeddings_for_bedrock(self):
        chunks = ["This is a test chunk.", "Another test chunk."]
        generate_and_store_embeddings(chunks)
        combined_context = combine_embeddings_for_bedrock("test query")
        self.assertTrue(len(combined_context) > 0)

    def test_sanitize_filename(self):
        sanitized = sanitize_filename("/path/to/somefile.exe")
        self.assertEqual(sanitized, "somefile")
        sanitized = sanitize_filename("/path/to/document.pdf")
        self.assertEqual(sanitized, "document.pdf")

    def test_get_ssm_parameter_failure(self):
        with self.assertRaises(ValueError) as context:
            get_ssm_parameter("non_existent_param")
        self.assertIn("Error retrieving SSM parameter", str(context.exception))

if __name__ == "__main__":
    unittest.main()
