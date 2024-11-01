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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import unittest

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
REGION = "eu-central-1"
TEMP_FAISS_PATH = "/tmp/faiss_local"

## Bedrock Clients 
try:
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
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

def data_ingestion(file_path):
    logger.info(f"Starting data ingestion for file: {file_path}")
    loader = PyPDFLoader(file_path)
    try:
        documents = loader.load()
        if not documents:
            logger.warning(f"No content found in the file: {file_path}")
            raise ValueError(f"No content found in the file: {file_path}")
        
        complete_text = " ".join([doc.page_content for doc in documents if doc.page_content.strip()])
        if not complete_text:
            logger.error("The document is empty.")
            raise ValueError("The document is empty.")
        
        logger.info(f"Complete document content length: {len(complete_text)} characters")
        
        # If the document is too large, summarize it to fit within the limit
        if len(complete_text) > 50000:
            logger.info("Document is too large, summarizing to fit within limit.")
            llm = get_claude_llm()
            complete_text = summarize_text(complete_text, llm)
        
        return complete_text
    except Exception as e:
        logger.error(f"Error loading PDF document: {str(e)}")
        raise ValueError(f"Error loading PDF document: {str(e)}")

def summarize_text(complete_text, llm):
    try:
        summarization_prompt = f"Summarize the following text in exact 49,999 characters and keep the original text:\n\n{complete_text}"
        # Pass the prompt as a list of strings to the llm.generate() method
        summary_result = llm.generate([summarization_prompt])
        
        # Extract the response from the LLMResult object
        if hasattr(summary_result, 'generations'):
            summary = summary_result.generations[0][0].text  # Assuming generations is a nested list with the text as an attribute
        else:
            raise ValueError("Unexpected LLMResult format. Missing 'generations' attribute.")
        
        return summary
    except Exception as e:
        logger.error(f"Error summarizing document: {str(e)}")
        raise ValueError(f"Error summarizing document: {str(e)}")

def get_vector_store_whole_document(complete_text):
    try:
        # Embed the entire document as one piece
        vectorstore_faiss = FAISS.from_texts([complete_text], bedrock_embeddings)
        vectorstore_faiss.save_local(TEMP_FAISS_PATH)
        logger.info("FAISS vector store for the whole document saved successfully.")
    except Exception as e:
        logger.error(f"Error creating FAISS vector store for the whole document: {str(e)}")
        raise ValueError(f"Error creating FAISS vector store for the whole document: {str(e)}")

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

def get_response_llm_whole_document(llm, input_variables, PROMPT, faiss_index): 
    try:
        # Create a retriever from the FAISS index
        retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Initialize the RetrievalQA chain with the retriever
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" type allows entire content to be presented
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # Update input_variables to match what the prompt expects
        input_variables_updated = {
            "context": input_variables.get("context", ""),
            "assessment_date": input_variables.get("assessment_date", ""),
            "location": input_variables.get("location", ""),
            "type": input_variables.get("type", ""),
            "title": input_variables.get("title", ""),
            "priority": input_variables.get("priority", ""),
            "due_date": input_variables.get("due_date", ""),
            "query": input_variables.get("query", "")
        }

        logger.info(f"Input to LLM: {input_variables_updated}")

        # Run the QA chain with the provided input
        answer = qa(input_variables_updated)
        logger.info("Response from LLM received successfully.")

        if not answer['result'] or "null" in answer['result']:
            logger.warning("No valid information found in the document for the query.")
            return {"message": "No relevant information found in the document for the query."}

        return answer['result']
    except Exception as e:
        logger.error(f"Error getting response from LLM: {str(e)}")
        raise ValueError(f"Error getting response from LLM: {str(e)}")

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

        complete_text = data_ingestion(download_path)
        get_vector_store_whole_document(complete_text)

        try:
            faiss_index = FAISS.load_local(TEMP_FAISS_PATH, bedrock_embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise ValueError(f"Error loading FAISS index: {str(e)}")

        # Get parameters dynamically using get_prompt_parameters()
        assessment_date_desc, location_desc, type_desc, title_desc, priority_desc, due_date_desc = get_prompt_parameters()

        query = """
                Using the attached document, you are an AI designed to analyze risk assessment documents and extract relevant actions, tasks, or remediations. For each action, task, or remediation in the document, please identify and output the following information in a structured and valid JSON format, using RFC 8259 specification, with the following attributes: title, location, priority, due_date, and type, assessment_date.
                Use "null" as a placeholder in the output if any information is missing or not explicitly provided. Ensure each action is extracted as a separate JSON object and provide the full list at the end. Remember to strip out special characters to ensure the JSON output matches the RFC 8259 specification. Do not add any introduction text to the JSON output. Only output the valid JSON response and nothing else.
                """

        prompt_template = f"""
        Human: 
        You are a helpful assistant. Extract the following details from the document and format the output as JSON using the keys. Skip any preamble text and generate the final answer.
        <context>
            {{context}}
            "assessment_date": {assessment_date_desc},
            "location": {location_desc},
            "type": {type_desc},
            "title": {title_desc},
            "priority": {priority_desc},
            "due_date": {due_date_desc}
        
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

        # Use the retrieved values for input variables dynamically
        input_variables = {
            "query": query,
            "context": complete_text,
            "assessment_date": assessment_date_desc,
            "location": location_desc,
            "type": type_desc,
            "title": title_desc,
            "priority": priority_desc,
            "due_date": due_date_desc
        }

        response = get_response_llm_whole_document(llm, input_variables, PROMPT, faiss_index)
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
    def test_data_ingestion_empty_file(self):
        with self.assertRaises(ValueError) as context:
            data_ingestion("/path/to/empty/file.pdf")
        self.assertIn("No content found in the file", str(context.exception))

    def test_summarize_text(self):
        llm = get_claude_llm()
        long_text = "This is a test document. " * 3000  # Make a long text to trigger summarization
        summary = summarize_text(long_text, llm)
        self.assertTrue(len(summary) < 50000)
        self.assertTrue(isinstance(summary, str))

    def test_sanitize_filename(self):
        sanitized = sanitize_filename("/path/to/somefile.exe")
        self.assertEqual(sanitized, "somefile")
        sanitized = sanitize_filename("/path/to/document.pdf")
        self.assertEqual(sanitized, "document.pdf")

    def test_get_ssm_parameter_failure(self):
        with self.assertRaises(ValueError) as context:
            get_ssm_parameter("non_existent_param")
        self.assertIn("Error retrieving SSM parameter", str(context.exception))

    def test_process_s3_event_failure(self):
        event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "non_existent_bucket"},
                        "object": {"key": "non_existent_file.pdf"}
                    }
                }
            ]
        }
        with self.assertRaises(ValueError) as context:
            process_s3_event(event)
        self.assertIn("Error downloading file from S3", str(context.exception))

    def test_lambda_handler_success(self):
        event = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "test_bucket"},
                        "object": {"key": "test_file.pdf"}
                    }
                }
            ]
        }
        context = None
        response = lambda_handler(event, context)
        self.assertEqual(response['statusCode'], 500)  # Assuming the file doesn't exist, should return 500

if __name__ == "__main__":
    unittest.main()
