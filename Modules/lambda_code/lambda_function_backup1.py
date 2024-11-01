import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
import json
from urllib.parse import unquote_plus
import os
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

## Bedrock Clients 
bedrock = boto3.client("bedrock-runtime", region_name="eu-central-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def get_ssm_parameter(parameter_name):
    try:
        ssm = boto3.client('ssm')
        response = ssm.get_parameter(Name=parameter_name)
        return response['Parameter']['Value']
    except Exception as e:
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
        raise ValueError(f"Failed to retrieve prompt parameters: {str(e)}")

def data_ingestion(file_path):
    logger.info(f"Starting data ingestion for file: {file_path}")
    loader = PyPDFLoader(file_path)
    try:
        documents = loader.load()
        if not documents:
            logger.warning(f"No content found in the file: {file_path}")
            raise ValueError(f"No content found in the file: {file_path}")
        
        for i, doc in enumerate(documents):
            logger.info(f"Document {i} content length: {len(doc.page_content)} characters")
        
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF document: {str(e)}")
        raise ValueError(f"Error loading PDF document: {str(e)}")

def get_vector_store(docs):
    non_empty_docs = [doc for doc in docs if doc.page_content.strip()]
    
    if not non_empty_docs:
        logger.error("No non-empty documents found to embed.")
        raise ValueError("No non-empty documents found to embed.")
    
    try:
        vectorstore_faiss = FAISS.from_documents(non_empty_docs, bedrock_embeddings)
        vectorstore_faiss.save_local("/tmp/faiss_local")
        logger.info("FAISS vector store saved successfully.")
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {str(e)}")
        raise ValueError(f"Error creating FAISS vector store: {str(e)}")

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

def get_response_llm(llm, vectorstore_faiss, query, PROMPT): 
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        answer = qa({"query": query})
        logger.info("Response from LLM received successfully.")

        # Check if the response contains any meaningful content
        if not answer['result'] or "null" in answer['result']:
            logger.warning("No valid information found in the document for the query.")
            return {"message": "No relevant information found in the document for the query."}

        return answer['result']
    except Exception as e:
        logger.error(f"Error getting response from LLM: {str(e)}")
        raise ValueError(f"Error getting response from LLM: {str(e)}")

def sanitize_filename(filename):
    # If the filename has unwanted extensions or is too long, strip it here
    filename = os.path.basename(filename)  # Ensure we get the file name and not a path
    base_name, extension = os.path.splitext(filename)  # Split filename and extension
    # Check if the extension looks valid (e.g., .pdf)
    if not extension.lower() in ['.pdf', '.txt', '.docx', '.csv']:  # Add allowed extensions
        extension = ''  # Remove invalid extensions
    # Return sanitized filename
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
        s3.download_file(bucketname, filename, download_path)
        logger.info(f"File downloaded from S3: {bucketname}/{filename}")

        docs = data_ingestion(download_path)
        get_vector_store(docs)

        faiss_index = FAISS.load_local("/tmp/faiss_local", bedrock_embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully.")

        assessment_date_desc, location_desc, type_desc, title_desc, priority_desc, due_date_desc = get_prompt_parameters()

        # Consolidate the SSM parameters into a context variable
        # context = {
        #     "assessment_date": assessment_date_desc,
        #     "location": location_desc,
        #     "type": type_desc,
        #     "title": title_desc,
        #     "priority": priority_desc,
        #     "due_date": due_date_desc
        # }

        # context_str = json.dumps(context, indent=4)  # Convert context to a formatted string

        query = """
                Using the attached document, you are an AI designed to analyze risk assessment documents and extract relevant actions, tasks, or remediations. For each action, task, or remediation in the document, please identify and output the following information in a structured and valid JSON format, using RFC 8259 specification, with the following attributes: title, location, priority, due_date, and type, assessment_date.
                Use "null" as a placeholder in the output if any information is missing or not explicitly provided. Ensure each action is extracted as a separate JSON object and provide the full list at the end. Remember to strip out special characters to ensure the JSON output matches the RFC 8259 specification. Do not add any introduction text to the JSON output. Only output the valid JSON response and nothing else.
                """

        prompt_template = f"""
        Human: 
        You are a helpful assistant. Extract the following details from the document and format the output as JSON using the keys. Skip any preamble text and generate the final answer.
        <context>

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

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])

        llm = get_claude_llm()

        response = get_response_llm(llm, faiss_index, query, PROMPT)
        logger.info(f"Response: {response}")

        return response, sanitized_filename

    except Exception as e:
        logger.error(f"Error processing S3 event: {str(e)}")
        raise

def upload_to_s3(content, target_bucket, target_key):
    # Upload the result to the target S3 bucket
    s3 = boto3.client('s3')
    s3.put_object(
        Body=content,
        Bucket=target_bucket,
        Key=target_key
    )

# This function serves as the Lambda entry point
def lambda_handler(event, context):
    import re
    # Call the S3 processing function
    response, sanitized_filename = process_s3_event(event)

    # Convert the result to JSON string for upload
    result_json = json.dumps(response)

    # Define the target bucket and key where the result will be uploaded
    target_bucket = 'rw-d1a-s3-invoke-bedrock-output-dev-868380198248' 
    target_key = f"processed_results/{sanitized_filename}_result.json"

    # Upload the result to the target S3 bucket
    upload_to_s3(result_json, target_bucket, target_key)

    # Return a success message
    return {
        'statusCode': 200,
        'body': json.dumps(f"Result uploaded to s3://{target_bucket}/{target_key}")
    }
