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

## Bedrock Clients 
bedrock = boto3.client("bedrock-runtime", region_name="eu-central-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Function to retrieve SSM parameters
def get_ssm_parameter(parameter_name):
    ssm = boto3.client('ssm')
    response = ssm.get_parameter(Name=parameter_name)
    return response['Parameter']['Value']

# Retrieve classification details from Parameter Store
def get_prompt_parameters():
    assessment_date_desc = get_ssm_parameter('assessment_date_desc')
    location_desc = get_ssm_parameter('location_desc')
    type_desc = get_ssm_parameter('type_desc')
    title_desc = get_ssm_parameter('title_desc')
    priority_desc = get_ssm_parameter('priority_desc')
    due_date_desc = get_ssm_parameter('due_date_desc')
    
    return assessment_date_desc, location_desc, type_desc, title_desc, priority_desc, due_date_desc

# Data ingestion for a single file
def data_ingestion(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # print ("This is the content of pypdfloader\n",documents)

    # # Testing character split works better with this pdf data set 
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=250000, chunk_overlap=100)
    # docs = text_splitter.split_documents(documents)

    # print("This is the output of text_splitter\n",docs)

     # Optional: Print for debugging to see the document size and content
    for i, doc in enumerate(documents):
        print(f"Document {i} content length: {len(doc.page_content)} characters")

    return documents

# Create vector store
def get_vector_store(docs):
    # Make sure there are no empty documents
    non_empty_docs = [doc for doc in docs if doc.page_content.strip()]
    
    if not non_empty_docs:
        raise ValueError("No non-empty documents found to embed.")
    
    vectorstore_faiss = FAISS.from_documents(non_empty_docs, bedrock_embeddings)
    vectorstore_faiss.save_local("/tmp/faiss_local")  # Use /tmp/ for Lambda temporary storage

# Create the Claude LLM model
def get_claude_llm():    
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock, model_kwargs={"stop_sequences":["\n\nHuman:"], "temperature":0.5, "top_p":0.5, "top_k":250, "max_tokens_to_sample":18000})
    return llm

# Generate response from the LLM using vector store
def get_response_llm(llm, vectorstore_faiss, query, PROMPT): 
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}            
    )
    answer = qa({"query": query})
    return answer['result']

def sanitize_filename(filename):
    # If the filename has unwanted extensions or is too long, strip it here
    filename = os.path.basename(filename)  # Ensure we get the file name and not a path
    base_name, extension = os.path.splitext(filename)  # Split filename and extension
    # Check if the extension looks valid (e.g., .pdf)
    if not extension.lower() in ['.pdf', '.txt', '.docx', '.csv']:  # Add allowed extensions
        extension = ''  # Remove invalid extensions
    # Return sanitized filename
    return f"{base_name}{extension}"


# Processing function triggered by the S3 event
def process_s3_event(event):
    # Load the document from S3
    file_obj = event["Records"][0]
    bucketname = str(file_obj["s3"]["bucket"]["name"])
    filename = unquote_plus(str(file_obj["s3"]["object"]["key"]))
    
    # Sanitize the filename
    sanitized_filename = sanitize_filename(filename)

    # Download the file from S3
    s3 = boto3.client('s3')
    download_path = f'/tmp/{sanitized_filename}'
    s3.download_file(bucketname, filename, download_path)
    
    # Process the file (single file ingestion)
    docs = data_ingestion(download_path)
    
    # Create the vector store
    get_vector_store(docs)

    # Load the FAISS index for similarity search
    faiss_index = FAISS.load_local("/tmp/faiss_local", bedrock_embeddings, allow_dangerous_deserialization=True)

    # Get prompt parameters
    assessment_date_desc, location_desc, type_desc, title_desc, priority_desc, due_date_desc = get_prompt_parameters()

    # Define the prompt with the retrieved SSM parameters
    prompt_template = f"""
    Human: 
    You are a helpful assistant. Extract the following details from the document and format the output as JSON using the keys. Skip any preamble text and generate the final answer.
    <context>
    "assessment_date" : {assessment_date_desc},
    "location" : {location_desc},
    "type" : {type_desc},
    "title" : {title_desc},
    "priority" : {priority_desc},
    "due_date" : {due_date_desc}
    </context>
    <question>
    {query}
    </question>
    Assistant: """
    
    PROMPT=PromptTemplate(
    template=prompt_template,input_variables=["context"]
    )
    # Create the LLM (Claude in this case)
    llm = get_claude_llm()

    # Define a sample query or input question
    query = """
            Using the attached document, you are an AI designed to analyze risk assessment documents and extract relevant actions, tasks, or remediations. For each action, task, or remediation in the document, please identify and output the following information in a structured and valid JSON format, using RFC 8259 specification, with the following attributes: title, location, priority, due_date, and type, assessment_date.
            
            Output each result in the following JSON format:
            {
            "title": "Action or task title",
            "location": "Location where the action is required",
            "priority": "Priority of the action or task",
            "due_date": "Due date, using YYYY-MM-DD format, or timeframe",
            "type": "Type of hazard"
            "assessment_date": "The date the site visit took place to conduct the risk assessment, format the date using YYYY-MM-DD"
            }
            Use "null" as a placeholder in the output if any information is missing or not explicitly provided. Ensure each action is extracted as a separate JSON object and provide the full list at the end. Remember to strip out special characters to ensure the JSON output matches the RFC 8259 specification. Do not add any introduction text to the JSON output. Only output the valid JSON response and nothing else. """

    # Get the response from the LLM
    response = get_response_llm(llm, faiss_index, query, PROMPT)

    # Return or store the response as needed (here you might upload it back to S3, etc.)
    print(response)
    return response, sanitized_filename

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

    # # Use regex to find content inside square brackets
    # json_data_match = re.search(r'\[(.*?)\]', result_json, re.DOTALL)
    # if json_data_match:
    #     json_data_str = json_data_match.group(1)  # Extract the content between the square brackets
    #     if not json_data_str.strip():  # Check if json_data_str is empty or contains only whitespace
    #         raise ValueError("Extracted JSON data is empty or invalid.")
        
    #     try:
    #         # Attempt to load the JSON data and handle errors
    #         json_data = json.loads(f"{json_data_str}")
    #     except json.JSONDecodeError as e:
    #         raise ValueError(f"Failed to decode JSON: {str(e)}")
    # else:
    #     raise ValueError("No data found between square brackets in the original data.")

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
