# import json 
# import os
# import sys
# import boto3


# ## using titan embedding model to generate vector embedding 

# from langchain_community.embeddings import BedrockEmbeddings
# from langchain_community.llms.bedrock import Bedrock

# ## Data Ingestions 

# import numpy as np
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFDirectoryLoader

# ## Convert into vector embeddings and Vector Store

# from langchain_community.vectorstores import FAISS

# ## LLM Models 

# from langchain.prompts import PromptTemplate
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval_qa.base import RetrievalQA
# ## Bedrock Clients 

# bedrock=boto3.client("bedrock-runtime", region_name="eu-central-1")
# bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# ## Data Ingestion 
# def data_ingestion():
#     loader=PyPDFDirectoryLoader("data")
#     documents=loader.load()

#     #  testing character split works better with this pdf data set 
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    
#     docs=text_splitter.split_documents(documents)
#     return docs

# ## Vector Embeddings 

# def get_vector_store(docs):
#     vectorstore_faiss=FAISS.from_documents(docs,bedrock_embeddings)
#     vectorstore_faiss.save_local("faiss_local")

# def get_claude_llm():
#     ## create anthropic model 
#     llm=Bedrock(model_id="anthropic.claude-v2",client=bedrock,model_kwargs={"stop_sequences":["\n\nHuman:"],"temperature":0.5,"top_p":0.5,"top_k":250})
#     return llm

# # def get_lama2_llm():
# #     ## create lama2 model 
# #     llm=Bedrock(model_id="",client=bedrock,model_kwargs={'max_gen_len':512})

# #     return llm

# Prompt_template = """

# Human: 
# You are a helpful assistant. Extract the following details from the document and format the output as JSON using the keys. Skip any preamble text and generate the final answer
# <context>
# {context}
# </context>

# Questions : {question}

# Assistant : """

# PROMPT=PromptTemplate(
#     template=Prompt_template,input_variables=["context","question"]
# )

# def get_response_llm(llm, vectorstore_faiss, query): 
#     qa = RetrievalQA.from_chain_type(
#         llm=llm, chain_type="stuff", retriever=vectorstore_faiss.as_retriever(
#         search_type="similarity", search_kwargs={"k": 3}
#     ),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": PROMPT}            
#     )
#     answer=qa({"query":query})
#     return answer['result']

# # def main():
# #     st.set_page_config("Chat PDF")
    
# #     st.header("Chat with PDF using AWS Bedrock💁")

# #     user_question = st.text_input("Ask a Question from the PDF Files")

# #     with st.sidebar:
# #         st.title("Update Or Create Vector Store:")
        
# #         if st.button("Vectors Update"):
# #             with st.spinner("Processing..."):
# #                 docs = data_ingestion()
# #                 get_vector_store(docs)
# #                 st.success("Done")

# #     if st.button("Claude Output"):
# #         with st.spinner("Processing..."):
# #             faiss_index = FAISS.load_local("faiss_local", bedrock_embeddings, allow_dangerous_deserialization=True)
# #             llm=get_claude_llm()
            
# #             #faiss_index = get_vector_store(docs)
# #             st.write(get_response_llm(llm,faiss_index,user_question))
# #             st.success("Done")

#     # if st.button("Llama2 Output"):
#     #     with st.spinner("Processing..."):
#     #         faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
#     #         llm=get_llama2_llm()
            
#     #         #faiss_index = get_vector_store(docs)
#     #         st.write(get_response_llm(llm,faiss_index,user_question))
#     #         st.success("Done")

# # if __name__ == "__main__":
# #     main()


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

    # Testing character split works better with this pdf data set 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Create vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("/tmp/faiss_local")  # Use /tmp/ for Lambda temporary storage

# Create the Claude LLM model
def get_claude_llm():    
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock, model_kwargs={"stop_sequences":["\n\nHuman:"], "temperature":0.5, "top_p":0.5, "top_k":250})
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

# Processing function triggered by the S3 event
def process_s3_event(event):
    # Load the document from S3
    file_obj = event["Records"][0]
    bucketname = str(file_obj["s3"]["bucket"]["name"])
    filename = unquote_plus(str(file_obj["s3"]["object"]["key"]))
    
    # Download the file from S3
    s3 = boto3.client('s3')
    download_path = f'/tmp/{filename}'
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
    <context>{{context}}</context>

    <question>
    Using the attached document, you are an AI designed to analyze risk assessment documents and extract relevant actions, tasks, or remediations. 
    For each action, task, or remediation, please identify and output the following information in a valid JSON format: 
    "assessment_date" : {assessment_date_desc},
    "location" : {location_desc},
    "type" : {type_desc},
    "title" : {title_desc},
    "priority" : {priority_desc},
    "due_date" : {due_date_desc}

    title, location, priority, due_date, type, and assessment_date. Output each result in the following JSON format. Ensure all output is valid JSON. Strip any special characters that could break the JSON format.
    </question>
    Assistant: """
    
    PROMPT=PromptTemplate(
    template=prompt_template,input_variables=["context"]
    )
    # Create the LLM (Claude in this case)
    llm = get_claude_llm()

    # Define a sample query or input question
    query = "Please extract the relevant details from the document."

    # Get the response from the LLM
    response = get_response_llm(llm, faiss_index, query, PROMPT)

    # Return or store the response as needed (here you might upload it back to S3, etc.)
    print(response)
    return response

# This function serves as the Lambda entry point
def lambda_handler(event, context):
    # Call the S3 processing function
    result = process_s3_event(event)
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
