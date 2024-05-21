import os
import time
import configparser
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

configParser = configparser.RawConfigParser()   
configFilePath = r'/home/issac/project/llm/.env'
configParser.read(configFilePath)

os.environ["LANCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=configParser.get('config','lan_api')
os.environ["LANGCHAIN_PROJECT"]='Grok project'

groq_api=configParser.get('config','grok_api')
goog_api=configParser.get('config','gog')

os.environ['GROQ_API_KEY']=groq_api
os.environ['GOOGLE_API_KEY']=goog_api

st.title("Gemma chat Issac Q&A")

llm=ChatGroq(groq_api_key=groq_api,model_name="Gemma-7b-it")

prompt=ChatPromptTemplate.from_template(
    """
Answer the question based on context provided.
<context>
{context}
<context>
Question:{input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        file_dir='/home/issac/project/llm/huggingface/us_census'
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        st.session_state.loader=PyPDFDirectoryLoader(file_dir)
        st.session_state.documents=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.documents[:10])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

prompt1=st.text_input("Whats on your mind?")

if st.button("Document Embeddings"):
    vector_embedding()
    st.write("Vector store DB created")



if prompt1:
    documents_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrival_chain=create_retrieval_chain(retriever,documents_chain)
    start=time.process_time()    
    response=retrival_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Documents search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('#'*40)