import streamlit as st
import os
import time
import configparser
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

configParser = configparser.RawConfigParser()   
configFilePath = r'/home/issac/project/llm/.env'
configParser.read(configFilePath)

os.environ["LANCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=configParser.get('config','lan_api')
os.environ["LANGCHAIN_PROJECT"]='Grok project'
groq_api=configParser.get('config','grok_api')
os.environ['GROQ_API_KEY']=groq_api




st.title("Chat groq with lama3 demo: Issac")

llm=ChatGroq(
             model_name="llama2:latest ")
prompt=ChatPromptTemplate.from_template(
    """
Anser the question based on context provided.
<context>
{context}
<context>
Question:{input}
"""
)


prompt1=st.text_input("Whats on your mind?")


def vector_embedding():
    if "vectors" not in st.session_state:
        file_dir='/home/issac/project/llm/huggingface/us_census'
        st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader(file_dir)
        st.session_state.documents=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.documents[:10])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


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
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('#'*40)