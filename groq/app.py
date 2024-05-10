import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import configparser
import time

configParser = configparser.RawConfigParser()   
configFilePath = r'/home/issac/project/llm/.env'
configParser.read(configFilePath)

os.environ["LANCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=configParser.get('config','lan_api')
os.environ["LANGCHAIN_PROJECT"]='Grok project'
groq_api=configParser.get('config','grok_api')

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
st.title("Chat groq for Issac")
llm=ChatGroq(groq_api=groq_api,
             model="Gemma-7b-It")

prompt=ChatPromptTemplate.from_template(
    """ Answer the questions based on the provided context only.
    Provide most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)


document_chain=create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.vectors.as_retreiver()
retrieval_chain = create_retrieval_chain(retriever,document_chain)

prompt=st.text_input("Input here:")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response:",time.process_time()-start)
    st.write(response['answer'])



