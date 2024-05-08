from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama


import streamlit as st 
import os

import configparser

configParser = configparser.RawConfigParser()   
configFilePath = r'/home/issac/project/llm/.env'
configParser.read(configFilePath)

os.environ["LANCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=configParser.get('config','lan_api')
os.environ["LANGCHAIN_PROJECT"]=configParser.get('config','LANGCHAIN_PROJECT')


## Prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant, Please respond to user questions"),
        ("user","Question:{questions}")

    ]
)

##Streamlit
st.title("Langchain Issac demo")
input_text=st.text_input("Ask Questions")


###

# llm=ChatOpenAI(model="gpt-3.5-turbo")
# output_parser=StrOutputParser()
# chain=prompt|llm|output_parser

###

llm=Ollama(model="llama2-uncensored")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser




if input_text:
    st.write(chain.invoke({'questions':input_text}))

