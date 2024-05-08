from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
#from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama


import configparser

configParser = configparser.RawConfigParser()   
configFilePath = r'/home/issac/project/llm/.env'
configParser.read(configFilePath)

os.environ["LANCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=configParser.get('config','lan_api')
os.environ["LANGCHAIN_PROJECT"]=configParser.get('config','LANGCHAIN_PROJECT')


app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description='Sample API'
)

# add_routes(
#     app,
#     ChatOpenAI(),
#     path="/openai"
# )

#model=ChatOpenAI()

##Olama
llm=Ollama(model="llama2-uncensored")
prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} with 100 words")

add_routes(
    app,
    prompt1|llm,
    path="/essay"
)


add_routes(
    app,
    prompt2|llm,
    path="/poem"
)


if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)



