import requests
import streamlit as st
import os
import urllib3

urllib3.disable_warnings()

def get_eassy_response(inputtext):
    url="http://localhost:8000/essay/invoke"
    json={'input':{'topic':inputtext}}
    response=requests.post(url,json=json,verify=False)
    return response.json()['output']


def get_poem_response(inputtext):
    url="http://localhost:8000/poem/invoke"
    json={'input':{'topic':inputtext}}
    response=requests.post(url,json=json,verify=False)
    return response.json()['output']

st.title("Langchain API 2 Issac")
input_text=st.text_input("write an essay on:")
input_text1=st.text_input("write a poem on:")

if input_text:
    st.write(get_eassy_response(input_text))

if input_text1:
    st.write(get_poem_response(input_text1))
    