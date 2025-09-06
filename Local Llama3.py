from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 
import streamlit as st 

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please response to the user questions"),
    ("user", "Question:{question}")
])

llm = Ollama(model = "llama3")

output_parser = StrOutputParser()

chain = prompt|llm|output_parser

st.title("Langchain Demo With Llama 3 API")
input_text = st.text_input("Search the topic you want")
if input_text:
    st.write(chain.invoke({'question': input_text}))