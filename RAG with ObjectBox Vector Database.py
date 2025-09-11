import streamlit as st 
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate 
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import os 
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    groq_api_key = groq_api_key,
    temperature = 0
)


prompt = ChatPromptTemplate.from_template(
    """
    فقط بر اساس متن ارائه شده، به سوال پاسخ بده.
    متن: {context}
    سوال: {input}
    """
)


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name = "myrkur/sentence-transformer-parsbert-fa-2.0")

        st.session_state.loader = PyPDFDirectoryLoader("C:/Users/Lenovo/Desktop/datapdf")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.splitter = RecursiveCharacterTextSplitter(
            separators = ["\n\n", "\n", " "],
            chunk_size = 1500,
            chunk_overlap = 150
        )
        st.session_state.chunks = st.session_state.splitter.split_documents(st.session_state.docs)

        st.session_state.vectors = ObjectBox.from_documents(st.session_state.chunks, st.session_state.embeddings, embedding_dimensions = 768)


st.header("ObjectBox VectorstoreDB with Llama 3 Demo")

input_prompt = st.text_input("Enter your questions from Documents")

if st.button("Document Embedding"):
    if "vectors" not in st.session_state:
        vector_embedding()
        st.write("ObjectBox Database is Ready")


if input_prompt:
    retriever = st.session_state.vectors.as_retriever(search_kwargs = {"k": 5})
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": input_prompt})
    st.write(response["answer"])