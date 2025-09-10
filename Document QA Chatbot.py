from langchain_groq import ChatGroq
import PyPDF2 as pdf
import streamlit as st 
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    groq_api_key = "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",
    temperature = 0
)

def vector_store():
    if "vector" not in st.session_state:
        st.session_state.texts = ""
        st.session_state.loader = pdf.PdfReader("C:/Users/Lenovo/Desktop/New folder (6)/rooze.pdf")
        for page in st.session_state.loader.pages:
            st.session_state.extracted = page.extract_text()
            if st.session_state.extracted:
                st.session_state.texts += st.session_state.extracted
        
        st.session_state.splitter = RecursiveCharacterTextSplitter(
            separators = ["\n\n", "\n", " "],
            chunk_size = 1500,
            chunk_overlap = 150
        )
        st.session_state.chunks = st.session_state.splitter.split_text(st.session_state.texts)

        st.session_state.docs = [Document(page_content = chunk) for chunk in st.session_state.chunks]

        st.session_state.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

        st.session_state.vectors = FAISS.from_documents(st.session_state.docs, embedding = st.session_state.embeddings)

prompt = ChatPromptTemplate.from_template(
    """
    فقط بر اساس متن ارائه شده، به سوال پاسخ بده.
    همچنین، بر اساس سوال پرسیده شده، پاسخ دقیق ارائه کن.
    متن: {context}
    سوال: {input}
    """
)

st.title("Groq Demo")
prompt1 = st.text_input("Enter your query")
if st.button("Documents Embedding"):
    if "vectors" not in st.session_state:
        vector_store()
    
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": prompt1})
    st.write(response["answer"])