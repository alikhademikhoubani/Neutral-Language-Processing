from langchain_groq import ChatGroq
import PyPDF2 as pdf 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st 

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    groq_api_key = "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",
    temperature = 0
)

def get_doc(pdf_path):
    loader = pdf.PdfReader(pdf_path)
    text = ""
    for page in loader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " "],
        chunk_size = 1500,
        chunk_overlap = 150
    )
    chunks = splitter.split_text(text)
    return chunks

prompt = ChatPromptTemplate.from_template(
    """
    فقط بر اساس متن ارائه شده به سوال پاسخ بده.
    لطفا بر اساس سوال مطرح شده، دقیق ترین پاسخ را ارائه بده.
    متن: {context}
    سوال: {input}
    """
)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config("Chat with PDF")
st.header("Chat with PDF")
if "db" not in st.session_state:
    st.session_state.db = None
with st.sidebar:
    st.title("Menu")
    pdf_doc = st.file_uploader("Upload your PDF file and click on the submit & process", type = ["pdf"])
    if st.button("Submit & Process") and pdf_doc is not None:
        with st.spinner("Processing..."):
            text = get_doc(pdf_doc)
            chunks = get_chunks(text)
            st.session_state.db = FAISS.from_texts(chunks, embedding = embeddings)
            st.success("Done")
if st.session_state.db is not None:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.db.as_retriever(search_kwargs = {"k": 1})
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    input_text = st.text_input("Enter your query here")
    if input_text:
        response = retriever_chain.invoke({"input": input_text})
        st.write(response["answer"])
else:
    st.warning("لطفا یک فایل PDF آپلود کنید و روی Submit بزنید")