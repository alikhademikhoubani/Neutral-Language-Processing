from langchain_groq import ChatGroq
import PyPDF2 as pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
from langchain.docstore.document import Document


llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    groq_api_key = "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",
    temperature = 0
)

def get_text(pdf_path):
    text = ""
    loader = pdf.PdfReader(pdf_path)
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

embeddings = HuggingFaceBgeEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True}
)

prompt = ChatPromptTemplate.from_template(
    template = """
    فقط بر اساس متن ارائه شده، به سوال پاسخ بده. 
    پاسخی که میدهی باید مناسب کاربرانی باشد که هیچ درکی از موضوع مورد بحث ندارند، بنابراین، پاسخ را تا حد ممکن ساده بیان کن.
    همچنین، از نقل قول مستقیم خودداری کن.
    متن: {context}
    سوال: {input}
    """
)


st.set_page_config("Chat with PDF")
st.header("Chat with PDF")
if "db" not in st.session_state:
    st.session_state.db = None
with st.sidebar:
    st.title("Menu")
    pdf_doc = st.file_uploader("Upload your PDF file and click on the submit & process", type = ["pdf"])
    if st.button("Submit & Process") and pdf_doc is not None:
        with st.spinner("Processing..."):
            text = get_text(pdf_doc)
            chunks = get_chunks(text)
            docs = [Document(page_content = chunk) for chunk in chunks]
            st.session_state.db = FAISS.from_documents(docs, embedding = embeddings)
            st.success("Done")
if st.session_state.db is not None:
    input_text = st.text_input("Enter your query")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": input_text})
    st.write(response["answer"])