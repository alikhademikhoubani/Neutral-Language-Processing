from typing import TypedDict, List, Literal, Annotated
from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq 
from langchain_community.vectorstores import FAISS 
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter 
from langchain_classic.schema import Document 
from dotenv import load_dotenv 
import os
import PyPDF2 as pdf 
from pydantic import BaseModel, Field 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph.message import add_messages


load_dotenv()


llm = ChatGroq(model_name = 'llama-3.3-70b-versatile')



class MessageClassifier(BaseModel):
    message_type: Literal["customers", "sellers"] = Field(
        ...,
        description = "classify the message into customers or saler"
    )



class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None 




def get_texts(pdf_path):
    with open (pdf_path, "rb") as file:
        reader = pdf.PdfReader(file)
        texts = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                texts += extracted
        return texts
    



def get_chunks(texts):
    splitter = RecursiveCharacterTextSplitter(
        separators = ["\n", "\n\n"],
        chunk_size = 500,
        chunk_overlap = 0
    )
    chunks = splitter.split_text(texts)
    docs = [Document(page_content = chunk) for chunk in chunks]
    return docs



embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")



f_texts = get_texts("C:/Users/Lenovo/Desktop/datapdf/forooshandegan.pdf")
k_texts = get_texts("C:/Users/Lenovo/Desktop/datapdf/kharidaran.pdf")


f_docs = get_chunks(f_texts)
k_docs = get_chunks(k_texts)


f_db = FAISS.from_documents(documents = f_docs, embedding = embeddings)
k_db = FAISS.from_documents(documents = k_docs, embedding = embeddings)


f_retriever = f_db.as_retriever(k = 2)
k_retriever = k_db.as_retriever(k = 2)



def classify_message(state: AgentState):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """پیام کاربر را به یکی از کلاس های زیر کلاس بندی کن:
            - 'customers': اگر سوال از جانب خریداران یک سایت فروشگاهی باشد
            - 'sellers': اگر سوال از جانب فروشندگان یک سایت فروشگاهی باشد
            """
        },
        {"role": "user", "content": last_message.content}
    ])

    return {"message_type": result.message_type}




def router(state: AgentState):
    message_type = state.get("message_type", "sellers")
    if message_type == "customers":
        return {"next": "customers"}
    
    return {"next": "sellers"}




def forooshandegan_agent(state: AgentState):
    last_message = state["messages"][-1]

    context = f_retriever.invoke(last_message.content)

    messages = [
        {
            "role": "system",
            "content": (
                "تو یک متخصص خبره در پاسخگویی به سوالات فروشندگان هستی. "
                "فقط بر اساس متن ارائه شده پاسخ بده و هیچ دانشی خارج از آن اضافه نکن. "
                "پاسخت باید دقیق، واضح و 300 کلمه باشد."
            )
        },
        {
            "role": "user",
            "content": (
                f"فقط بر اساس متن ارائه شده زیر، به سوال مطرح شده پاسخ بده:\n\n"
                f"متن:\n{context}\n\n"
                f"سوال:\n{last_message.content}"
            ),
        }
    ]

    reply = llm.invoke(messages)
    
    return {"messages": [{"role": "assistant", "content": reply.content}]}




def kharidaran_agent(state: AgentState):
    last_message = state["messages"][-1]

    context = k_retriever.invoke(last_message.content)

    messages = [
        {
            "role": "system",
            "content": (
                "تو یک متخصص خبره در پاسخگویی به سوالات خریداران هستی. "
                "فقط بر اساس متن ارائه شده پاسخ بده و هیچ دانشی خارج از آن اضافه نکن. "
                "پاسخت باید دقیق، واضح و 300 کلمه باشد."
            )
        },
        {
            "role": "user",
            "content": (
                f"فقط بر اساس متن ارائه شده زیر، به سوال مطرح شده پاسخ بده:\n\n"
                f"متن:\n{context}\n\n"
                f"سوال:\n{last_message.content}"
            ),
        }
    ]

    reply = llm.invoke(messages)
    
    return {"messages": [{"role": "assistant", "content": reply.content}]}




graph_builder = StateGraph(AgentState)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("kharidaran", kharidaran_agent)
graph_builder.add_node("forooshandegan", forooshandegan_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"customers": "kharidaran", "sellers": "forooshandegan"}
)
graph_builder.add_edge("kharidaran", END)
graph_builder.add_edge("forooshandegan", END)

graph = graph_builder.compile()




def run_chatbot():
    state = {"messages": [], "message_type": None}

    print("Chatbot is ready! (برای خروج بنویس: خداحافظ)")

    while True:
        user_input = input("User: ").strip()

        if user_input in ["خداحافظ", "bye", "exit", "quit"]:
            print("Assistant: روز خوبی داشته باشید!")
            break

        state["messages"].append({"role": "user", "content": user_input})

        state = graph.invoke(state)

        last_message = state["messages"][-1]

        print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()