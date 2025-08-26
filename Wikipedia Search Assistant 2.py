from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
import streamlit as st 



llm = ChatGroq(
    model_name = "qwen/qwen3-32b",
    groq_api_key = "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",
    temperature = 0
)


wiki_wrapper = WikipediaAPIWrapper(lang = "fa", top_k_results = 1)
wiki_tool = WikipediaQueryRun(api_wrapper = wiki_wrapper)


tools = [wiki_tool]


agent = initialize_agent(
    tools,
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors = True,
    verbose = False
)


st.header("Wikipedia Searcher")

prompt = st.text_area("Enter what you want to search")

submit = st.button("Search")

if submit:
    if prompt:
        st.write(agent.run(prompt))
    else:
        st.warning("Please enter what you want to search")