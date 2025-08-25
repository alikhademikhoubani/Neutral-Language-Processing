import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType



llm = ChatGroq(
    model_name = "qwen/qwen3-32b",
    groq_api_key = "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",
    temperature = 0
)


tavily_tool = TavilySearchResults(max_results = 1, tavily_api_key = "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")


tools = [tavily_tool]


agent = initialize_agent(
    tools,
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_sparsing_errors = True,
    verbose = False
)


st.header("Web Searcher")

prompt = st.text_area("Enter what you want to search")

submit = st.button("Search")

if submit:
    if prompt:
        st.write(agent.run(prompt))
    else:
        st.warning("Please enter what you want to search")