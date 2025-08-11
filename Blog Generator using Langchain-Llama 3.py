import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain

def getLlamaresponse(input_text, no_words, blog_style):
    llm = ChatGroq(
        model_name = "llama-3.3-70b-versatile",
        groq_api_key = "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",
        temperature = 0.01
    )

    template = """
    Write a blog for {blog_style} job profile for a topic {input_text}
    within {no_words} words.
    """

    prompt = PromptTemplate(
        input_variables = ["blog_style", "input_text", "no_words"],
        template = template
    )

    chain = LLMChain(llm = llm, prompt = prompt)
    response = chain.run({"blog_style": blog_style,
                         "input_text": input_text,
                         "no_words": no_words})
    return response




st.set_page_config(page_title = "Generate Blogs",
page_icon = "_",
layout = 'centered',
initial_sidebar_state = 'collapsed'
)

st.header("Generate Blogs")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("No of Words")
with col2:
    blog_style = st.selectbox('Writing the blog for',
    ('Researchers', 'Data Scientist', 'Common People'), index = 0)

    submit = st.button("Generate")

    if submit:
        st.write(getLlamaresponse(input_text, no_words, blog_style))