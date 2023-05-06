import streamlit as st
import openai
from streamlit.components.v1 import html
from langchain.document_loaders import UnstructuredPowerPointLoader

openai.api_key = st.secrets["openaiKey"]

st.set_page_config(page_title="Pitch Analyzer")

html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """

with st.sidebar:
    st.markdown("""
    # About 
    Pitch Analyzer is a helper tool built on [LangChain](https://langchain.com) to review the pitch of projects/startups to filter them, saving time for your analysts and investors.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work
    Simply upload the Powerpoint file and wait some seconds to receive the information.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    Made by [Néstor Campos](https://www.linkedin.com/in/nescampos/)
    """,
    unsafe_allow_html=True,
    )

st.markdown("""
    # Pitch Analyzer
    """)


uploaded_file = st.file_uploader("Upload the pitch in Powerpoint format", type=["ppt","pptx"], accept_multiple_files=False)

if uploaded_file is not None:
  #loader = UnstructuredPowerPointLoader(uploaded_file.name)
  #pitch_data = loader.load()
  st.write("data:", uploaded_file.name)
  

