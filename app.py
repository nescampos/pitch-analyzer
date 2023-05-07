import streamlit as st
import openai
import os
from streamlit.components.v1 import html
from langchain.document_loaders import UnstructuredPDFLoader
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
import time
from CAMELAgent import CAMELAgent
from PIL import Image

os.environ["OPENAI_API_KEY"] = st.secrets["openaiKey"]

assistant_role_name = "Startup Founder"
user_role_name = "Venture Capital Investor"
task = "Decide if the startup presented by the founder meets the expectations to be able to receive investment from the Venture Capital fund."
word_limit = 50

image = Image.open('logo.jpg')
st.set_page_config(page_title="Pitch Analyzer")
st.image(image, caption='')

html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """

with st.sidebar:
    st.markdown("""
    # About 
    Pitch Analyzer is a helper tool built on [LangChain](https://langchain.com) and [CAMEL](https://github.com/lightaime/camel) to review the pitch of projects/startups to train founders and prepare for the questions they receive from investors and stakeholders.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work
    Simply upload the text of your pitch. [Here you have samples](https://github.com/nescampos/pitch-analyzer/tree/main/samples)
    *Due to resource issues, this initial version (alpha) generates each text every 21 seconds.*
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

st.markdown("""
#### Enter all your pitch text and you will see the possible questions you would receive from investors and stakeholders, while an agent will try to answer from your pitch text, and if there is no information, infer an answer or not answer, so you can prepare for future events to present your startup.
""")


startup_pitch = st.text_area("Upload your startup pitch in text format", placeholder="Write all your pitch")

def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
  assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
  assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
  user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
  user_sys_msg = user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
  return assistant_sys_msg, user_sys_msg

if startup_pitch:
  assistant_inception_prompt = (
  """Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles! Never instruct me!
  We share a common interest in collaborating to successfully complete a task.
  You must help me to complete the task.
  I must help you to complete the task.
  Here is the task: {task}. Never forget our task!
  I must ask you based on your knowledge and what you have worked to be able to fulfill the task.

  I must give you one question at a time.
  You must specifically answer my question. 
  You must be honest and answer that you don't have the concrete answer if you can't answer my question well.
  Do not add additional information that I have not requested.
  You are never supposed to ask me any questions you only answer questions.
  Your answer must be declarative sentences and simple present tense.
  Unless I say the task is completed, you should always start with:

  Answer: <YOUR_SOLUTION>

  <YOUR_SOLUTION> should be specific and provide preferable answer to the question.
  Always end <YOUR_SOLUTION> with: Next request.
  The information of your startup to answer the questions is """+str(startup_pitch)
  )

  user_inception_prompt = (
  """Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always ask me questions.
  We share a common interest in collaborating to successfully complete a task.
  I must help you to complete the task.
  Here is the task: {task}. Never forget our task!
  You must ask me questions based on my knowledge so that you decide if you want to invest in my company.

  1. Instruct in the following format:
  Question: <YOUR_INSTRUCTION>

  The "Question" describes a question. In the tag <YOUR_INSTRUCTION> you need to replace with your question.

  You must give me one question at a time.
  I must write a response that appropriately completes the requested question.
  I must decline your question honestly if I cannot perform the answer due to knowledge, information, legal reasons or my capability and explain the reasons.
  You should ask me questions.
  Now you must start to ask me using the way described above.
  Do not add anything else other than your question. 
  Keep giving me questionyou think the task is completed.
  When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
  Never say <CAMEL_TASK_DONE> unless my responses have solved your task.
  The information of the startup to make the questions is """+str(startup_pitch)
  )
  task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
  task_specifier_prompt = (
  """Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
  Please make it more specific. Be creative and imaginative.
  Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
  )
  task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
  task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=1.0))
  task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                               user_role_name=user_role_name,
                                                               task=task, word_limit=word_limit)[0]
  specified_task_msg = task_specify_agent.step(task_specifier_msg)
  # st.text(f"Specified task: {specified_task_msg.content}")
  specified_task = specified_task_msg.content
  
  assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)
  assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
  user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

  # Reset agents
  assistant_agent.reset()
  user_agent.reset()

  # Initialize chats 
  assistant_msg = HumanMessage(
      content=(f"{user_sys_msg.content}. "
                  "Now start asking questions one by one. "))

  user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
  user_msg = assistant_agent.step(user_msg)
  
  chat_turn_limit, n = 10, 0
  while n < chat_turn_limit:
      n += 1
      user_ai_msg = user_agent.step(assistant_msg)
      user_msg = HumanMessage(content=user_ai_msg.content)
      st.text(f"VC Investor ({user_role_name}):\n\n{user_msg.content}\n\n")
      
      time.sleep(21)

      assistant_ai_msg = assistant_agent.step(user_msg)
      assistant_msg = HumanMessage(content=assistant_ai_msg.content)
      st.text(f"Startup Founder ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
      time.sleep(21)
      
      if "<CAMEL_TASK_DONE>" in user_msg.content:
          break

