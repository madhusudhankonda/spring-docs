import streamlit as st
import langchain
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.cache import InMemoryCache
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
import os
from PIL import Image
from chroma_main import answer, answer_no_retriever

langchain.cache = InMemoryCache()


img = Image.open("./src/spring-bot.png")
img = img.resize((800, 200))
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
            max-width: 800px; /* Adjust the maximum width as needed */
            padding: 0 20px; /* Add padding to the sides */
        }
    </style>
    """, unsafe_allow_html=True)
st.image(img, use_column_width=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    option = st.selectbox('Version', ('6.0', '5.3', '5.2'))
with col6:
    st.write("")
    if st.button("clear history", type="primary"):
        st.session_state.messages = []

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant","content": "Ask anything about the  Spring Boot!"}
    ]

if "chat_memory" not in st.session_state.keys():
        st.session_state["chat_memory"] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
memory = st.session_state["chat_memory"]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user","content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response= answer(user_prompt,memory)
            st.write(ai_response)
    new_ai_message = {"role":"assistant", "content":ai_response}
    st.session_state.messages.append(new_ai_message)