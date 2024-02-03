import streamlit as st
import langchain
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda
from langchain_community.chat_models import ChatOllama
from langchain.cache import InMemoryCache
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
import os
from PIL import Image
from chroma_main import answer_with_retriever, answer_no_retriever

langchain.cache =  InMemoryCache()

load_dotenv()

CHROMA_DB = "./chroma_db"
MODEL = os.getenv("MODEL", "codellama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL, temperature=0.0)
ollama_embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model="codellama")

img = Image.open("./src/spring-bot.png")
size = (800, 200)
img = img.resize(size)
st.image(img)


st.markdown("<h1 style='text-align: center; color: grey;'>Spring Framework Docs AI Assistant</h1>", unsafe_allow_html=True)
# st.markdown("<h2 style='text-align: center; color: black;'>An AI helper for Spring Framework Docs </h2>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: black;'>Projects </h2>", unsafe_allow_html=True)
    option = st.selectbox("Spring Framework Version", options=("6.0", "5.3", "5.2"))
            
    st.write("Spring Boot ")
    st.write("Spring Data ")
    st.write("Spring Cloud ")
    st.write("Spring Integration ")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your query?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # prompt = prompt+f". Make sure the answer is explained with examples. Mention the references/citations and the page numbers."
    
    response = answer_with_retriever(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # message_placeholder.markdown(response + "▌")
        message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})