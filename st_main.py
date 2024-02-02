import streamlit as st
import langchain
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda
from langchain_community.chat_models import ChatOllama
from langchain.cache import InMemoryCache
from dotenv import load_dotenv
from langchain.embeddings import OllamaEmbeddings
import os

from chroma_main import answer_with_retriever, answer_no_retriever

langchain.cache =  InMemoryCache()

load_dotenv()

CHROMA_DB = "./chroma_db"
MODEL = os.getenv("MODEL", "llama2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL, temperature=0.0)
ollama_embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model="llama2")
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

st.markdown("#### How can I help you?")
question = st.text_input("What is your question?",key="english-q")
if question:
    with st.spinner("Getting the answer..."):
        question = question+f". Make sure the answer is explained with examples. Mention the references/citations and the page numbers"
        response = answer_no_retriever(question)
        st.write(response)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    print("Prompt", prompt)
    prompt = prompt+f". Make sure the answer is explained with examples. Mention the references/citations and the page numbers"

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    response = answer_no_retriever(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(response + "▌")
        message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})