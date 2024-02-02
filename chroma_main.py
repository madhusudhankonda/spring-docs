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

langchain.cache =  InMemoryCache()

load_dotenv()

CHROMA_DB = "./chroma_db"
MODEL = os.getenv("MODEL", "llama2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL)
ollama_embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model="llama2")
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


template = """
Answer the question based on the following context. 

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def get_store():
    store = Chroma(
        persist_directory=CHROMA_DB,
        embedding_function=ollama_embeddings)

    return store

def answer_with_retriever(question):
    retriever = get_store().as_retriever(search_kwargs={"k":3})

    chain = (
        {"context":retriever, "question":RunnablePassthrough()}
        |prompt
        |llm
        |StrOutputParser()
    )

    print("q", question)
    try:
        print("Invoking the chain..", question)
        results = chain.invoke(question)
    except:
        print("Exception")


def add_qa_context(question):
    search = get_store().similarity_search(question)
    
    context = "\n".join(s.page_content for s in search) 
    return context

def answer_no_retriever(question):
    
    chain = (
    {"context":RunnableLambda(add_qa_context), "question":RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
    )

    res = chain.invoke(question)

    return res

