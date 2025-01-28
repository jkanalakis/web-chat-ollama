import streamlit as st

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:
"""

ollama_embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(ollama_embeddings)
ollama_model = OllamaLLM(model="llama3.2")


def load_page_content(url: str):
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    return documents


def split_documents_into_chunks(documents: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs


def index_document_chunks(document_chunks: list):
    vector_store.add_documents(document_chunks)


def search_similar_documents(query: str):
    return vector_store.similarity_search(query)


def generate_answer(question: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    qa_chain = prompt | ollama_model
    result = qa_chain.invoke({"question": question, "context": context})
    return result


st.title("web-chat-ollamar")

url_input = st.text_input("Enter URL:")

if url_input:
    documents = load_page_content(url_input)
    chunked_documents = split_documents_into_chunks(documents)
    index_document_chunks(chunked_documents)

user_query = st.chat_input()
if user_query:
    st.chat_message("user").write(user_query)

    similar_docs = search_similar_documents(user_query)

    context_text = "\n\n".join([doc.page_content for doc in similar_docs])
    answer = generate_answer(user_query, context_text)

    st.chat_message("assistant").write(answer)