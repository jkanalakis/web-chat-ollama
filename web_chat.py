import streamlit as st

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Template for generating concise answers using retrieved context
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:
"""

# Initialize embeddings, vector store, and LLM model
ollama_embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(ollama_embeddings)
ollama_model = OllamaLLM(model="llama3.2")


def load_page_content(url: str):
    """
    Load web page content from a given URL using Selenium.

    Args:
        url (str): The URL of the webpage to load.

    Returns:
        list: A list of Document objects containing the web page content.
    """
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    return documents


def split_documents_into_chunks(documents: list):
    """
    Split large documents into smaller chunks for improved indexing and retrieval.

    Args:
        documents (list): A list of Document objects.

    Returns:
        list: A list of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs


def index_document_chunks(document_chunks: list):
    """
    Add chunked documents to the vector store for similarity-based retrieval.

    Args:
        document_chunks (list): A list of chunked Document objects.
    """
    vector_store.add_documents(document_chunks)


def search_similar_documents(query: str):
    """
    Retrieve the most relevant documents from the vector store based on the provided query.

    Args:
        query (str): The user's query.

    Returns:
        list: A list of Document objects deemed most relevant to the query.
    """
    return vector_store.similarity_search(query)


def generate_answer(question: str, context: str) -> str:
    """
    Generate a concise answer to the user's question using the provided context.

    Args:
        question (str): The question asked by the user.
        context (str): The retrieved context from relevant documents.

    Returns:
        str: A concise answer to the user's question.
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    qa_chain = prompt | ollama_model
    result = qa_chain.invoke({"question": question, "context": context})
    return result


# --- Streamlit UI Setup ---
st.title("web-chat-ollama")

url_input = st.text_input("Enter URL:")

# Process the input URL and load content
if url_input:
    documents = load_page_content(url_input)
    chunked_documents = split_documents_into_chunks(documents)
    index_document_chunks(chunked_documents)

# Chat interface
user_query = st.chat_input()
if user_query:
    st.chat_message("user").write(user_query)

    # Retrieve relevant documents from the vector store
    similar_docs = search_similar_documents(user_query)

    # Extract the context and generate an answer
    context_text = "\n\n".join([doc.page_content for doc in similar_docs])
    answer = generate_answer(user_query, context_text)

    st.chat_message("assistant").write(answer)