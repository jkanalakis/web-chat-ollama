import streamlit as st

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# Template for generating concise answers using retrieved context
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:
"""

###############################################################################
# Streamlit UI - Model Selection
###############################################################################
st.title("AI Crawler with Model Toggle")

# Provide a dropdown or radio button for model selection
model_choice = st.selectbox(
    "Select LLM model:",
    ("Llama3.3", "deepseek-r1:14b")
)

# Convert the user choice into a valid model string
if model_choice == "Llama3.3":
    selected_model = "llama3.3"
else:
    selected_model = "deepseek-r1:14b"

# Now initialize Ollama embeddings and LLM with the user-selected model
ollama_embeddings = OllamaEmbeddings(model=selected_model)
vector_store = InMemoryVectorStore(ollama_embeddings)
ollama_model = OllamaLLM(model=selected_model)

###############################################################################
# Helper Functions
###############################################################################
def load_page_content(url: str):
    """
    Load web page content from a given URL using Selenium.

    Args:
        url (str): The URL of the webpage to load.

    Returns:
        list: A list of Document objects containing the web page content.
    """
    loader = SeleniumURLLoader(urls=[url])
    return loader.load()

def split_documents_into_chunks(documents):
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
    return text_splitter.split_documents(documents)

def index_document_chunks(document_chunks):
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
    return qa_chain.invoke({"question": question, "context": context})

###############################################################################
# Main UI Logic
###############################################################################
url_input = st.text_input("Enter URL:")
if url_input:
    documents = load_page_content(url_input)
    chunked_docs = split_documents_into_chunks(documents)
    index_document_chunks(chunked_docs)

user_query = st.chat_input()
if user_query:
    st.chat_message("user").write(user_query)

    retrieved_docs = search_similar_documents(user_query)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    answer = generate_answer(user_query, context_text)

    st.chat_message("assistant").write(answer)