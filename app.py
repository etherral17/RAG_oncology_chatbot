import os
import streamlit as st
import asyncio
import sys
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient


# known issue with libraries that use asyncio in a threaded environment
# like Streamlit. We need to manually create and set an event loop.

# For Windows compatibility, a specific policy is sometimes needed.
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Check if an event loop already exists. If not, create and set a new one.
try:
    loop = asyncio.get_running_loop()
except RuntimeError:  # 'RuntimeError: There is no current event loop...'
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# ==============================================================================
#  END OF THE FIX
# ==============================================================================


# --- Configuration and Initialization ---
load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

DB_NAME = "clinical_trials_db"
COLLECTION_NAME = "trials_collection"
INDEX_NAME = "vector_index"

if not all([MONGO_URI, GOOGLE_API_KEY]):
    st.error("Missing environment variables. Please check your .env file.")
    st.stop()

# Use Streamlit's cache to load models and database connections only once
@st.cache_resource
def get_retriever():
    """Initializes and returns the MongoDB Atlas vector search retriever."""
    try:
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        
        if collection.count_documents({}) == 0:
            st.error(
                "Your MongoDB collection is empty! "
                "Please run the `ingest.py` script first to load the data."
            )
            return None

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=INDEX_NAME,
            text_key="text" # The key where the text summary is stored by default
        )
        # Retrieve top 3 most relevant documents
        return vector_store.as_retriever(search_kwargs={'k': 10})
    except Exception as e:
        st.error(f"Failed to connect to MongoDB or initialize retriever: {e}")
        return None

@st.cache_resource
def get_llm():
    """Initializes and returns the Gemini language model."""
    try:
        # This is the line that was causing the error
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    except Exception as e:
        st.error(f"Failed to initialize the language model: {e}")
        return None

# --- RAG Chain Definition ---
def create_rag_chain():
    """Creates the complete Retrieval-Augmented Generation chain."""
    retriever = get_retriever()
    llm = get_llm()

    if not retriever or not llm:
        return None

    # This prompt template is crucial for directing the LLM
    prompt_template = """
    You are an expert assistant for querying clinical trial data.
    Your task is to answer the user's question based ONLY on the following context.
    If the information is not present in the context, clearly state that you cannot find the answer in the provided data.
    Do not make up information. Be concise and precise.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # The RAG chain pipeline
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

try:
    # --- Streamlit UI ---
    st.set_page_config(page_title="Oncology Trials Chatbot", layout="wide")
    st.title("🤖 Oncology Clinical Trials Chatbot")
    st.caption("Ask me anything about the clinical trials in our database!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Main chat input
    if prompt := st.chat_input("e.g., Which trials for multiple myeloma are withdrawn?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get the RAG chain
        rag_chain = create_rag_chain()

        if rag_chain:
            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching for answers..."):
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # This will now show a more specific error from the @st.cache_resource functions
            st.error("The RAG chain could not be initialized. Please check the error messages above.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    print(e)
    raise e