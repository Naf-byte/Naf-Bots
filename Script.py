import collections.abc
from collections.abc import Iterable
from collections.abc import Mapping, MutableSet, MutableMapping
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import time
import logging

# Assume Google Gemini API Embedding library
# (Note: Replace this with the actual library once Google Gemini's embeddings API is available)
# from google_gemini_api import GoogleGeminiEmbeddings

# Fallback to OpenAI (or other embedding providers) in case Gemini API is not available
from langchain.embeddings import OpenAIEmbeddings


# Set your API key for Google Gemini (once available)
os.environ['GOOGLE_GEMINI_API_KEY'] = 'AIzaSyCotlHqchfsY1nhLEcT3H4Pg9uLckrMOtU'


# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create the vector store using Google Gemini API
def embed_with_google_gemini(text_chunks):
    try:
        # Assuming that GoogleGeminiEmbeddings is available for embeddings
        from google_gemini_api import GoogleGeminiEmbeddings

        # Initialize the Google Gemini Embeddings object
        embeddings = GoogleGeminiEmbeddings(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        logging.error(f"Google Gemini Embedding failed: {str(e)}")
        raise e


# Function to create the vector store, using fallback to OpenAI if needed
def get_vector_store(text_chunks):
    try:
        # Try to embed using Google Gemini API
        vector_store = embed_with_google_gemini(text_chunks)
    except Exception as gemini_error:
        logging.warning("Google Gemini failed. Switching to OpenAI embeddings as a fallback.")
        
        # Fallback to OpenAI Embeddings if Google Gemini fails
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
    return vector_store


# Function to create the conversational chain
def get_conversational_chain(vector_store):
    # Assuming Google Gemini can still work with langchain's LLM framework
    from google_gemini_api import GoogleGeminiLLM

    # Initialize the LLM for chat
    llm = GoogleGeminiLLM(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(),
                                                               memory=memory)
    return conversation_chain


# Function to handle user input and display chat messages
def user_input(user_question):
    conversation = st.session_state.conversation

    if conversation is None:
        st.error("""
        Conversation chain not initialized!! \n
        Please upload PDF files first by clicking arrow button on the Top left corner
        """)
        return
        
    with st.spinner("Generating response..."):
        time.sleep(1)  # Reduce spinner time to 1 second
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chatHistory = response['chat_history']

        # Reverse the order of chat history to display recent messages on top
        st.session_state.chatHistory.reverse()

        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                st.markdown(
                    f"<div style='background-color: #32CD32; color: black; padding: 10px; border-radius: 10px; margin: 10px;'>"
                    f" [Me]🤖 :  {message.content}"
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                user_response = message.content
                st.markdown(
                    f"<div style='background-color: #00008B; color: white; padding: 10px; border-radius: 10px; margin: 10px;'>"
                    f" [You]👱 :  {user_response}"
                    "</div>",
                    unsafe_allow_html=True
                )


# Set page config and title
st.set_page_config(page_title="Naf-Chat", page_icon=":speech_balloon:")

# Sidebar CSS styling
st.sidebar_css = """
    <style>
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }
        .sidebar .sidebar-content .stButton>button {
            background-color: #2980b9 !important;
            color: white !important;
            border-radius: 5px;
        }
    </style>
"""
st.markdown(st.sidebar_css, unsafe_allow_html=True)

# Introduction popup with animated bot and title
st.write(
    """
    <div style='text-align:center;'>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1>Naf-Chat: Chat with Multiple PDF Files </h1>
        </div>
        <h2>Welcome to Naf-Chat!</h2>
        <p>Ask me anything about your PDF files, and I'll do my best to help you out.</p>
        <p>I'm a chatbot created by Nafay Ur Rehman. My purpose is to assist you with your PDF-related queries.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize conversation and chat history
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chatHistory" not in st.session_state:
    st.session_state.chatHistory = None

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    st.subheader("Upload your Documents")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Upload Button", accept_multiple_files=True)
    col1, col2 = st.columns(2)
    if col1.button("Upload"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversational_chain(vector_store)
            st.success("Processing Completed!!")

    if col2.button("Cancel"):
        st.warning("Upload canceled")

# Display user input field
user_question = st.text_input("Ask a Question from the PDF Files")
if user_question:
    user_input(user_question)
