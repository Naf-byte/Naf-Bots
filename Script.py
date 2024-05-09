import collections.abc
# try:
#     from collections.abc import Iterable
# except ImportError:
#     from collections import Iterable
from collections.abc import Iterable
# Manually create aliases for the classes expected by the third-party library
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
# Now import the third-party library that was causing the issue
import hyper
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import time


os.environ['GOOGLE_API_KEY'] = 'AIzaSyBZP1trD9aesCfXyiV5vQClnbKsVXlM89s'


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


# Function to create the vector store
def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# Function to create the conversational chain
def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(),
                                                               memory=memory)
    return conversation_chain

# Function to handle user input and display chat messages
def user_input(user_question):
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


# Function to create the conversational chain
def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(),
                                                               memory=memory)
    return conversation_chain


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
            st.success("Processing Completed!!<")

    if col2.button("Cancel"):
        st.warning("Upload canceled")

# Display user input field
user_question = st.text_input("Ask a Question from the PDF Files")
if user_question:
    user_input(user_question)


# Function to handle user input and display chat messages
# def user_input(user_question):
#     if user_question.lower() == "who is nafay" or "who is nafay ur rehman" or "who is nafay?" or "who is nafay ur rehman?":
#         st.markdown(
#             f"<div style='background-color: #32CD32; color: black; padding: 10px; border-radius: 10px; margin: 10px;'>"
#             f" **{{Naf-Chat}}** :  He is the Founder of me, He used different Generative AI techniques to create me"
#             f" **{{Naf-Chat}}** :  Why are you asking me?"
#             "</div>",
#             unsafe_allow_html=True
#         )
#         return

#     with st.spinner("Generating response..."):
#         time.sleep(1)  # Reduce spinner time to 1 second
#         response = st.session_state.conversation({'question': user_question})
#         st.session_state.chatHistory = response['chat_history']

#         # Reverse the order of chat history to display recent messages on top
#         st.session_state.chatHistory.reverse()

#         for i, message in enumerate(st.session_state.chatHistory):
#             if i % 2 == 0:
#                 st.markdown(
#                     f"<div style='background-color: #32CD32; color: black; padding: 10px; border-radius: 10px; margin: 10px;'>"
#                     f" **{{Naf-Chat}}** :  {message.content}"
#                     "</div>",
#                     unsafe_allow_html=True
#                 )
#             else:
#                 user_response = message.content
#                 st.markdown(
#                     f"<div style='background-color: #00008B; color: white; padding: 10px; border-radius: 10px; margin: 10px;'>"
#                     f" **{{You}}** :  {user_response}"
#                     "</div>",
#                     unsafe_allow_html=True
#                 )


# # Set page config and title
# st.set_page_config(page_title="Naf-Chat", page_icon=":speech_balloon:")

# # Sidebar CSS styling
# st.sidebar_css = """
#     <style>
#         .sidebar .sidebar-content {
#             background-color: #2c3e50;
#             color: white;
#             padding: 20px;
#             border-radius: 10px;
#         }
#         .sidebar .sidebar-content .stButton>button {
#             background-color: #2980b9 !important;
#             color: white !important;
#             border-radius: 5px;
#         }
#     </style>
# """
# st.markdown(st.sidebar_css, unsafe_allow_html=True)

# # Introduction popup with animated bot and title
# st.write(
#     """
#     <div style='text-align:center;'>
#         <div style="display: flex; justify-content: space-between; align-items: center;">
#             <img src='https://cdn.dribbble.com/userupload/10543014/file/original-4703d0ba72b72f87fa49a618a24a1f6d.gif' alt='Naf-Chat Bot' style='width: 100px; height: 100px; border-radius: 50%; margin-right: 10px;'/>
#             <h1>Naf-Chat: Chat with Multiple PDF Files </h1>
#             <img src='https://cdn.dribbble.com/userupload/10543014/file/original-4703d0ba72b72f87fa49a618a24a1f6d.gif' alt='Naf-Chat Bot' style='width: 100px; height: 100px; border-radius: 50%; margin-left: 10px;'/>
#         </div>
#         <h2>Welcome to Naf-Chat!</h2>
#         <p>Ask me anything about your PDF files, and I'll do my best to help you out.</p>
#         <p>I'm a chatbot created by Nafay Ur Rehman. My purpose is to assist you with your PDF-related queries.</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # Initialize conversation and chat history
# if "conversation" not in st.session_state:
#     st.session_state.conversation = None
# if "chatHistory" not in st.session_state:
#     st.session_state.chatHistory = None

# # Sidebar for settings
# with st.sidebar:
#     st.title("Settings")
#     st.subheader("Upload your Documents")
#     pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Upload Button", accept_multiple_files=True)
#     col1, col2 = st.columns(2)
#     if col1.button("Upload"):
#         with st.spinner("Processing..."):
#             raw_text = get_pdf_text(pdf_docs)
#             text_chunks = get_text_chunks(raw_text)
#             vector_store = get_vector_store(text_chunks)
#             st.session_state.conversation = get_conversational_chain(vector_store)
#             st.success("Processing Completed!!<")

#     if col2.button("Cancel"):
#         st.warning("Upload canceled")

# # Display user input field
# user_question = st.text_input("Ask a Question from the PDF Files")
# if user_question:
#     user_input(user_question)
