import collections.abc
from collections.abc import Iterable
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()
# Use the environment variable "GOOGLE_API_KEY" for both configuring generativeai and embeddings.
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

question_answer_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, respond with: "Answer is not available in the context."\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    # Change model name from "gemini-pro" to "gemini" (or another valid model id as per your API documentation)
    # model = ChatGoogleGenerativeAI(model="gemini", temperature=0.3)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    question_answer_history.append({"question": user_question, "answer": response["output_text"]})
    st.write(f"**Response:** {response['output_text']}")

def main():
    st.set_page_config("Lytical Multi PDF Agent", page_icon=":scroll:")
    st.header("📚 Lytical Agent 🤖")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")
    if user_question:
        user_input(user_question)
        for i, pair in enumerate(question_answer_history, start=1):
            st.write(f"**Question {i}:** {pair['question']}")
            st.write(f"**Answer:** {pair['answer']}")
            st.write("---")

    with st.sidebar:
        st.write("---")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & Click on Submit & Process Button", 
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Completed!")
        st.write("---")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/NightingaleNath" target="_blank">Nathaniel Nkrumah Nightingale</a> | Made with ❤️ 🤖
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()



#import collections.abc
# # try:
# #     from collections.abc import Iterable
# # except ImportError:
# #     from collections import Iterable
# from collections.abc import Iterable
# # Manually create aliases for the classes expected by the third-party library
# collections.Iterable = collections.abc.Iterable
# collections.Mapping = collections.abc.Mapping
# collections.MutableSet = collections.abc.MutableSet
# collections.MutableMapping = collections.abc.MutableMapping
# # Now import the third-party library that was causing the issue
# from PyPDF2 import PdfReader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os
# import streamlit as st


# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("AIzaSyCotlHqchfsY1nhLEcT3H4Pg9uLckrMOtU"))

# question_answer_history = []

# # Function to extract text from PDF files
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Function to split the extracted text into manageable chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Function to create and save the FAISS vector store
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Define the conversational LLM chain
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. If the answer is not in
#     the context, respond with: "Answer is not available in the context."\n\n
#     Context:\n{context}?\n
#     Question:\n{question}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# # Handle user input and display chat results
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     try:
#         new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     except Exception as e:
#         st.error(f"Error loading FAISS index: {e}")
#         return

#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

#     question_answer_history.append({"question": user_question, "answer": response["output_text"]})

#     st.write(f"**Response:** {response['output_text']}")

# # Main application layout
# def main():
#     st.set_page_config("Lytical Multi PDF Agent", page_icon=":scroll:")
#     st.header("📚 Lytical Agent 🤖")

#     # User input section
#     user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")
#     if user_question:
#         user_input(user_question)

#         # Display the conversation history
#         for i, pair in enumerate(question_answer_history, start=1):
#             st.write(f"**Question {i}:** {pair['question']}")
#             st.write(f"**Answer:** {pair['answer']}")
#             st.write("---")

#     # Sidebar with settings and PDF upload
#     with st.sidebar:
#         # st.image("img/Robot.jpg")
#         st.write("---")
#         st.title("📁 PDF File's Section")
#         pdf_docs = st.file_uploader("Upload your PDF Files & Click on Submit & Process Button", 
#                                     accept_multiple_files=True)

#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Processing Completed!")

#         st.write("---")
#         # st.image("img/gkj.jpg")
#         # st.write("Lytical Agent created by @ CodeLytical")

#     # Footer with credits
#     st.markdown(
#         """
#         <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
#             © <a href="https://github.com/NightingaleNath" target="_blank">Nathaniel Nkrumah Nightingale</a> | Made with ❤️ 🤖
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# if __name__ == "__main__":
#     main()
