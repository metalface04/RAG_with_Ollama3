import streamlit as st
import requests
import json
from typing import List

# FastAPI backend URL
BACKEND_URL = "http://localhost:8080"

def upload_pdf(file):
    files = {"file": file}
    response = requests.post(f"{BACKEND_URL}/pdf", files=files)
    return response.json()

def ask_question(query):
    payload = {"query": query}
    response = requests.post(f"{BACKEND_URL}/ask_pdf", json=payload)
    return response.json()

def query_ai(query):
    payload = {"query": query}
    response = requests.post(f"{BACKEND_URL}/ai", json=payload)
    return response.json()

def generate_quiz():
    payload = {"query": "quiz"}  # send a 'quiz' query to trigger quiz generation
    response = requests.post(f"{BACKEND_URL}/quiz", json=payload)
    return response.json()

st.title("PDF Question Answering and Quiz Generation System")

# file upload section
st.header("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    if st.button("Upload PDF"):
        with st.spinner("Uploading and processing PDF..."):
            try:
                result = upload_pdf(uploaded_file)
                st.success(f"PDF uploaded successfully: {result['filename']}")
                st.write(f"Document length: {result['doc_len']}")
                st.write(f"Number of chunks: {result['chunks']}")
            except Exception as e:
                st.error(f"An error occurred while uploading the PDF: {str(e)}")

# question answering section
st.header("Ask a Question about the PDF")
pdf_question = st.text_input("Enter your question about the uploaded PDF")
if st.button("Ask PDF Question"):
    if pdf_question:
        with st.spinner("Processing your question..."):
            try:
                # check if the user query contains the word "quiz"
                if "quiz" in pdf_question.lower():
                    # if "quiz" is in the query, trigger the quiz generation endpoint
                    quiz_response = generate_quiz()
                    st.subheader("Generated Quiz:")
                    st.write(quiz_response["answer"])  # display the quiz questions and answers
                    if quiz_response["sources"]:
                        st.subheader("Sources:")
                        for source in quiz_response["sources"]:
                            st.write(f"Source: {source['source']}")
                            st.write(f"Content: {source['page_content']}")
                            st.write("---")
                else:
                    # otherwise, trigger the regular question-answering endpoint
                    response = ask_question(pdf_question)
                    st.subheader("Answer:")
                    st.write(response["answer"])
                    if response["sources"]:
                        st.subheader("Sources:")
                        for source in response["sources"]:
                            st.write(f"Source: {source['source']}")
                            st.write(f"Content: {source['page_content']}")
                            st.write("---")
            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")
    else:
        st.warning("Please enter a question.")

# general AI query section
st.header("General AI Query")
ai_query = st.text_input("Enter a general question for the AI")
if st.button("Query AI"):
    if ai_query:
        with st.spinner("Processing your query..."):
            try:
                result = query_ai(ai_query)
                st.subheader("AI Response:")
                st.write(result["answer"])
            except Exception as e:
                st.error(f"An error occurred while querying the AI: {str(e)}")
    else:
        st.warning("Please enter a query.")
