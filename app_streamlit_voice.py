import streamlit as st
import requests
import json
import base64
from typing import List

# FastAPI backend URL
BACKEND_URL = "http://localhost:8080"

# sarvam API URL
url = "https://api.sarvam.ai/text-to-speech"

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
    payload = {"query": "quiz"}
    response = requests.post(f"{BACKEND_URL}/quiz", json=payload)
    return response.json()

def text_to_speech(text, language_code="en-IN"):
    # utilizing sarvam's api here
    payload = {
        "target_language_code": language_code,
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.65,
        "loudness": 1,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1",
        "inputs": [text]
        }
    headers = {
            "api-subscription-key": "35cfb45a-6116-4157-b9bc-78b32b0cf87a",
            "Content-Type": "application/json"
        }
    try:
        response = requests.request("POST",url, json=payload, headers=headers)
        response.raise_for_status()  
        return response.json()["audios"][0]
    except requests.exceptions.RequestException as e:
        if response.status_code == 401:
            raise Exception("Authentication failed. Please check your API key.")
        elif response.status_code == 400:
            raise Exception(f"Bad request: {response.text}")
        else:
            raise Exception(f"Text-to-speech API error: {str(e)}\nResponse: {response.text}")
        
st.title("PDF Question Answering and Quiz Generation System")

# File upload section
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

# Question answering section
st.header("Ask a Question about the PDF")
pdf_question = st.text_input("Enter your question about the uploaded PDF")
if st.button("Ask PDF Question"):
    if pdf_question:
        with st.spinner("Processing your question..."):
            try:
                if "quiz" in pdf_question.lower():
                    quiz_response = generate_quiz()
                    st.subheader("Generated Quiz:")
                    st.write(quiz_response["answer"])
                    
                    # Generate audio for quiz
                    audio_data = text_to_speech(quiz_response["answer"])
                    st.audio(base64.b64decode(audio_data), format="audio/wav")
                    
                    if quiz_response["sources"]:
                        st.subheader("Sources:")
                        for source in quiz_response["sources"]:
                            st.write(f"Source: {source['source']}")
                            st.write(f"Content: {source['page_content']}")
                            st.write("---")
                else:
                    response = ask_question(pdf_question)
                    st.subheader("Answer:")
                    st.write(response["answer"])
                    
                    # Generate audio for answer
                    audio_data = text_to_speech(response["answer"])
                    st.audio(base64.b64decode(audio_data), format="audio/wav")
                    
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

# General AI query section
st.header("General AI Query")
ai_query = st.text_input("Enter a general question for the AI")
if st.button("Query AI"):
    if ai_query:
        with st.spinner("Processing your query..."):
            try:
                result = query_ai(ai_query)
                st.subheader("AI Response:")
                st.write(result["answer"])
                
                # Generate audio for AI response
                audio_data = text_to_speech(result["answer"])
                st.audio(base64.b64decode(audio_data), format="audio/wav")
            except Exception as e:
                st.error(f"An error occurred while querying the AI: {str(e)}")
    else:
        st.warning("Please enter a query.")