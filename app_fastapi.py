from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import os
import spacy
import re
import random

# initializing the name of the app
app = FastAPI()

# path where the vector database will be stored, for this i have used chromadb
folder_path = "db"

# model used to extract relevant words
nlp = spacy.load("en_core_web_sm")

# cached LLM initialization, will be using Ollama for this
cached_llm = Ollama(model="llama3")

# using FastEmbedEmbeddings() for the embeddings
embedding = FastEmbedEmbeddings()

# using RecursiveCharacterTextSplitter which is better than CharacterTextSplitter and having a large chunk_size and chunk_overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " "]
)

# models for query input and output
# using BaseModel from PyDantic for data validation
class Query(BaseModel):
    query: str

class Source(BaseModel):
    source: str
    page_content: str

class ResponseAnswer(BaseModel):
    answer: str
    sources: List[Source] = []

# define a global variable to store relevant keywords for the last uploaded PDF
relevant_keywords_storage: List[str] = []

# Define stop words
STOP_WORDS = set([
    "i", "me", "my", "you", "your", "he", "him", "his", "she", "her", "it", "its",
    "we", "us", "our", "they", "them", "their", "what", "is", "are", "was",
    "were", "be", "been", "am", "do", "does", "did", "have", "has", "had",
    "this", "that", "these", "those", "and", "or", "but", "if", "then",
    "so", "because", "although", "however", "while", "when", "where",
    "why", "who", "whom", "whose", "which", "what", "with", "at",
    "by", "for", "to", "in", "of", "the", "a", "an", "not","how"
])

# removes characters that are not alphanumeric
def remove_non_alphanumeric(string):
    return re.sub(r'[^A-Za-z0-9]', '', string)

# function to extract relevant keywords from document text
def extract_relevant_keywords(document: str) -> List[str]:
    doc = nlp(document)
    return list(set(token.text for token in doc if token.text.lower() not in STOP_WORDS and token.is_alpha))

# function to filter out stop words from a query
def filter_query(query: str) -> List[str]:
    words = query.lower().split()
    return [remove_non_alphanumeric(word) for word in words if word not in STOP_WORDS]

# function to check if the query is relevant based on extracted keywords
def is_relevant_query(query: str, relevant_keywords: List[str]) -> bool:
    filtered_query = filter_query(query)
    print('filtered_query is ', filtered_query)
    return any(word in relevant_keywords for word in filtered_query)

# endpoint for uploading and processing PDF documents
@app.post("/pdf", response_model=dict)
async def pdf_post(file: UploadFile = File(...)):
    global relevant_keywords_storage  # use the global variable to store relevant keywords

    # check if file is pdf or not
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    save_file = f"pdf/{file.filename}"
    with open(save_file, "wb") as buffer:
        # using await to ensure that this process in asynchronous as large files might take too long
        buffer.write(await file.read())

    print(f"filename: {file.filename}")

    # load and split the PDF into docs
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()

    print(f"docs len={len(docs)}")

    # chunking the pdf
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    # extract keywords from the chunks and store them in relevant_keywords_storage
    relevant_keywords_storage = []
    for chunk in chunks:
        relevant_keywords_storage.extend(extract_relevant_keywords(chunk.page_content))

    # remove duplicates and convert to lowercase
    relevant_keywords_storage = list(set(relevant_keywords_storage))
    relevant_keywords_storage = [word.lower() for word in relevant_keywords_storage]

    # create the vector store 
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )
    # and persist the document embeddings (saving it)
    vector_store.persist()
    
    print('Relevant keywords are:', relevant_keywords_storage)
    return {
        "status": "Successfully Uploaded",
        "filename": file.filename,
        "doc_len": len(docs),
        "chunks": len(chunks),
        "relevant_keywords": relevant_keywords_storage  # include the extracted keywords in the response
    }

# post request for PDF querying
@app.post("/ask_pdf", response_model=ResponseAnswer)
async def ask_pdf_post(query: Query):
    print(f"query: {query.query}")

    # condition checks if the user's query will be needed to invoke the quiz agent
    if "quiz" in query.query:
        print("going to quiz endpoint")
        quiz_response = await ai_quiz(query)
        return quiz_response

    # check if the query is relevant based on the stored relevant keywords
    if not is_relevant_query(query.query, relevant_keywords_storage):
        print("Query is irrelevant to the document content.")
        #return ResponseAnswer(answer="The query does not match the document's content.", sources=[])
        gen_response = await ai_post(query)
        return gen_response

    # load vector store from disk
    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    # create a retriever with similarity score threshold calculated by cosine similarity
    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    # retrieve relevant documents
    docs = retriever.get_relevant_documents(query.query)
    print(f"Retrieved {len(docs)} relevant documents")

    # prepare context for the LLM
    sources = [
        Source(source=doc.metadata.get("source", "Unknown"), page_content=doc.page_content)
        for doc in docs
    ]

    # combine the page contents of the sources
    source_context = " ".join([source.page_content for source in sources])

    # use the LLM to generate a response based on the retrieved document context
    input_data = f"Answer the query '{query.query}' based on this context: {source_context}"
    
    # generate the response using the cached LLM
    response = cached_llm.generate([input_data])
    print(f"Answer: {response.generations[0][0].text}")

    return ResponseAnswer(answer=response.generations[0][0].text, sources=sources)

# post request to handle simple LLM queries
@app.post("/ai", response_model=ResponseAnswer)
async def ai_post(query: Query):
    print(f"query: {query.query}")
    response = cached_llm.generate([query.query])
    print(response)
    return ResponseAnswer(answer=response.generations[0][0].text)

# the endpoint for quiz
@app.post("/quiz", response_model=ResponseAnswer)
async def ai_quiz(query: Query):
    print(f"query: {query.query}")

    # load vector store from disk
    print("Loading vector store for quiz generation")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    # create a retriever to get relevant document sections
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 1,  # give the topmost relevant one, we can increase this and randomly select one for the question for that particular keyword
            "score_threshold": 0.1,  # threshold for relevance
        },
    )

    # randomly select 10 relevant keywords
    if len(relevant_keywords_storage) < 10:
        raise HTTPException(status_code=400, detail="Not enough keywords to generate a quiz.")

    words_for_qs = random.sample(relevant_keywords_storage, 10)
    docs = []

    # retrieve documents for each keyword
    for word in words_for_qs:
        relevant_docs = retriever.get_relevant_documents(word)
        if relevant_docs:
            docs.extend(relevant_docs)

    print(f"Retrieved {len(docs)} relevant document sections for quiz")

    if not docs:
        return ResponseAnswer(
            answer="No relevant document sections were found for quiz generation.",
            sources=[]
        )

    # combine the retrieved sections' content to use in the quiz prompt
    document_context = " ".join([doc.page_content for doc in docs])

    # Create a structured prompt for generating quiz questions
    prompt_template = """
    Based on the following document content, create a quiz with 10 multiple-choice questions.
    Each question should have 4 options and indicate the correct answer.

    Document Content:
    {document_content}

    Generate 10 questions in the following format:
    1. Question?
    a) Option 1
    b) Option 2
    c) Option 3
    d) Option 4
    Answer: [Correct answer letter]
    """

    # fill in the template with the actual document context
    prompt = prompt_template.format(document_content=document_context)

    # generate the quiz using the cached LLM
    print("Generating quiz with LLM")
    response = cached_llm.generate([prompt])

    # extract the quiz from the LLM response
    quiz_text = response.generations[0][0].text
    print(f"Generated Quiz: {quiz_text}")

    # format the result to include the quiz questions and sources
    sources = [
        Source(source=doc.metadata.get("source", "Unknown"), page_content=doc.page_content)
        for doc in docs
    ]

    return ResponseAnswer(answer=quiz_text, sources=sources)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
