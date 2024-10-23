import streamlit as st
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize the OpenAI API
openai.api_key = "sk--w1I2q8IU7qRryqygc1mj-yji7Gb-DAyKNoIPJASZfT3BlbkFJr1Li__qoys32kBGdqiEfhBmGFr7K8TVPVaAI86_BwA"

# Store uploaded documents
uploaded_documents = []

# Vectorizer and document vectors will be set later after file upload
vectorizer = TfidfVectorizer()
doc_vectors = None

# Function to read and store text from uploaded files
def load_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        file_content = file.read().decode("utf-8")
        documents.append(file_content)
    return documents

# Function to perform the document retrieval
def retrieve_relevant_documents(query):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors)
    best_match_index = similarities.argmax()
    return uploaded_documents[best_match_index]
# Updated function to generate response using GPT with the latest API
def generate_response_with_retrieval(query):
    relevant_document = retrieve_relevant_documents(query)
    
    # Using the 'openai.ChatCompletion.create' method for chat-based completions
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Still using GPT-3.5-turbo
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": f"Based on the following document: '{relevant_document}', answer the question: '{query}'"}
        ],
        max_tokens=150
    )
    
    # Extracting and returning the message from the response
    return response.choices[0].message.content


# Streamlit App UI
st.title("Document Analyzer")

# File uploader
uploaded_files = st.file_uploader("Upload your text files", type="txt", accept_multiple_files=True)

if uploaded_files:
    st.write("Files uploaded successfully!")

    # Load documents from uploaded files
    uploaded_documents = load_documents(uploaded_files)

    # Display uploaded files
    for idx, doc in enumerate(uploaded_documents):
        st.write(f"**Document {idx+1}**: {doc[:200]}...")  # Preview first 200 characters

    # Vectorizing the uploaded documents
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(uploaded_documents)

# Query input
query = st.text_input("Ask a question:")

if query and uploaded_documents:
    st.write("Retrieving relevant documents...")
    relevant_doc = retrieve_relevant_documents(query)
    st.write(f"Relevant document: {relevant_doc[:300]}...")  # Preview first 300 characters

    st.write("Generating response using GPT...")
    response = generate_response_with_retrieval(query)
    st.write(f"Response: {response}")
