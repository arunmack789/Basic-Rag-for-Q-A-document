import os

# Disable Streamlit file watcher to avoid torch module issues
# os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import torch
torch.classes.__path__ = [] # add this line to manually set it to empty. 


import streamlit as st
import re
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Set up Streamlit UI
st.title("📄 Document Q&A with Gemini")
st.markdown("Upload a PDF document and ask questions about its content.")

# Sidebar for API key management
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Google API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

@st.cache_resource
def get_embeddings():
    """Cache and return the embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def clean_text(text):
    """Clean text content."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'Page \d+', '', text)
    return text

def process_document(file):
    """Process uploaded PDF and create a vector store."""
    documents = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(
                    page_content=clean_text(text),
                    metadata={"page": i + 1}
                ))

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    return FAISS.from_documents(docs, embeddings)

# Process document when file is uploaded
if uploaded_file and api_key:
    with st.spinner("Processing document..."):
        st.session_state.vector_store = process_document(uploaded_file)
    st.success("Document processed successfully!")

# Input for question
query = st.text_input("Enter your question about the document:")

# Answer retrieval
if query and st.session_state.vector_store and api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
    retriever = st.session_state.vector_store.as_retriever()

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    with st.spinner("Searching for answers..."):
        response = rag_chain({"query": query})

    st.subheader("Answer:")
    st.write(response["result"])

    st.subheader("Sources:")
    for doc in response["source_documents"]:
        st.write(f"📄 Page {doc.metadata['page']}:")
        st.caption(doc.page_content[:500] + "...")
