import os
import hashlib
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDFs
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.schema import Document

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"


# Define ChromaDB storage location
persist_directory = "./vector_db"
embeddings = OpenAIEmbeddings()
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Streamlit UI
st.title("📄 AI-Powered Document Search")
st.sidebar.header("Upload Documents")

# Function to compute file hash (MD5 checksum)
def compute_file_hash(file_content):
    hasher = hashlib.md5()
    hasher.update(file_content)
    return hasher.hexdigest()

# Function to load PDFs
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to load CSVs (each row as a document)
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return [" ".join(map(str, row.values)) for _, row in df.iterrows()]

# File Upload UI
uploaded_files = st.sidebar.file_uploader("Upload TXT, PDF, or CSV", type=["txt", "pdf", "csv"], accept_multiple_files=True)

if uploaded_files:
    new_chunks = []
    
    # Get existing metadata (to check for duplicates)
    stored_metadata = vector_store.get()["metadatas"]
    existing_files = {meta["source"]: meta.get("hash", "") for meta in stored_metadata if "source" in meta}

    for file in uploaded_files:
        file_path = os.path.join("/Users/kirankolla/PersonalDocs", file.name)
        file_content = file.read()
        file_hash = compute_file_hash(file_content)

        # Check if file exists and changed
        if file.name in existing_files and existing_files[file.name] == file_hash:
            st.sidebar.warning(f"⚠️ {file.name} already indexed and unchanged.")
            continue  # Skip unchanged files

        # Save file
        with open(file_path, "wb") as f:
            f.write(file_content)

        # Load content based on file type
        if file.name.endswith(".txt"):
            text = file_content.decode("utf-8")
        elif file.name.endswith(".pdf"):
            text = load_pdf(file_path)
        elif file.name.endswith(".csv"):
            text = "\n".join(load_csv(file_path))

        # Chunk text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_text(text)

        # Create document objects
        for chunk in chunks:
            new_chunks.append(Document(page_content=chunk, metadata={"source": file.name, "hash": file_hash}))

    # Add new documents to ChromaDB
    if new_chunks:
        vector_store.add_documents(new_chunks)
        vector_store.persist()
        st.sidebar.success("✅ Files indexed successfully!")

# Query Interface
st.subheader("🔍 Ask a Question")
query = st.text_input("Type your question here and press Enter")

if query:
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)
    response = qa_chain.run(query)
    
    st.write("### 🤖 AI Answer:")
    st.success(response)

    # Show sources used
    st.write("### 📚 Sources:")
    sources = [meta["source"] for meta in vector_store.get()["metadatas"] if "source" in meta]
    st.write("\n".join(set(sources)))
