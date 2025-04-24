from langchain.document_loaders import TextLoader, CSVLoader, UnstructuredWordDocumentLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import os

CHROMA_DIR = "chroma_db"

def load_docs_from_dir(directory, loader_cls, file_ext, source_name):
    docs = []
    for file in os.listdir(directory):
        if file.endswith(file_ext):
            path = os.path.join(directory, file)
            loader = loader_cls(path)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = source_name
                doc.metadata["filename"] = file  # Add filename for better traceability
            docs.extend(loaded)
    return docs

def ingest_all():
    all_docs = []

    # === Personal info (.txt)
    all_docs += load_docs_from_dir("personal", TextLoader, ".txt", "personal")

    # === Documents (.csv)
    all_docs += load_docs_from_dir("personal", CSVLoader, ".csv", "personal")

    # === Work experience (.docx)
    all_docs += load_docs_from_dir("work_experience", UnstructuredWordDocumentLoader, ".docx", "work_experience")

    # === Utility bills, invoices, etc. (.pdf)
    all_docs += load_docs_from_dir("bills", PyPDFLoader, ".pdf", "bills")

    all_docs += load_docs_from_dir("bills", PyPDFLoader, ".pdf", "bills")

    print(f"📚 Loaded {len(all_docs)} documents.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(all_docs)

    # Create embedding & store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    db.persist()

    print(f"✅ Indexed {len(chunks)} chunks across all sources.")

if __name__ == "__main__":
    ingest_all()
