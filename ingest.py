from langchain.document_loaders import TextLoader, CSVLoader, UnstructuredWordDocumentLoader
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
            docs.extend(loaded)
    return docs

def ingest_all():
    all_docs = []

    # Personal info (.txt)
    all_docs += load_docs_from_dir("personal_info", TextLoader, ".txt", "personal")

    # Documents (.csv)
    all_docs += load_docs_from_dir("documents", CSVLoader, ".csv", "documents")

    # Work experience (.docx)
    all_docs += load_docs_from_dir("work_experience", UnstructuredWordDocumentLoader, ".docx", "work")

    print(f"Loaded {len(all_docs)} docs.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    db.persist()

    print(f"✅ Indexed {len(chunks)} chunks across all sources.")

if __name__ == "__main__":
    ingest_all()
