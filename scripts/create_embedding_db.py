import os
import faiss
import pickle
import uuid
import logging
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, CSVLoader

# Load environment variables (for API keys, etc.)
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths for FAISS index and document store
FAISS_INDEX_FILE = "faiss_index.bin"
DOCSTORE_FILE = "docstore.pkl"

# Initialize embedding function
embedding_function = OpenAIEmbeddings()

# Set embedding dimension
embedding_dimension = len(embedding_function.embed_query("hello world"))

# Load or initialize FAISS index and document store
def load_faiss_index(file_path):
    if os.path.exists(file_path):
        logger.info(f"Loading FAISS index from {file_path}")
        return faiss.read_index(file_path)
    else:
        logger.warning(f"FAISS index file {file_path} not found. Creating a new index.")
        return faiss.IndexFlatL2(embedding_dimension)

def load_docstore(file_path):
    if os.path.exists(file_path):
        logger.info(f"Loading document store from {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        logger.warning(f"Document store file {file_path} not found. Creating a new store.")
        return InMemoryDocstore()

index = load_faiss_index(FAISS_INDEX_FILE)
docstore = load_docstore(DOCSTORE_FILE)

vector_store = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=docstore,
    index_to_docstore_id={}
)

# Save FAISS index and document store functions
def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)
    logger.info(f"FAISS index saved to {file_path}")

def save_docstore(docstore, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(docstore, f)
    logger.info(f"Document store saved to {file_path}")

# Function to split documents into manageable chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    return splitter.split_documents(docs)

# Function to process and add documents to the vector store
def add_documents_to_vector_store(splits):
    for chunk in splits:
        doc_id = str(uuid.uuid4())  # Unique ID for the document chunk
        document = chunk

        # Generate embedding and add it to FAISS index
        vector = embedding_function.embed_query(chunk.page_content)
        vector_store.index.add(np.array([vector]))

        # Manually update the InMemoryDocstore's internal dictionary
        vector_store.docstore._dict[doc_id] = document

        # Map FAISS index to document ID
        vector_store.index_to_docstore_id[len(vector_store.index_to_docstore_id)] = doc_id

    # Save updated index and document store
    save_faiss_index(vector_store.index, FAISS_INDEX_FILE)
    save_docstore(vector_store.docstore, DOCSTORE_FILE)
    logger.info(f"Added {len(splits)} document chunks to the vector store.")


# Function to process individual files based on their file extension
def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.txt':
        loader = TextLoader(file_path)
    elif ext == '.pdf':
        loader = PyMuPDFLoader(file_path)
    elif ext == '.csv':
        loader = CSVLoader(file_path)
    else:
        logger.error(f"Unsupported file type: {ext}")
        return

    # Load and split the documents
    docs = loader.load_and_split()
    logger.info(f"Loaded {len(docs)} documents from {file_path}")
    
    splits = split_documents(docs)
    logger.info(f"Split into {len(splits)} chunks")
    
    # Add documents to vector store
    add_documents_to_vector_store(splits)

# Function to process all files in a folder
def process_folder(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            logger.info(f"Processing file: {file_path}")
            process_file(file_path)
    logger.info("Processing of all files in folder completed successfully.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        logger.error("Usage: python process_and_embed_folder.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    if not os.path.exists(directory_path):
        logger.error(f"Directory does not exist: {directory_path}")
        sys.exit(1)

    logger.info(f"Processing folder: {directory_path}")
    process_folder(directory_path)
    logger.info("All files processed successfully.")
