"""Step 4: similarity search retrieval from ChromaDB."""

import os
import warnings
import logging

warnings.filterwarnings("ignore")                          
os.environ["ANONYMIZED_TELEMETRY"] = "False"               
logging.getLogger("chromadb").setLevel(logging.CRITICAL)  

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DB_DIR = os.path.join(BASE_DIR, "..", "chroma_db")


def get_retriever(persist_directory=DB_DIR):
    """Load Chroma store and return a similarity retriever (top-3)."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    return retriever


def retrieve(query, retriever):
    """Run similarity search and print results."""
    results = retriever.invoke(query)

    print(f"Q: {query}")
    print(f"Found {len(results)} relevant chunk(s)\n")
    for i, doc in enumerate(results):
        print(f"  Result {i+1}: {doc.page_content[:100]}")

    return results


if __name__ == "__main__":
    retriever = get_retriever()
    query = "What should I do if someone gets bitten?"
    retrieve(query, retriever)
