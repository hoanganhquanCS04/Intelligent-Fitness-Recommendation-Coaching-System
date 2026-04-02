import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from rag_module.cloud.embedding_service import EmbeddingService
from modal import Cls
from dotenv import load_dotenv

load_dotenv(override=True)

class EmbeddingModel:
    def __init__(self, collection_name=os.getenv("vector_collection_name")):
        self.embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL_NAME"))
        self.vector_store = PGVector(connection=os.getenv("db_connection_string"), collection_name=collection_name, embeddings=self.embeddings)
    
    def add_documents(self, documents):
        ids = [doc.metadata["id"] for doc in documents]
        self.vector_store.add_documents(documents, ids=ids, metadatas=[doc.metadata for doc in documents])
        return True
    
    def get_retriever(self, search_type="similarity", k=4):
        return self.vector_store.as_retriever(search_type=search_type, k=k)