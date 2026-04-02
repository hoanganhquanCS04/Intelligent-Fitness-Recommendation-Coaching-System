import os
from rag_module.util.embedding import EmbeddingModel
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain 
from langchain.memory import ConversationBufferMemory
from langchain_postgres import PostgresChatMessageHistory
from uuid import uuid1
import psycopg
from dotenv import load_dotenv

load_dotenv(override=True)

class RAGModel:
    def __init__(self, session_id=None):
        if session_id is None:
            session_id = uuid1()
        self.session_id = str(session_id)
        self.sync_connection = psycopg.connect(os.getenv("db_connection_string"))
        self.history_table_name = os.getenv("historical_conversation_collection_name")
        PostgresChatMessageHistory.create_tables(self.sync_connection, self.history_table_name)
        self.chat_model = ChatOpenAI(model=os.getenv("GEMINI_MODEL_NAME"), base_url=os.getenv("GEMINI_URL_BASE"), api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3)
        self.embedding_model = EmbeddingModel()
        self.history = self.get_history()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer", chat_memory=self.history)
        self.chain = ConversationalRetrievalChain.from_llm(llm=self.chat_model, retriever=self.embedding_model.get_retriever(k=10), memory=self.memory, return_source_documents=True)

    def get_history(self):
        return PostgresChatMessageHistory(
            self.history_table_name,
            self.session_id,
            sync_connection=self.sync_connection,
        )

    def ask(self, question):
        result = self.chain.invoke(question)
        return result