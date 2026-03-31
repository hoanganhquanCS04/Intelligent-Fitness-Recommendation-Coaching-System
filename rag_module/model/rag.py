import os

import psycopg
from embedding import EmbeddingModel
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_postgres import PostgresChatMessageHistory
from uuid import uuid1

class RAGModel:
    def __init__(self):
        self.model_name = os.getenv("GEMINI_MODEL_NAME")
        self.url_base = os.getenv("GEMINI_URL_BASE")
        self.chat_model = ChatOpenAI(model=self.model_name, base_url=self.url_base, temperature=0, api_key=os.getenv("GEMINI_API_KEY"))
        self.embedding_model = EmbeddingModel("fitness_coach_embeddings")
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)
        self.conversational_retrieval_chain = ConversationalRetrievalChain.from_llm(self.chat_model, retriever=self.embedding_model.get_retriever(), memory=self.memory, return_source_documents=True)
    
    def ask(self, query):
        result = self.conversational_retrieval_chain({"question": query})
        return result

    def save_conversation_to_db(self, user_id):
        sync_connection = psycopg.connect(os.getenv("db_connection_string"), autocommit=False)
        PostgresChatMessageHistory.create_tables(sync_connection, "chat_history")
        session_id = str(uuid1())
        history = PostgresChatMessageHistory("chat_history", session_id, sync_connection=sync_connection)
        for message in self.memory.load_memory_variables({})["chat_history"]:
            if message.type == "human":
                history.add_user_message(message.content)
            elif message.type == "ai":
                history.add_ai_message(message.content)
        sync_connection.commit()
        sync_connection.close()