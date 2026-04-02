from rag_module.util.rag import RAGModel
from rag_module.repository.session_repository import SessionRepository
from fastapi import Depends


class AnswerService:
    def __init__(self, session_id):
        self.rag_model = RAGModel(session_id=session_id)

    def get_answer(self, question):
        result = self.rag_model.ask(question)
        return result