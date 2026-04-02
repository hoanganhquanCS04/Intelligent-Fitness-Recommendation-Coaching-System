from fastapi import Depends, Body, APIRouter
from rag_module.service.answer_service import AnswerService

answer_router = APIRouter()

class AnswerRoute:
    def __init__(self):
        pass

    @answer_router.post(f"/api/public/v1/answers/{{session_id}}")
    def get_answer(session_id: str, question: str = Body()):
        answer_service = AnswerService(session_id=session_id)
        return answer_service.get_answer(question)
