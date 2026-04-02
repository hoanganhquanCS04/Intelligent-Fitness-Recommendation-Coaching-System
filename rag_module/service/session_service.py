from fastapi import Depends
from rag_module.dto.session_dto import SessionDTO
from rag_module.model.session import Session
from rag_module.repository.session_repository import SessionRepository
from rag_module.repository.user_repository import UserRepository


class SessionService:
    def __init__(self, session_repository: SessionRepository = Depends(SessionRepository), user_repository: UserRepository = Depends(UserRepository)):
        self.session_repository = session_repository
        self.user_repository = user_repository


    def create_session(self, session_dto: SessionDTO):
        user = self.user_repository.get_user_by_id(session_dto.user_id)
        session = Session()
        session.session_id = session_dto.session_id
        session.user_id = session_dto.user_id
        session.user = user
        self.session_repository.save_session(session)
        return True
    
    def get_sessions_by_user_id(self, user_id: int):
        sessions = self.session_repository.get_sessions_by_user_id(user_id)
        return sessions