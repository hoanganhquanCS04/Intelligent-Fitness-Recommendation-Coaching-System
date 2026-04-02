from rag_module.util.database import Session_local
from rag_module.model.session import Session


class SessionRepository:
    def __init__(self):
        self.db = Session_local()
    
    def save_session(self, session: Session):
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)

    def get_session_by_id(self, session_id: int):
        return self.db.query(Session).filter(Session.id == session_id).first()
    
    def get_sessions_by_user_id(self, user_id: int):
        return self.db.query(Session).filter(Session.user_id == user_id).all()