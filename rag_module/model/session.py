from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from pytz import timezone
from rag_module.util.database import Model

class Session(Model):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="sessions")
    created_at = Column(String, nullable=True, default=datetime.now(timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S"))
    updated_at = Column(String, nullable=True, default=datetime.now(timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S"))

    def __init__(self):
        pass