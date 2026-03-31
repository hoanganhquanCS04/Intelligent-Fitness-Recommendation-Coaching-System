from sqlalchemy import declarative_base, Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from pytz import timezone

Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="sessions")
    created_at = Column(String, nullable=True)
    updated_at = Column(String, nullable=True)

    def create_session(self, session_id, user_id):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now(timezone("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at = datetime.now(timezone("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d %H:%M:%S")