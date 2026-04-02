from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from rag_module.util.database import Model
from datetime import datetime

class User(Model):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    birth = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    height = Column(Integer, nullable=True)
    weight = Column(Integer, nullable=True)
    goal = Column(String, nullable=True)
    exercise_favourite = Column(String, nullable=True)
    diet_favourite = Column(String, nullable=True)
    practice_plan = Column(String, nullable=True)
    sessions = relationship("Session", back_populates="user")
    created_at = Column(String, nullable=True, default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    updated_at = Column(String, nullable=True, default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    update_by = Column(String, nullable=True)

    def __init__(self):
        pass