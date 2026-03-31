from model.user import User
from dto.user_dto import UserDTO
from passlib.hash import bcrypt
from util.database import Session_local

class UserService:

    def __init__(self, user):
        self.db = Session_local()
        self.user = user
    
    def create_user(self, user_dto):
        self.user.username = user_dto.username
        self.user.email = user_dto.email
        self.user.password = bcrypt.hash(user_dto.password)
        self.user.full_name = user_dto.full_name
        self.user.birth = user_dto.birth
        self.user.gender = user_dto.gender
        self.user.height = user_dto.height
        self.user.weight = user_dto.weight
        self.user.goal = user_dto.goal
        self.user.exercise_favourite = user_dto.exercise_favourite
        self.user.diet_favourite = user_dto.diet_favourite
        self.user.practice_plan = user_dto.practice_plan
        self.db.add(self.user)
        self.db.commit()
        self.db.refresh(self.user)
        return True

