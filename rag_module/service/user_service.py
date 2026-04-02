from passlib.hash import bcrypt
from rag_module.model.user import User
from rag_module.util.database import Session_local
from fastapi import Depends
from rag_module.repository.user_repository import UserRepository

class UserService:

    def __init__(self, UserRepository: UserRepository = Depends(UserRepository)):
        self.userRepository = UserRepository()
    
    def create_user(self, user_dto):
        user = User()
        user.username = user_dto.username
        user.email = user_dto.email
        user.password = bcrypt.hash(user_dto.password)
        user.full_name = user_dto.full_name
        user.birth = user_dto.birth
        user.gender = user_dto.gender
        user.height = user_dto.height
        user.weight = user_dto.weight
        user.goal = user_dto.goal
        user.exercise_favourite = user_dto.exercise_favourite
        user.diet_favourite = user_dto.diet_favourite
        user.practice_plan = user_dto.practice_plan
        self.userRepository.save_user(user)
        return True

