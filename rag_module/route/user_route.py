from fastapi import APIRouter, Depends
from rag_module.service.user_service import UserService

router = APIRouter()

class UserRoute:
    def __init__(self, userService: UserService = Depends(UserService)):
        self.userService = userService
    
    @router.post("/api/public/v1/users")
    def create_user(self, user_dto):
        return self.userService.create_user(user_dto)
    
    @router.get(f"/api/public/v1/users/{{user_id}}")
    def get_user(self, user_id: int):
        return self.userService.get_user(user_id)