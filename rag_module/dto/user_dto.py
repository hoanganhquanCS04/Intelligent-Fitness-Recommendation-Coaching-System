from pydantic import BaseModel

class UserDTO(BaseModel):
    id: int
    username: str
    email: str
    password: str
    full_name: str | None = None
    birth: str | None = None
    gender: str | None = None
    height: int | None = None  
    weight: int | None = None
    goal: str | None = None
    exercise_favourite: str | None = None
    diet_favourite: str | None = None
    practice_plan: str | None = None