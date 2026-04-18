from pydantic import BaseModel
from typing import Optional

class DocExerciseQuery(BaseModel):
    n: Optional[int] = 100
    page: Optional[int] = 1
    size: Optional[int] = 10