from rag_module.repository.excercise_doc_repository import ExerciseDocRepository
from rag_module.repository.user_repository import UserRepository
from fastapi import Depends, Request
from rag_module.util.query import DocExerciseQuery


class ExerciseDocService:
    def __init__(self, request: Request, exercise_doc_repository: ExerciseDocRepository = Depends(), user_repository: UserRepository = Depends()):
        self.exercise_doc_repository = exercise_doc_repository
        self.user_repository = user_repository
        self.vector_store = request.app.state.vector_store_1

    def get_exercise_doc_by_id(self, exercise_doc_id: int):
        return self.exercise_doc_repository.get_exercise_doc_by_id(exercise_doc_id)
    
    def get_exercise_doc(self, credentials: int, query_param: DocExerciseQuery, query: str):
        user = self.user_repository.get_user_by_id(credentials)
        self.vector_store.search_kwargs = {"n": query_param.n, "page": query_param.page, "size": query_param.size}
        if not query:
            query = f"""
                {user.goal}
            """
        results = self.vector_store.similarity_search(query)
        return results