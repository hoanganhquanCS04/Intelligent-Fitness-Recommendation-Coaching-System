from fastapi import FastAPI
from rag_module.route.answer_route import answer_router

app = FastAPI()

app.include_router(answer_router)
