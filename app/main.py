from fastapi import FastAPI
from .routes import users

app = FastAPI()

app.include_router(users.router, prefix="/users", tags=["users"])

@app.get("/")
def home():
    return {"message": "Welcome to the AI-Powered E-Book Accessibility Tool"}
