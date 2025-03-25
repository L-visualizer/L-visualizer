from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from .auth import create_access_token, get_password_hash, verify_password
from datetime import timedelta

router = APIRouter()

fake_users_db = {}  # Replace with actual database

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

@router.post("/register/")
def register(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    fake_users_db[user.username] = get_password_hash(user.password)
    return {"message": "User registered successfully"}

@router.post("/token/")
def login(user: UserLogin):
    hashed_password = fake_users_db.get(user.username)
    if not hashed_password or not verify_password(user.password, hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=30))
    return {"access_token": access_token, "token_type": "bearer"}

from services.firebase import add_user_to_firestore

@router.post("/register/")
def register(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    fake_users_db[user.username] = get_password_hash(user.password)
    add_user_to_firestore(user.username)  # Store user in Firebase
    return {"message": "User registered successfully"}
