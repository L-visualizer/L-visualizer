from sqlalchemy.orm import Session
from . import models, schemas
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"],deprecated="auto")

def get_user_by_email(db:Session, email: str):
    return db.query(models.User).filte(models.User.email==email).first()

def create_user(db:Session, user:schemas.UserCreate):
    hashed_password=pwd_conect.hash(user.password)
    db_user=models.User(email=user.email,password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


    
