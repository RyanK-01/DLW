from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from firebase import create_user_document, get_user_role, get_email_by_identifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------

class UserCreate(BaseModel):
    uid: str        # Firebase Auth UID
    username: str
    phone: str      # e.g. +601xxxxxxxx
    email: str


# ---------- Routes ----------

@app.get("/")
def read_root():
    return {"message": "FastAPI Backend is alive!"}


@app.post("/users", status_code=201)
def create_user(user: UserCreate):
    created = create_user_document(user.uid, user.username, user.phone, user.email)

    if not created:
        raise HTTPException(status_code=409, detail="User already exists.")

    return {"message": "User created successfully.", "uid": user.uid}


@app.get("/resolve-login")
def resolve_login(identifier: str):
    """
    Given a username or phone number, return the associated email for Firebase Auth sign-in.
    """
    email = get_email_by_identifier(identifier)

    if email is None:
        raise HTTPException(status_code=404, detail="No account found for that username or phone number.")

    return {"email": email}


@app.get("/role")
def get_role(uid: str):
    role = get_user_role(uid)

    if role is None:
        raise HTTPException(status_code=404, detail="User not found.")

    return {"role": role}