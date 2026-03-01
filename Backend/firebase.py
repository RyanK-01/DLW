import os
import json
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

# On Vercel: read credentials from FIREBASE_CREDENTIALS_JSON env var (full JSON string)
# Locally: fall back to serviceAccountKey.json file
_creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
if _creds_json:
    cred = credentials.Certificate(json.loads(_creds_json))
else:
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "./serviceAccountKey.json")
    cred = credentials.Certificate(cred_path)

firebase_admin.initialize_app(cred)

db = firestore.client()


def create_user_document(uid: str, username: str, phone: str, email: str) -> bool:
    """
    Writes a new user document to users/{uid}.
    Returns False if the user already exists, True if created.
    """
    user_ref = db.collection("users").document(uid)

    if user_ref.get().exists:
        return False

    # Determine role at registration time
    responders = db.collection("responders").where("phone", "==", phone).limit(1).get()
    role = "responder" if len(responders) > 0 else "public"

    user_ref.set({
        "username": username,
        "phone": phone,
        "email": email,
        "role": role,
        "createdAt": datetime.utcnow(),
    })

    return True


def get_email_by_identifier(identifier: str) -> str | None:
    """
    Looks up a user's email by their username or phone number.
    Returns the email string, or None if no match found.
    """
    # Try username first
    by_username = db.collection("users").where("username", "==", identifier).limit(1).get()
    if len(by_username) > 0:
        return by_username[0].to_dict().get("email")

    # Try phone number
    by_phone = db.collection("users").where("phone", "==", identifier).limit(1).get()
    if len(by_phone) > 0:
        return by_phone[0].to_dict().get("email")

    return None


def get_user_role(uid: str) -> str | None:
    """
    Checks if the user is a responder or public.
    Returns "responder", "public", or None if user not found.
    """
    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get()

    if not user_doc.exists:
        return None

    phone = user_doc.to_dict().get("phone")

    responders = db.collection("responders").where("phone", "==", phone).limit(1).get()

    if len(responders) > 0:
        return "responder"

    return "public"
