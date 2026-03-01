import firebase_admin
from firebase_admin import credentials, firestore
import os
from pathlib import Path
from dotenv import load_dotenv

backend_dir = Path(__file__).resolve().parent
load_dotenv(backend_dir / ".env")

# Initialize Firebase Admin SDK
def initialize_firebase():
    """
    Initialize Firebase with service account credentials.
    
    Set FIREBASE_CREDENTIALS environment variable to path of service account JSON,
    or GOOGLE_APPLICATION_CREDENTIALS for automatic discovery.
    """
    if not firebase_admin._apps:
        # Check for explicit credentials path
        cred_path = os.getenv("FIREBASE_CREDENTIALS") or os.getenv("FIREBASE_CREDENTIALS_PATH")

        if cred_path:
            cred_file = Path(cred_path)
            if not cred_file.is_absolute():
                cred_file = (backend_dir / cred_file).resolve()

            if not cred_file.exists():
                raise RuntimeError(
                    "Firebase credentials file was configured but not found: "
                    f"{cred_file}. Set FIREBASE_CREDENTIALS (or FIREBASE_CREDENTIALS_PATH) correctly in Backend/.env."
                )

            cred = credentials.Certificate(str(cred_file))
            firebase_admin.initialize_app(cred)
        else:
            # Fall back to default credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
            firebase_admin.initialize_app()
    
    return firestore.client()


# Global Firestore client
db = None

def get_db():
    """Get Firestore database client (singleton)"""
    global db
    if db is None:
        db = initialize_firebase()
    return db


# Collection names
INCIDENTS_COLLECTION = "incidents"
OFFICERS_COLLECTION = "officers"
CAMERAS_COLLECTION = "cameras"
USERS_COLLECTION = "users"
ALERTS_COLLECTION = "alerts"
INCIDENT_REPORTS_COLLECTION = "incident_reports"
