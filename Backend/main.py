from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from firebase import create_user_document, db, get_user_role
from models import CameraHeartbeat, IncidentAlert

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
    email: str


# ---------- Routes ----------

@app.get("/")
def read_root():
    return {"message": "FastAPI Backend is alive!"}


@app.post("/users", status_code=201)
def create_user(user: UserCreate):
    created = create_user_document(user.uid, user.username, user.email)

    if not created:
        raise HTTPException(status_code=409, detail="User already exists.")

    return {"message": "User created successfully.", "uid": user.uid}


@app.get("/role")
def get_role(uid: str):
    role = get_user_role(uid)

    if role is None:
        raise HTTPException(status_code=404, detail="User not found.")

    return {"role": role}


@app.post("/api/incidents/alert", status_code=201)
def ingest_incident_alert(payload: dict):
    try:
        alert = IncidentAlert.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    if alert.location is None:
        raise HTTPException(status_code=400, detail="location is required for incident alerts.")

    snapshot = payload.get("snapshot")
    latest_frame = payload.get("latestFrameJpeg")
    if not latest_frame and isinstance(snapshot, str) and snapshot:
        latest_frame = f"data:image/jpeg;base64,{snapshot}"

    now = datetime.now(timezone.utc)
    incident_doc = {
        "incident_type": alert.incident_type.value,
        "confidence": float(alert.confidence),
        "camera_id": alert.camera_id,
        "location": alert.location.model_dump(),
        "status": payload.get("status", "NEW"),
        "riskScore": float(payload.get("riskScore", alert.confidence)),
        "category": str(payload.get("category", alert.incident_type.value)),
        "lat": float(payload.get("lat", alert.location.latitude)),
        "lng": float(payload.get("lng", alert.location.longitude)),
        "latestFrameJpeg": latest_frame,
        "createdAt": now,
        "updatedAt": now,
        "source": "edge",
        "pipeline_stage": alert.pipeline_stage,
        "timestamp": alert.timestamp,
        "frame_stats": alert.frame_stats.model_dump(mode="json") if alert.frame_stats else None,
        "inference_window": alert.inference_window.model_dump(mode="json") if alert.inference_window else None,
        "fight_inference": alert.fight_inference.model_dump(mode="json") if alert.fight_inference else None,
        "evidence": alert.evidence.model_dump(mode="json") if alert.evidence else None,
    }

    doc_ref = db.collection("incidents").document()
    doc_ref.set(incident_doc)

    return {"message": "Incident alert ingested", "id": doc_ref.id}


@app.post("/api/edge/cameras/{camera_id}/heartbeat", status_code=200)
def ingest_camera_heartbeat(camera_id: str, heartbeat: CameraHeartbeat):
    if heartbeat.camera_id != camera_id:
        raise HTTPException(status_code=400, detail="camera_id mismatch between path and payload")

    db.collection("camera_heartbeats").document(camera_id).set(
        heartbeat.model_dump(mode="json")
    )
    return {"message": "heartbeat received", "camera_id": camera_id}