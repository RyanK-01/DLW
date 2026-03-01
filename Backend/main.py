from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from datetime import datetime
from typing import List, Optional
import os
import importlib
import json
from pathlib import Path
from urllib.parse import urlparse
from firebase_admin import firestore
from pydantic import BaseModel, Field
try:
    from .firebase_config import (
        get_db,
        INCIDENTS_COLLECTION,
        OFFICERS_COLLECTION,
        CAMERAS_COLLECTION,
        USERS_COLLECTION,
        ALERTS_COLLECTION,
        INCIDENT_REPORTS_COLLECTION,
    )
    from .models import (
        IncidentAlert, Incident, IncidentStatus, IncidentClaimRequest,
        IncidentVerifyRequest, IncidentAttendRequest, IncidentCompleteRequest,
        Officer, OfficerStatus, OfficerLocationUpdate,
        Camera, User, UserRegistration, Location, CameraHeartbeat
    )
except ImportError:
    from firebase_config import (
        get_db,
        INCIDENTS_COLLECTION,
        OFFICERS_COLLECTION,
        CAMERAS_COLLECTION,
        USERS_COLLECTION,
        ALERTS_COLLECTION,
        INCIDENT_REPORTS_COLLECTION,
    )
    from models import (
        IncidentAlert, Incident, IncidentStatus, IncidentClaimRequest,
        IncidentVerifyRequest, IncidentAttendRequest, IncidentCompleteRequest,
        Officer, OfficerStatus, OfficerLocationUpdate,
        Camera, User, UserRegistration, Location, CameraHeartbeat
    )

app = FastAPI(title="DLW CCTV Incident Detection API")


class AlertAcknowledgeRequest(BaseModel):
    officer_id: str
    accepted: bool = True


class OfficerConnectionManager:
    def __init__(self):
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, officer_id: str, websocket: WebSocket):
        await websocket.accept()
        self._connections[officer_id] = websocket

    def disconnect(self, officer_id: str):
        self._connections.pop(officer_id, None)

    def get(self, officer_id: str) -> Optional[WebSocket]:
        return self._connections.get(officer_id)

    def is_connected(self, officer_id: str) -> bool:
        return officer_id in self._connections


ws_manager = OfficerConnectionManager()


def encode_geohash(latitude: float, longitude: float) -> Optional[str]:
    """Best-effort geohash encoding; returns None if geofire package is unavailable."""
    try:
        geofire = importlib.import_module("geofire")
        return geofire.encode(latitude, longitude)
    except Exception:
        return None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to Vercel frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "DLW CCTV Backend is running", "status": "ok"}


@app.get("/health")
def health_check():
    """Health check endpoint for Render"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.websocket("/ws/officers/{officer_id}")
async def officer_ws(websocket: WebSocket, officer_id: str):
    """Real-time officer channel for dispatch alerts."""
    await ws_manager.connect(officer_id, websocket)
    db = get_db()

    # Deliver pending alerts on connect
    pending = fetch_pending_alerts_for_officer(db, officer_id)
    if pending:
        await push_alerts_to_officer(officer_id, pending)
        mark_alerts_delivered(db, [a["id"] for a in pending])

    try:
        while True:
            # Keep socket alive and support lightweight commands.
            raw_message = await websocket.receive_text()
            if raw_message == "fetch_pending":
                pending = fetch_pending_alerts_for_officer(db, officer_id)
                if pending:
                    await push_alerts_to_officer(officer_id, pending)
                    mark_alerts_delivered(db, [a["id"] for a in pending])
            elif raw_message == "ping":
                await websocket.send_text("pong")
            elif raw_message:
                # Ignore unknown messages to keep channel tolerant.
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(officer_id)


@app.post("/api/edge/cameras/{camera_id}/heartbeat")
async def upsert_camera_heartbeat(camera_id: str, heartbeat: CameraHeartbeat):
    """
    Receive periodic edge heartbeat for a camera stream.
    Tracks stream health, fps, and latest scene metrics for monitoring.
    """
    if heartbeat.camera_id != camera_id:
        raise HTTPException(status_code=400, detail="camera_id path/body mismatch")

    db = get_db()
    camera_ref = db.collection(CAMERAS_COLLECTION).document(camera_id)
    camera_doc = camera_ref.get()

    if not camera_doc.exists:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_ref.update({
        "runtime_state.last_heartbeat_at": heartbeat.timestamp,
        "runtime_state.latest_people_count": heartbeat.latest_people_count,
        "runtime_state.latest_motion_score": heartbeat.latest_motion_score,
        "runtime_state.operational_status": "online",
        "runtime_state.incident_state": "monitoring",
        "last_seen_at": heartbeat.timestamp,
        "edge_metrics.input_fps": heartbeat.input_fps,
        "edge_metrics.processed_fps": heartbeat.processed_fps,
        "edge_metrics.dropped_frames": heartbeat.dropped_frames,
    })

    return {"status": "heartbeat_recorded", "camera_id": camera_id}


# ============================================================================
# INCIDENT ENDPOINTS
# ============================================================================

@app.post("/api/incidents/alert", status_code=status.HTTP_201_CREATED)
async def create_incident_alert(alert: IncidentAlert):
    """
    Receive incident alert from edge ML processors.
    
    Flow:
    1. Validate alert and lookup camera location if needed
    2. Create incident record in Firestore
    3. Find officers within configured radius
    4. Rank officers by ETA
    5. Notify top N officers (WebSocket/push)
    6. Return incident ID
    """
    db = get_db()
    
    # TODO: Lookup camera location if not provided
    if not alert.location:
        camera_ref = db.collection(CAMERAS_COLLECTION).document(alert.camera_id)
        camera_doc = camera_ref.get()
        if not camera_doc.exists:
            raise HTTPException(status_code=404, detail=f"Camera {alert.camera_id} not found")
        camera_data = camera_doc.to_dict()
        alert.location = Location(**camera_data["location"])
    
    # Generate geohash for location
    geohash = encode_geohash(alert.location.latitude, alert.location.longitude)
    alert.location.geohash = geohash
    
    # Create incident record
    incident = Incident(
        incident_type=alert.incident_type,
        confidence=alert.confidence,
        camera_id=alert.camera_id,
        location=alert.location,
        status=IncidentStatus.DETECTED,
        created_at=alert.timestamp,
        source=alert.pipeline_stage,
        evidence=alert.evidence,
        frame_stats=alert.frame_stats,
        inference_window=alert.inference_window,
        fight_inference=alert.fight_inference,
    )
    
    # Save to Firestore
    incident_ref = db.collection(INCIDENTS_COLLECTION).add(incident.dict(exclude={"id"}))
    incident.id = incident_ref[1].id

    # Dispatch configuration
    dispatch_radius_km = float(os.getenv("OFFICER_NOTIFICATION_RADIUS_KM", "5.0"))
    top_n = int(os.getenv("TOP_N_OFFICERS_TO_NOTIFY", "3"))

    ranked_officers = rank_officers_by_eta(
        db=db,
        incident_lat=alert.location.latitude,
        incident_lon=alert.location.longitude,
        radius_km=dispatch_radius_km,
        limit=top_n,
    )

    notified_officer_ids = [o["id"] for o in ranked_officers]

    # Persist dispatch alerts (for websocket/push delivery workers)
    dispatch_alert_ids = create_dispatch_alert_records(
        db=db,
        incident_id=incident.id,
        incident_type=incident.incident_type.value,
        camera_id=incident.camera_id,
        ranked_officers=ranked_officers,
    )

    # Update incident state
    update_payload = {
        "notified_officers": notified_officer_ids,
        "dispatch": {
            "radius_km": dispatch_radius_km,
            "top_n": top_n,
            "alert_ids": dispatch_alert_ids,
        },
    }
    if notified_officer_ids:
        update_payload["status"] = IncidentStatus.NOTIFIED.value

    db.collection(INCIDENTS_COLLECTION).document(incident.id).update(update_payload)

    # Attempt immediate realtime delivery to connected officers.
    delivered_count = await dispatch_alerts_realtime(db, dispatch_alert_ids)

    return {
        "incident_id": incident.id,
        "status": "alert_received",
        "notified_officers": notified_officer_ids,
        "dispatch_alert_count": len(dispatch_alert_ids),
        "realtime_delivered_count": delivered_count,
    }


@app.get("/api/incidents/{incident_id}")
async def get_incident(incident_id: str):
    """Get incident details by ID"""
    db = get_db()
    doc = db.collection(INCIDENTS_COLLECTION).document(incident_id).get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    data = doc.to_dict()
    data["id"] = doc.id
    return data


@app.post("/api/incidents/{incident_id}/claim")
async def claim_incident(incident_id: str, request: IncidentClaimRequest):
    """
    Officer claims an incident to respond.
    Uses Firestore transaction to prevent duplicate claims.
    """
    db = get_db()
    incident_ref = db.collection(INCIDENTS_COLLECTION).document(incident_id)
    officer_ref = db.collection(OFFICERS_COLLECTION).document(request.officer_id)
    
    @firestore.transactional
    def claim_transaction(transaction):
        # Read current incident state
        incident_doc = incident_ref.get(transaction=transaction)
        if not incident_doc.exists:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        incident_data = incident_doc.to_dict()
        
        # Check if already claimed
        if incident_data.get("claimed_by"):
            raise HTTPException(
                status_code=409, 
                detail=f"Incident already claimed by officer {incident_data['claimed_by']}"
            )
        
        # Update incident
        transaction.update(incident_ref, {
            "status": IncidentStatus.CLAIMED.value,
            "claimed_by": request.officer_id,
            "claimed_at": datetime.utcnow(),
        })
        
        # Update officer status
        transaction.update(officer_ref, {
            "status": OfficerStatus.RESPONDING.value,
            "current_incident": incident_id,
        })
    
    transaction = db.transaction()
    claim_transaction(transaction)

    return {
        "status": "claimed",
        "incident_id": incident_id,
        "officer_id": request.officer_id,
        "public_advisory": {
            "status": "not_triggered",
            "reason": "advisory_triggers_on_attend",
        },
    }


@app.post("/api/incidents/{incident_id}/verify")
async def verify_incident_detection(incident_id: str, request: IncidentVerifyRequest):
    """Officer verifies whether incident detection is true/false positive after media review."""
    db = get_db()
    incident_ref = db.collection(INCIDENTS_COLLECTION).document(incident_id)
    incident_doc = incident_ref.get()

    if not incident_doc.exists:
        raise HTTPException(status_code=404, detail="Incident not found")

    incident_data = incident_doc.to_dict() or {}
    assert_officer_can_verify_incident(db, incident_data, request.officer_id)

    verdict = "true_positive" if request.is_true_positive else "false_positive"
    incident_ref.update(
        {
            "verification": {
                "verified_by": request.officer_id,
                "verdict": verdict,
                "notes": request.notes,
                "verified_at": datetime.utcnow(),
            }
        }
    )

    return {
        "incident_id": incident_id,
        "officer_id": request.officer_id,
        "verification": {
            "verdict": verdict,
            "notes": request.notes,
        },
    }


@app.post("/api/incidents/{incident_id}/attend")
async def attend_incident(incident_id: str, request: IncidentAttendRequest):
    """Officer marks incident as actively attended/responding on scene."""
    db = get_db()
    incident_ref = db.collection(INCIDENTS_COLLECTION).document(incident_id)
    officer_ref = db.collection(OFFICERS_COLLECTION).document(request.officer_id)

    @firestore.transactional
    def attend_transaction(transaction):
        incident_doc = incident_ref.get(transaction=transaction)
        if not incident_doc.exists:
            raise HTTPException(status_code=404, detail="Incident not found")

        officer_doc = officer_ref.get(transaction=transaction)
        if not officer_doc.exists:
            raise HTTPException(status_code=404, detail="Officer not found")

        incident_data = incident_doc.to_dict() or {}
        claimed_by = incident_data.get("claimed_by")
        if claimed_by and claimed_by != request.officer_id:
            raise HTTPException(status_code=409, detail=f"Incident is claimed by officer {claimed_by}")

        now = datetime.utcnow()
        update_fields = {
            "status": IncidentStatus.ATTENDING.value,
            "attending_by": request.officer_id,
            "attending_at": now,
            "attendance": {
                "officer_id": request.officer_id,
                "notes": request.notes,
                "started_at": now,
            },
        }
        if not claimed_by:
            update_fields["claimed_by"] = request.officer_id
            update_fields["claimed_at"] = now

        transaction.update(incident_ref, update_fields)
        transaction.update(
            officer_ref,
            {
                "status": OfficerStatus.RESPONDING.value,
                "current_incident": incident_id,
                "last_updated": now,
            },
        )

    transaction = db.transaction()
    attend_transaction(transaction)

    advisory_result = trigger_public_safety_advisory(
        db=db,
        incident_id=incident_id,
        triggered_by=f"attend:{request.officer_id}",
    )

    return {
        "incident_id": incident_id,
        "status": IncidentStatus.ATTENDING.value,
        "officer_id": request.officer_id,
        "public_advisory": advisory_result,
        "message": "Officer is now attending this incident",
    }


@app.post("/api/incidents/{incident_id}/complete")
async def complete_incident(incident_id: str, request: IncidentCompleteRequest):
    """Officer completes incident and generates an incident report."""
    db = get_db()
    incident_ref = db.collection(INCIDENTS_COLLECTION).document(incident_id)
    officer_ref = db.collection(OFFICERS_COLLECTION).document(request.officer_id)

    completion_context: dict = {}

    @firestore.transactional
    def complete_transaction(transaction):
        incident_doc = incident_ref.get(transaction=transaction)
        if not incident_doc.exists:
            raise HTTPException(status_code=404, detail="Incident not found")

        officer_doc = officer_ref.get(transaction=transaction)
        if not officer_doc.exists:
            raise HTTPException(status_code=404, detail="Officer not found")

        incident_data = incident_doc.to_dict() or {}
        assigned_officer = incident_data.get("attending_by") or incident_data.get("claimed_by")
        if assigned_officer and assigned_officer != request.officer_id:
            raise HTTPException(status_code=409, detail=f"Incident is assigned to officer {assigned_officer}")

        now = datetime.utcnow()
        transaction.update(
            incident_ref,
            {
                "status": IncidentStatus.COMPLETED.value,
                "resolved_at": now,
                "completed_by": request.officer_id,
                "completion": {
                    "officer_id": request.officer_id,
                    "resolution_summary": request.resolution_summary,
                    "actions_taken": request.actions_taken,
                    "casualties_reported": request.casualties_reported,
                    "injuries_reported": request.injuries_reported,
                    "notes": request.notes,
                    "completed_at": now,
                },
            },
        )

        transaction.update(
            officer_ref,
            {
                "status": OfficerStatus.AVAILABLE.value,
                "current_incident": None,
                "last_updated": now,
            },
        )

        completion_context["incident_data"] = incident_data
        completion_context["completed_at"] = now

    transaction = db.transaction()
    complete_transaction(transaction)

    report_payload = build_incident_report_payload(
        incident_id=incident_id,
        officer_id=request.officer_id,
        incident_data=completion_context.get("incident_data") or {},
        completed_at=completion_context.get("completed_at") or datetime.utcnow(),
        resolution_summary=request.resolution_summary,
        actions_taken=request.actions_taken,
        casualties_reported=request.casualties_reported,
        injuries_reported=request.injuries_reported,
        notes=request.notes,
    )
    report_ref = db.collection(INCIDENT_REPORTS_COLLECTION).add(report_payload)
    report_id = report_ref[1].id

    incident_ref.update({"report_id": report_id})

    return {
        "incident_id": incident_id,
        "status": IncidentStatus.COMPLETED.value,
        "officer_id": request.officer_id,
        "report_id": report_id,
    }


@app.get("/api/incidents/{incident_id}/report")
async def get_incident_report(incident_id: str):
    """Fetch generated report for a completed incident."""
    db = get_db()
    incident_doc = db.collection(INCIDENTS_COLLECTION).document(incident_id).get()
    if not incident_doc.exists:
        raise HTTPException(status_code=404, detail="Incident not found")

    incident_data = incident_doc.to_dict() or {}
    report_id = incident_data.get("report_id")
    if not report_id:
        raise HTTPException(status_code=404, detail="Incident report not generated yet")

    report_doc = db.collection(INCIDENT_REPORTS_COLLECTION).document(report_id).get()
    if not report_doc.exists:
        raise HTTPException(status_code=404, detail="Incident report document not found")

    report = report_doc.to_dict() or {}
    report["id"] = report_doc.id
    return report


@app.get("/api/officers/{officer_id}/alerts/pending")
async def get_officer_pending_alerts(officer_id: str):
    """Pull endpoint for pending alerts (fallback when websocket unavailable)."""
    db = get_db()
    alerts = fetch_pending_alerts_for_officer(db, officer_id)
    return {"officer_id": officer_id, "pending_count": len(alerts), "alerts": alerts}


@app.post("/api/alerts/{alert_id}/ack")
async def acknowledge_alert(alert_id: str, payload: AlertAcknowledgeRequest):
    """Officer acknowledges (accepts/declines) a dispatch alert."""
    db = get_db()
    alert_ref = db.collection(ALERTS_COLLECTION).document(alert_id)
    alert_doc = alert_ref.get()

    if not alert_doc.exists:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert_data = alert_doc.to_dict() or {}
    if alert_data.get("officer_id") != payload.officer_id:
        raise HTTPException(status_code=403, detail="Officer is not assigned to this alert")

    new_status = "acknowledged" if payload.accepted else "declined"
    alert_ref.update({
        "status": new_status,
        "acknowledged_at": datetime.utcnow(),
        "accepted": payload.accepted,
    })

    return {
        "alert_id": alert_id,
        "status": new_status,
        "public_advisory": {
            "status": "not_triggered",
            "reason": "advisory_triggers_on_attend",
        },
    }


@app.post("/api/incidents/{incident_id}/advisory/test")
async def test_public_advisory(incident_id: str, force: bool = False):
    """
    Manual test endpoint for public safety advisories.
    - force=false: respects idempotency and existing advisory status
    - force=true: resets advisory state before re-triggering (for QA/testing)
    """
    db = get_db()
    incident_ref = db.collection(INCIDENTS_COLLECTION).document(incident_id)
    incident_doc = incident_ref.get()

    if not incident_doc.exists:
        raise HTTPException(status_code=404, detail="Incident not found")

    if force:
        incident_ref.update({"public_advisory": {"status": "reset_for_test", "updated_at": datetime.utcnow()}})

    result = trigger_public_safety_advisory(
        db=db,
        incident_id=incident_id,
        triggered_by="manual_test_endpoint",
    )

    return {
        "incident_id": incident_id,
        "force": force,
        "public_advisory": result,
    }


@app.get("/api/incidents/{incident_id}/video/stream")
async def get_incident_video_stream(incident_id: str):
    """
    Proxy live video feed for officer verification.
    Returns camera stream URL or proxies feed when connectivity permits.
    """
    db = get_db()
    incident_doc = db.collection(INCIDENTS_COLLECTION).document(incident_id).get()
    
    if not incident_doc.exists:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    incident_data = incident_doc.to_dict()
    camera_id = incident_data.get("camera_id")
    
    # Lookup camera stream URL
    camera_doc = db.collection(CAMERAS_COLLECTION).document(camera_id).get()
    if not camera_doc.exists:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    camera_data = camera_doc.to_dict()
    stream_url = camera_data.get("stream_url")
    
    if not stream_url:
        raise HTTPException(status_code=503, detail="Video stream unavailable")
    
    evidence = incident_data.get("evidence") or {}

    # TODO: Implement actual video proxy/streaming logic for RTSP/HLS sources
    return {
        "stream_url": stream_url,
        "camera_id": camera_id,
        "incident_id": incident_id,
        "evidence_clip_uri": evidence.get("clip_uri"),
        "evidence_keyframe_uris": evidence.get("keyframe_uris", []),
    }


@app.get("/api/incidents/{incident_id}/verification-media")
async def get_incident_verification_media(incident_id: str, officer_id: str):
    """
    Officer verification media descriptor for a specific incident.
    Returns live stream URL + evidence clip/keyframes produced by edge worker.
    """
    db = get_db()
    incident_doc = db.collection(INCIDENTS_COLLECTION).document(incident_id).get()
    if not incident_doc.exists:
        raise HTTPException(status_code=404, detail="Incident not found")

    incident_data = incident_doc.to_dict() or {}
    assert_officer_can_verify_incident(db, incident_data, officer_id)

    camera_id = incident_data.get("camera_id")
    if not camera_id:
        raise HTTPException(status_code=400, detail="Incident has no camera_id")

    camera_doc = db.collection(CAMERAS_COLLECTION).document(camera_id).get()
    if not camera_doc.exists:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_data = camera_doc.to_dict() or {}
    stream_url = camera_data.get("stream_url")
    evidence = incident_data.get("evidence") or {}
    clip_uri = evidence.get("clip_uri")
    resolved_clip = resolve_local_evidence_path(clip_uri) if clip_uri else None

    return {
        "incident_id": incident_id,
        "officer_id": officer_id,
        "camera_id": camera_id,
        "live_stream": {
            "url": stream_url,
            "available": bool(stream_url),
        },
        "evidence": {
            "clip_uri": clip_uri,
            "clip_duration_seconds": evidence.get("clip_duration_seconds"),
            "keyframe_uris": evidence.get("keyframe_uris", []),
            "generated_at": evidence.get("generated_at"),
            "clip_download_available": bool(resolved_clip),
            "clip_download_url": (
                f"/api/incidents/{incident_id}/evidence/clip?officer_id={officer_id}"
                if resolved_clip
                else None
            ),
        },
        "source": incident_data.get("source", "edge"),
        "created_at": incident_data.get("created_at"),
    }


@app.get("/api/incidents/{incident_id}/evidence/clip")
async def download_incident_evidence_clip(incident_id: str, officer_id: str):
    """Download/stream locally available evidence clip for officer verification."""
    db = get_db()
    incident_doc = db.collection(INCIDENTS_COLLECTION).document(incident_id).get()
    if not incident_doc.exists:
        raise HTTPException(status_code=404, detail="Incident not found")

    incident_data = incident_doc.to_dict() or {}
    assert_officer_can_verify_incident(db, incident_data, officer_id)

    evidence = incident_data.get("evidence") or {}
    clip_uri = evidence.get("clip_uri")
    if not clip_uri:
        raise HTTPException(status_code=404, detail="Incident evidence clip not available")

    resolved_clip = resolve_local_evidence_path(clip_uri)
    if resolved_clip is None:
        raise HTTPException(
            status_code=404,
            detail="Evidence clip is not locally accessible by backend (may be remote edge path)",
        )

    return FileResponse(
        path=str(resolved_clip),
        media_type="video/mp4",
        filename=f"incident_{incident_id}.mp4",
    )


# ============================================================================
# OFFICER ENDPOINTS
# ============================================================================

@app.get("/api/officers/nearby")
async def get_nearby_officers(
    latitude: float, 
    longitude: float, 
    radius_km: float = 5.0,
    limit: int = 10
):
    """
    Query officers within radius of incident location.
    Uses geohash for efficient spatial queries.
    """
    db = get_db()
    
    # Generate geohash bounds for radius
    center = (latitude, longitude)
    geohash_center = encode_geohash(latitude, longitude)
    
    # TODO: Implement proper geohash range query
    # For now, fetch all available officers and filter
    officers_ref = db.collection(OFFICERS_COLLECTION).where("status", "==", OfficerStatus.AVAILABLE.value)
    officers = []
    
    for doc in officers_ref.stream():
        officer_data = doc.to_dict()
        officer_data["id"] = doc.id
        
        # Calculate distance (basic haversine)
        officer_loc = officer_data["location"]
        distance = calculate_distance(
            latitude, longitude,
            officer_loc["latitude"], officer_loc["longitude"]
        )
        
        if distance <= radius_km:
            officer_data["distance_km"] = distance
            officers.append(officer_data)
    
    # Sort by distance (TODO: integrate ETA calculation)
    officers.sort(key=lambda x: x["distance_km"])
    
    return {"officers": officers[:limit], "count": len(officers)}


@app.put("/api/officers/{officer_id}/location")
async def update_officer_location(officer_id: str, location: OfficerLocationUpdate):
    """Update officer's current location with geohash"""
    db = get_db()
    
    geohash = encode_geohash(location.latitude, location.longitude)
    
    officer_ref = db.collection(OFFICERS_COLLECTION).document(officer_id)
    officer_ref.update({
        "location": {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "geohash": geohash,
        },
        "last_updated": datetime.utcnow(),
    })
    
    return {"status": "location_updated", "officer_id": officer_id}


# ============================================================================
# USER ENDPOINTS
# ============================================================================

@app.post("/api/users/register", status_code=status.HTTP_201_CREATED)
async def register_user(registration: UserRegistration):
    """
    Register user with phone number for SMS public safety advisories.
    Optionally capture initial location.
    """
    db = get_db()
    
    user = User(
        id="",  # Will be assigned by Firestore
        phone_number=registration.phone_number,
        location=Location(
            latitude=registration.latitude,
            longitude=registration.longitude,
            geohash=encode_geohash(registration.latitude, registration.longitude)
        ) if registration.latitude and registration.longitude else None,
        opted_in=True,
        created_at=datetime.utcnow(),
    )
    
    user_ref = db.collection(USERS_COLLECTION).add(user.dict(exclude={"id"}))
    user_id = user_ref[1].id
    
    return {"user_id": user_id, "status": "registered"}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    Returns distance in kilometers.
    """
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def resolve_local_evidence_path(raw_uri: str) -> Optional[Path]:
    """
    Resolve incident evidence clip URI to a local file path if it is safe and accessible.
    Only allows files under configured evidence roots.
    """
    if not raw_uri:
        return None

    parsed = urlparse(raw_uri)
    if parsed.scheme in {"http", "https", "rtsp", "s3", "gs"}:
        return None

    if parsed.scheme == "file":
        candidate = Path(parsed.path)
    else:
        candidate = Path(raw_uri)

    if not candidate.is_absolute():
        candidate = (Path(__file__).resolve().parent / candidate).resolve()
    else:
        candidate = candidate.resolve()

    allowed_roots = get_allowed_evidence_roots()
    if not any(is_subpath(candidate, root) for root in allowed_roots):
        return None

    if not candidate.exists() or not candidate.is_file():
        return None

    return candidate


def assert_officer_can_verify_incident(db, incident_data: dict, officer_id: str) -> None:
    """Allow verification only for officers notified or assigned to the incident."""
    if not officer_id:
        raise HTTPException(status_code=400, detail="officer_id is required")

    officer_doc = db.collection(OFFICERS_COLLECTION).document(officer_id).get()
    if not officer_doc.exists:
        raise HTTPException(status_code=404, detail="Officer not found")

    notified = set(incident_data.get("notified_officers") or [])
    claimed_by = incident_data.get("claimed_by")
    if officer_id != claimed_by and officer_id not in notified:
        raise HTTPException(status_code=403, detail="Officer is not authorized to view this incident media")


def compute_duration_minutes(start_time: Optional[datetime], end_time: datetime) -> Optional[float]:
    if not start_time:
        return None
    return round(max(0.0, (end_time - start_time).total_seconds() / 60.0), 2)


def build_incident_report_payload(
    incident_id: str,
    officer_id: str,
    incident_data: dict,
    completed_at: datetime,
    resolution_summary: str,
    actions_taken: list[str],
    casualties_reported: Optional[int],
    injuries_reported: Optional[int],
    notes: Optional[str],
) -> dict:
    created_at = incident_data.get("created_at")
    claimed_at = incident_data.get("claimed_at")
    attending_at = incident_data.get("attending_at")
    started_response_at = attending_at or claimed_at or created_at

    return {
        "incident_id": incident_id,
        "camera_id": incident_data.get("camera_id"),
        "incident_type": incident_data.get("incident_type"),
        "officer_id": officer_id,
        "verification": incident_data.get("verification"),
        "location": incident_data.get("location"),
        "timeline": {
            "created_at": created_at,
            "claimed_at": claimed_at,
            "attending_at": attending_at,
            "completed_at": completed_at,
            "response_duration_minutes": compute_duration_minutes(started_response_at, completed_at),
            "total_incident_age_minutes": compute_duration_minutes(created_at, completed_at),
        },
        "resolution": {
            "summary": resolution_summary,
            "actions_taken": actions_taken,
            "casualties_reported": casualties_reported,
            "injuries_reported": injuries_reported,
            "notes": notes,
        },
        "generated_at": datetime.utcnow(),
        "generated_by": "system",
    }


def get_allowed_evidence_roots() -> list[Path]:
    configured = os.getenv("EVIDENCE_ALLOWED_ROOTS", "").strip()
    if configured:
        roots: list[Path] = []
        for entry in configured.split(";"):
            value = entry.strip()
            if not value:
                continue
            p = Path(value)
            if not p.is_absolute():
                p = (Path(__file__).resolve().parent / p).resolve()
            else:
                p = p.resolve()
            roots.append(p)
        if roots:
            return roots

    return [
        (Path(__file__).resolve().parent / "artifacts").resolve(),
        (Path(__file__).resolve().parent / "edge_worker" / "artifacts").resolve(),
    ]


def is_subpath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def estimate_eta_minutes(distance_km: float, avg_speed_kmph: float = 35.0) -> float:
    """Estimate ETA in minutes from distance and average response speed."""
    if avg_speed_kmph <= 0:
        return float("inf")
    return (distance_km / avg_speed_kmph) * 60.0


def rank_officers_by_eta(
    db,
    incident_lat: float,
    incident_lon: float,
    radius_km: float,
    limit: int,
) -> list[dict]:
    """
    Find officers within radius and rank by estimated ETA.
    Current ETA uses distance/speed approximation.
    """
    officers_ref = db.collection(OFFICERS_COLLECTION)
    candidates = []

    # Include both available and on_duty officers as candidates
    for status_value in [OfficerStatus.AVAILABLE.value, OfficerStatus.ON_DUTY.value]:
        for doc in officers_ref.where("status", "==", status_value).stream():
            officer_data = doc.to_dict()
            location = officer_data.get("location") or {}
            lat = location.get("latitude")
            lon = location.get("longitude")

            if lat is None or lon is None:
                continue

            distance_km = calculate_distance(incident_lat, incident_lon, float(lat), float(lon))
            if distance_km > radius_km:
                continue

            eta_minutes = estimate_eta_minutes(distance_km)

            candidates.append({
                "id": doc.id,
                "status": officer_data.get("status"),
                "distance_km": round(distance_km, 3),
                "eta_minutes": round(eta_minutes, 2),
                "badge_number": officer_data.get("badge_number"),
                "name": officer_data.get("name"),
            })

    candidates.sort(key=lambda x: (x["eta_minutes"], x["distance_km"]))
    return candidates[:max(0, limit)]


def create_dispatch_alert_records(
    db,
    incident_id: str,
    incident_type: str,
    camera_id: str,
    ranked_officers: list[dict],
) -> list[str]:
    """Persist per-officer dispatch alerts for downstream push/websocket workers."""
    alert_ids: list[str] = []

    for rank, officer in enumerate(ranked_officers, start=1):
        alert_doc = {
            "incident_id": incident_id,
            "incident_type": incident_type,
            "camera_id": camera_id,
            "officer_id": officer["id"],
            "rank": rank,
            "eta_minutes": officer.get("eta_minutes"),
            "distance_km": officer.get("distance_km"),
            "status": "pending_delivery",
            "delivery_channel": "websocket",
            "created_at": datetime.utcnow(),
            "delivered_at": None,
            "acknowledged_at": None,
        }
        ref = db.collection(ALERTS_COLLECTION).add(alert_doc)
        alert_ids.append(ref[1].id)

    return alert_ids


def serialize_firestore_doc(doc) -> dict:
    data = doc.to_dict() or {}
    data["id"] = doc.id
    for k, v in list(data.items()):
        if isinstance(v, datetime):
            data[k] = v.isoformat()
    return data


def fetch_pending_alerts_for_officer(db, officer_id: str) -> list[dict]:
    query = (
        db.collection(ALERTS_COLLECTION)
        .where("officer_id", "==", officer_id)
        .where("status", "==", "pending_delivery")
    )
    docs = list(query.stream())
    alerts = [serialize_firestore_doc(d) for d in docs]
    alerts.sort(key=lambda x: x.get("rank", 9999))
    return alerts


def mark_alerts_delivered(db, alert_ids: list[str]):
    delivered_at = datetime.utcnow()
    for alert_id in alert_ids:
        db.collection(ALERTS_COLLECTION).document(alert_id).update({
            "status": "delivered",
            "delivered_at": delivered_at,
        })


async def push_alerts_to_officer(officer_id: str, alerts: list[dict]) -> int:
    websocket = ws_manager.get(officer_id)
    if not websocket:
        return 0

    sent = 0
    for alert in alerts:
        await websocket.send_text(json.dumps({
            "type": "dispatch_alert",
            "payload": alert,
        }))
        sent += 1
    return sent


async def dispatch_alerts_realtime(db, alert_ids: list[str]) -> int:
    """Push newly created alerts to online officers and mark delivered when sent."""
    delivered = 0
    delivered_ids: list[str] = []

    for alert_id in alert_ids:
        doc = db.collection(ALERTS_COLLECTION).document(alert_id).get()
        if not doc.exists:
            continue

        alert = serialize_firestore_doc(doc)
        officer_id = alert.get("officer_id")
        if not officer_id or not ws_manager.is_connected(officer_id):
            continue

        sent = await push_alerts_to_officer(officer_id, [alert])
        if sent > 0:
            delivered += sent
            delivered_ids.append(alert_id)

    if delivered_ids:
        mark_alerts_delivered(db, delivered_ids)

    return delivered


def trigger_public_safety_advisory(db, incident_id: str, triggered_by: str) -> dict:
    """
    Send one-time SMS advisory to nearby opted-in users after officer assignment.
    Idempotent per incident via `incident.public_advisory.status`.
    """
    incident_ref = db.collection(INCIDENTS_COLLECTION).document(incident_id)

    @firestore.transactional
    def reserve_advisory(transaction):
        doc = incident_ref.get(transaction=transaction)
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Incident not found")

        data = doc.to_dict() or {}
        advisory = data.get("public_advisory") or {}
        status_value = advisory.get("status")

        if status_value in {"processing", "sent", "no_recipients", "sms_unavailable", "failed"}:
            return False, data, status_value

        transaction.update(
            incident_ref,
            {
                "public_advisory": {
                    "status": "processing",
                    "triggered_by": triggered_by,
                    "started_at": datetime.utcnow(),
                }
            },
        )
        return True, data, "processing"

    transaction = db.transaction()
    should_process, incident_data, previous_status = reserve_advisory(transaction)
    if not should_process:
        return {"status": "already_processed", "incident_id": incident_id, "previous_status": previous_status}

    location = incident_data.get("location") or {}
    lat = location.get("latitude")
    lon = location.get("longitude")
    if lat is None or lon is None:
        incident_ref.update(
            {
                "public_advisory": {
                    "status": "failed",
                    "reason": "incident_location_missing",
                    "triggered_by": triggered_by,
                    "finished_at": datetime.utcnow(),
                }
            }
        )
        return {"status": "failed", "reason": "incident_location_missing"}

    radius_km = float(os.getenv("CIVILIAN_ADVISORY_RADIUS_KM", "1.0"))
    recipients = find_nearby_opted_in_users(db, float(lat), float(lon), radius_km)

    if not recipients:
        incident_ref.update(
            {
                "public_advisory": {
                    "status": "no_recipients",
                    "radius_km": radius_km,
                    "recipient_count": 0,
                    "triggered_by": triggered_by,
                    "finished_at": datetime.utcnow(),
                }
            }
        )
        return {"status": "no_recipients", "recipient_count": 0, "radius_km": radius_km}

    incident_type = incident_data.get("incident_type", "incident")
    sms_body = (
        f"DLW Safety Advisory: Suspected {incident_type.replace('_', ' ')} reported nearby. "
        "Avoid the area and follow police instructions."
    )

    sms_result = send_sms_batch([r["phone_number"] for r in recipients], sms_body)

    status_value = "sent" if sms_result["sent_count"] > 0 else "sms_unavailable"
    incident_ref.update(
        {
            "public_advisory": {
                "status": status_value,
                "radius_km": radius_km,
                "recipient_count": len(recipients),
                "sent_count": sms_result["sent_count"],
                "failed_count": sms_result["failed_count"],
                "errors": sms_result.get("errors", [])[:5],
                "triggered_by": triggered_by,
                "finished_at": datetime.utcnow(),
            }
        }
    )

    # Persist advisory alert audit records.
    for r in recipients:
        db.collection(ALERTS_COLLECTION).add(
            {
                "incident_id": incident_id,
                "incident_type": incident_type,
                "user_id": r["id"],
                "phone_number": r["phone_number"],
                "status": "sent" if r["phone_number"] in sms_result["sent_numbers"] else "failed",
                "delivery_channel": "sms",
                "created_at": datetime.utcnow(),
            }
        )

    return {
        "status": status_value,
        "radius_km": radius_km,
        "recipient_count": len(recipients),
        "sent_count": sms_result["sent_count"],
        "failed_count": sms_result["failed_count"],
    }


def find_nearby_opted_in_users(db, incident_lat: float, incident_lon: float, radius_km: float) -> list[dict]:
    users_ref = db.collection(USERS_COLLECTION).where("opted_in", "==", True)
    nearby: list[dict] = []

    for doc in users_ref.stream():
        data = doc.to_dict() or {}
        phone = data.get("phone_number")
        location = data.get("location") or {}
        lat = location.get("latitude")
        lon = location.get("longitude")

        if not phone or lat is None or lon is None:
            continue

        distance_km = calculate_distance(incident_lat, incident_lon, float(lat), float(lon))
        if distance_km <= radius_km:
            nearby.append(
                {
                    "id": doc.id,
                    "phone_number": phone,
                    "distance_km": round(distance_km, 3),
                }
            )

    nearby.sort(key=lambda x: x["distance_km"])
    return nearby


def send_sms_batch(phone_numbers: list[str], body: str) -> dict:
    """Send SMS messages via Twilio when configured; otherwise no-op with diagnostics."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")

    sent_numbers: list[str] = []
    errors: list[str] = []

    if not account_sid or not auth_token or not from_number:
        return {
            "sent_count": 0,
            "failed_count": len(phone_numbers),
            "sent_numbers": sent_numbers,
            "errors": ["twilio_not_configured"],
        }

    try:
        twilio_rest = importlib.import_module("twilio.rest")
        client = twilio_rest.Client(account_sid, auth_token)
    except Exception as exc:
        return {
            "sent_count": 0,
            "failed_count": len(phone_numbers),
            "sent_numbers": sent_numbers,
            "errors": [f"twilio_import_failed:{exc}"],
        }

    for phone in phone_numbers:
        try:
            client.messages.create(body=body, from_=from_number, to=phone)
            sent_numbers.append(phone)
        except Exception as exc:
            errors.append(f"{phone}:{exc}")

    return {
        "sent_count": len(sent_numbers),
        "failed_count": max(0, len(phone_numbers) - len(sent_numbers)),
        "sent_numbers": sent_numbers,
        "errors": errors,
    }
