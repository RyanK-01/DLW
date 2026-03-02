from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class IncidentType(str, Enum):
    FIGHT = "fight"
    ARMED_THREAT = "armed_threat"


class IncidentStatus(str, Enum):
    DETECTED = "detected"
    NOTIFIED = "notified"
    CLAIMED = "claimed"
    ATTENDING = "attending"
    COMPLETED = "completed"
    RESOLVED = "resolved"


class CameraOperationalStatus(str, Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class CameraIncidentState(str, Enum):
    IDLE = "idle"
    MONITORING = "monitoring"
    FIGHT_INCIDENT_ACTIVE = "fight_incident_active"
    COOLDOWN = "cooldown"


class Location(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    geohash: Optional[str] = None


class BoundingBox(BaseModel):
    """Normalized or pixel bbox coordinates from detector output."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(..., ge=0.0, le=1.0)


class PersonFrameStats(BaseModel):
    """Per-frame summary emitted by the YOLOv8 detector for a camera."""
    timestamp: datetime
    people_count: int = Field(..., ge=0)
    person_bboxes: list[BoundingBox] = Field(default_factory=list)
    min_bbox_distance_px: Optional[float] = Field(None, ge=0)
    overlap_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    motion_score: Optional[float] = Field(None, ge=0.0)


class ActionInferenceWindow(BaseModel):
    """Rolling temporal window used for action/fight classification."""
    start_ts: datetime
    end_ts: datetime
    duration_seconds: float = Field(..., ge=0.5, le=10.0)
    sampled_fps: float = Field(..., ge=1.0, le=30.0)
    frame_count: int = Field(..., ge=1)
    source_clip_uri: Optional[str] = None


class FightInferenceResult(BaseModel):
    """Output from action model over a rolling clip window."""
    model_name: str
    model_version: Optional[str] = None
    raw_fight_probability: float = Field(..., ge=0.0, le=1.0)
    smoothed_fight_probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(0.3, ge=0.0, le=1.0)
    triggered: bool = False
    inferred_at: datetime


class EdgeFightPipelineConfig(BaseModel):
    """
    Camera-level edge pipeline configuration.

    Gate action inference unless at least 2 persons and either proximity or high motion.
    """
    input_fps_min: float = Field(10.0, ge=1.0, le=60.0)
    input_fps_max: float = Field(15.0, ge=1.0, le=60.0)
    min_people_for_fight_inference: int = Field(2, ge=2)
    proximity_distance_px_threshold: float = Field(80.0, ge=0.0)
    overlap_ratio_threshold: float = Field(0.1, ge=0.0, le=1.0)
    motion_score_threshold: float = Field(0.25, ge=0.0)
    window_seconds: float = Field(4.0, ge=0.5, le=10.0)
    action_inference_interval_seconds: float = Field(1.0, ge=0.5, le=10.0)
    fight_probability_threshold: float = Field(0.3, ge=0.0, le=1.0)
    smoothing_alpha: float = Field(0.6, gt=0.0, le=1.0)
    cooldown_seconds: int = Field(30, ge=0)
    keyframe_count: int = Field(3, ge=1, le=10)
    incident_clip_seconds: int = Field(10, ge=1, le=30)


class CameraRuntimeState(BaseModel):
    """Mutable per-camera runtime state for 24/7 stream monitoring."""
    operational_status: CameraOperationalStatus = CameraOperationalStatus.ONLINE
    incident_state: CameraIncidentState = CameraIncidentState.MONITORING
    latest_people_count: int = Field(0, ge=0)
    latest_min_bbox_distance_px: Optional[float] = Field(None, ge=0.0)
    latest_overlap_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    latest_motion_score: Optional[float] = Field(None, ge=0.0)
    rolling_buffer_seconds: float = Field(0.0, ge=0.0)
    latest_smoothed_fight_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    last_inference_at: Optional[datetime] = None
    fight_incident_active: bool = False
    cooldown_until: Optional[datetime] = None
    last_heartbeat_at: Optional[datetime] = None


class IncidentEvidence(BaseModel):
    """Evidence package captured when incident is triggered."""
    keyframe_uris: list[str] = Field(default_factory=list)
    clip_uri: Optional[str] = None
    clip_duration_seconds: Optional[int] = Field(None, ge=1, le=30)
    generated_at: datetime


class IncidentAlert(BaseModel):
    """Alert received from edge ML processors"""
    incident_type: IncidentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    camera_id: str
    timestamp: datetime
    location: Optional[Location] = None  # Optional if camera location lookup needed
    pipeline_stage: str = "edge"
    frame_stats: Optional[PersonFrameStats] = None
    inference_window: Optional[ActionInferenceWindow] = None
    fight_inference: Optional[FightInferenceResult] = None
    evidence: Optional[IncidentEvidence] = None


class Incident(BaseModel):
    """Full incident record in Firestore"""
    id: Optional[str] = None
    incident_type: IncidentType
    confidence: float
    camera_id: str
    location: Location
    status: IncidentStatus = IncidentStatus.DETECTED
    created_at: datetime
    notified_officers: list[str] = Field(default_factory=list)  # Officer IDs notified
    claimed_by: Optional[str] = None  # Officer ID who claimed
    claimed_at: Optional[datetime] = None
    attending_by: Optional[str] = None
    attending_at: Optional[datetime] = None
    verification: Optional[dict] = None
    report_id: Optional[str] = None
    resolved_at: Optional[datetime] = None
    source: str = "edge"
    evidence: Optional[IncidentEvidence] = None
    frame_stats: Optional[PersonFrameStats] = None
    inference_window: Optional[ActionInferenceWindow] = None
    fight_inference: Optional[FightInferenceResult] = None


class OfficerStatus(str, Enum):
    AVAILABLE = "available"
    ON_DUTY = "on_duty"
    RESPONDING = "responding"
    OFFLINE = "offline"


class Officer(BaseModel):
    """Officer record in Firestore"""
    id: str
    name: str
    badge_number: str
    location: Location
    status: OfficerStatus = OfficerStatus.AVAILABLE
    current_incident: Optional[str] = None  # Incident ID if responding
    last_updated: datetime


class Camera(BaseModel):
    """CCTV camera record in Firestore"""
    id: str
    location: Location
    name: Optional[str] = None
    stream_url: Optional[str] = None  # RTSP/HLS feed URL
    is_active: bool = True
    edge_node_id: Optional[str] = None
    timezone: Optional[str] = None
    pipeline_config: EdgeFightPipelineConfig = Field(default_factory=EdgeFightPipelineConfig)
    runtime_state: CameraRuntimeState = Field(default_factory=CameraRuntimeState)
    yolo_model_name: str = "yolov8"
    yolo_model_version: Optional[str] = None
    action_model_name: Optional[str] = None
    action_model_version: Optional[str] = None
    last_seen_at: Optional[datetime] = None


class User(BaseModel):
    """Public user for SMS advisories"""
    id: str
    phone_number: str = Field(..., pattern=r"^\+[1-9]\d{7,14}$")  # E.164 format
    location: Optional[Location] = None  # Last known location
    opted_in: bool = True  # SMS consent
    created_at: datetime


class OfficerLocationUpdate(BaseModel):
    """Request body for officer location updates"""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class IncidentClaimRequest(BaseModel):
    """Request body for claiming an incident"""
    officer_id: str


class IncidentVerifyRequest(BaseModel):
    """Officer verification decision after reviewing stream/clip evidence."""
    officer_id: str
    is_true_positive: bool
    notes: Optional[str] = None


class IncidentAttendRequest(BaseModel):
    """Officer confirms they are attending/responding to the incident."""
    officer_id: str
    notes: Optional[str] = None


class IncidentCompleteRequest(BaseModel):
    """Officer marks incident completed and includes closure details."""
    officer_id: str
    resolution_summary: str
    actions_taken: list[str] = Field(default_factory=list)
    casualties_reported: Optional[int] = Field(None, ge=0)
    injuries_reported: Optional[int] = Field(None, ge=0)
    notes: Optional[str] = None


class UserRegistration(BaseModel):
    """User signup request"""
    phone_number: str = Field(..., pattern=r"^\+[1-9]\d{7,14}$")  # E.164 format
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)


class CameraHeartbeat(BaseModel):
    """Periodic edge heartbeat for camera runtime + load monitoring."""
    camera_id: str
    timestamp: datetime
    input_fps: float = Field(..., ge=0.0, le=60.0)
    processed_fps: float = Field(..., ge=0.0, le=60.0)
    dropped_frames: int = Field(0, ge=0)
    latest_people_count: int = Field(0, ge=0)
    latest_motion_score: Optional[float] = Field(None, ge=0.0)
