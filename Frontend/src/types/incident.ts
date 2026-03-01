
import type { Timestamp } from "firebase/firestore";

export type IncidentStatus = "NEW" | "TRIAGED" | "CONFIRMED" | "FALSE_ALARM" | "CLOSED";

export interface Incident {
  id: string;
  status: IncidentStatus;
  riskScore: number;
  category: string; // e.g. "suspected_violence"
  lat: number;
  lng: number;
  createdAt: Timestamp;
  updatedAt: Timestamp;
  notes?: string;

  // optional for CCTV / live view:
  latestFrameJpeg?: string; // base64 dataURL OR storage URL
  live?: boolean;
}