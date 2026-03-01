
import type { Timestamp } from "firebase/firestore";

export interface Advisory {
  id: string;
  incidentId: string;
  title: string;
  message: string;
  lat: number;
  lng: number;
  radiusM: number;
  published: boolean;
  createdBy: string;
  createdAt?: Timestamp;
}