
import React, { useEffect, useMemo, useState } from "react";
import { collection, onSnapshot, query, where } from "firebase/firestore";
import { db } from "../firebase";
import type { Advisory } from "../types/advisory";
import { TopBar } from "./TopBar";

import { MapContainer, TileLayer, Marker, Popup, Circle } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

L.Icon.Default.mergeOptions({ iconRetinaUrl: markerIcon2x, iconUrl: markerIcon, shadowUrl: markerShadow });

function haversineM(lat1: number, lng1: number, lat2: number, lng2: number) {
  const R = 6371000;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

export function PublicPage() {
  const [advs, setAdvs] = useState<Advisory[]>([]);
  const [me, setMe] = useState<{ lat: number; lng: number } | null>(null);
  const [radiusM, setRadiusM] = useState(1500);
  const [filterDate, setFilterDate] = useState<string>("");

  useEffect(() => {
    const q1 = query(collection(db, "advisories"), where("published", "==", true));
    const unsub = onSnapshot(q1, (snap) => {
      const items: Advisory[] = snap.docs.map((d) => ({ id: d.id, ...(d.data() as any) }));
      setAdvs(items);
    });
    return () => unsub();
  }, []);

  useEffect(() => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => setMe({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
      () => setMe(null),
      { enableHighAccuracy: true, timeout: 8000 }
    );
  }, []);

  const center = useMemo<[number, number]>(() => {
    if (me) return [me.lat, me.lng];
    return [1.3521, 103.8198]; // SG fallback
  }, [me]);

  const nearby = useMemo(() => {
    let filtered = advs;
    
    // Filter by date if selected
    if (filterDate) {
      const selectedDate = new Date(filterDate);
      filtered = filtered.filter((a) => {
        const advDate = new Date(a.createdAt);
        return advDate.toDateString() === selectedDate.toDateString();
      });
    }
    
    // Filter by proximity if location available
    if (me) {
      filtered = filtered.filter((a) => haversineM(me.lat, me.lng, a.lat, a.lng) <= radiusM);
    }
    
    return filtered;
  }, [advs, me, radiusM, filterDate]);

  return (
    <>
      <TopBar />
      <div className="container">
        <div className="grid2" style={{ gridTemplateColumns: "1fr 2.5fr" }}>
          <div className="card">
            <h2 style={{ marginTop: 0 }}>Completed Cases</h2>
            <hr />

            <div style={{ display: "flex", flexDirection: "column", gap: 6, maxWidth: 180 }}>
              <label className="small" style={{ fontWeight: 500 }}>Filter by date</label>
              <input 
                className="input" 
                type="date" 
                value={filterDate} 
                onChange={(e) => setFilterDate(e.target.value)} 
              />
            </div>

            <hr />

            <div className="col">
              {nearby.length === 0 && <div className="small">No cases found.</div>}
              {nearby.map((a) => (
                <div key={a.id} className="card" style={{ padding: 12 }}>
                  <div style={{ marginBottom: 6 }}>
                    <b>{a.title}</b>
                  </div>
                  <div className="small">{a.message}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h2 style={{ marginTop: 0 }}>Cases Near Me</h2>
            <div className="mapWrap">
              <MapContainer center={center} zoom={12} style={{ height: "100%", width: "100%" }}>
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution="&copy; OpenStreetMap contributors"
                />

                {me && (
                  <>
                    <Marker position={[me.lat, me.lng]}>
                      <Popup>You are here</Popup>
                    </Marker>
                    <Circle center={[me.lat, me.lng]} radius={radiusM} />
                  </>
                )}

                {nearby.map((a) => (
                  <React.Fragment key={a.id}>
                    <Marker position={[a.lat, a.lng]}>
                      <Popup>
                        <b>{a.title}</b><br />
                        Radius: {a.radiusM}m<br />
                        {a.message}
                      </Popup>
                    </Marker>
                    <Circle center={[a.lat, a.lng]} radius={a.radiusM} />
                  </React.Fragment>
                ))}
              </MapContainer>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}