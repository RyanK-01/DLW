import React, { useEffect, useMemo, useState } from "react";
import {
  collection, onSnapshot, orderBy, query, where,
} from "firebase/firestore";
import { db } from "../firebase";
import type { Advisory } from "../types/advisory";
import type { Incident } from "../types/incident";
import { TopBar } from "./TopBar";
import { useAuth } from "../auth/AuthContext";
import { Link } from "react-router-dom";

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

function chip(status: string) {
  const base: React.CSSProperties = {
    padding: "4px 10px", border: "1px solid #ddd", borderRadius: 999, fontSize: 12,
  };
  if (status === "CONFIRMED") return <span style={{ ...base, background: "#e8fff1", borderColor: "#bde8cc" }}>CONFIRMED</span>;
  if (status === "FALSE_ALARM") return <span style={{ ...base, background: "#ffe8ea", borderColor: "#ffb3bc" }}>FALSE</span>;
  if (status === "TRIAGED") return <span style={{ ...base, background: "#fff4e5", borderColor: "#ffd8a8" }}>TRIAGED</span>;
  return <span style={base}>{status}</span>;
}

export function DashboardPage() {
  const { role } = useAuth();
  const isResponder = role === "responder";

  // ── Shared: advisories & location ──
  const [advs, setAdvs] = useState<Advisory[]>([]);
  const [me, setMe] = useState<{ lat: number; lng: number } | null>(null);
  const [radiusM, setRadiusM] = useState(1500);

  useEffect(() => {
    const unsub = onSnapshot(
      query(collection(db, "advisories"), where("published", "==", true)),
      (snap) => setAdvs(snap.docs.map((d) => ({ id: d.id, ...(d.data() as any) })))
    );
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

  const center = useMemo<[number, number]>(
    () => (me ? [me.lat, me.lng] : [1.3521, 103.8198]),
    [me]
  );

  const nearby = useMemo(
    () => (me ? advs.filter((a) => haversineM(me.lat, me.lng, a.lat, a.lng) <= radiusM) : advs),
    [advs, me, radiusM]
  );

  // ── Responder-only: incident queue ──
  const [incs, setIncs] = useState<Incident[]>([]);
  const [minRisk, setMinRisk] = useState(0);

  useEffect(() => {
    if (!isResponder) return;
    const unsub = onSnapshot(
      query(collection(db, "incidents"), orderBy("updatedAt", "desc")),
      (snap) => setIncs(snap.docs.map((d) => ({ id: d.id, ...(d.data() as any) })))
    );
    return () => unsub();
  }, [isResponder]);

  const filteredIncs = useMemo(
    () => incs.filter((i) => (i.riskScore ?? 0) >= minRisk),
    [incs, minRisk]
  );

  return (
    <>
      <TopBar />
      <div className="container">

        {/* ── Responder: Incident Queue ── */}
        {isResponder && (
          <div className="grid2" style={{ marginBottom: 24 }}>
            <div className="card">
              <h2 style={{ marginTop: 0 }}>Responder Queue</h2>
              <div className="small">Live updates via Firestore listener.</div>
              <hr />
              <label className="small">Minimum risk score</label>
              <input
                className="input"
                type="number"
                value={minRisk}
                onChange={(e) => setMinRisk(Number(e.target.value))}
              />
              <hr />
              <div className="col">
                {filteredIncs.length === 0 && <div className="small">No incidents.</div>}
                {filteredIncs.map((i) => (
                  <Link
                    key={i.id}
                    to={`/incidents/${i.id}`}
                    className="card"
                    style={{ padding: 12 }}
                  >
                    <div className="row" style={{ justifyContent: "space-between" }}>
                      <b>Incident #{i.id.slice(0, 6)}</b>
                      {chip(i.status)}
                    </div>
                    <div className="row" style={{ justifyContent: "space-between", marginTop: 8 }}>
                      <span className="badge">Risk {Number(i.riskScore ?? 0).toFixed(1)}</span>
                      <span className="small">{i.category}</span>
                    </div>
                    <div className="small" style={{ marginTop: 8, opacity: 0.7 }}>
                      {i.lat.toFixed(5)}, {i.lng.toFixed(5)}
                    </div>
                  </Link>
                ))}
              </div>
            </div>

            <div className="card">
              <h2 style={{ marginTop: 0 }}>Workflow</h2>
              <div className="small">
                TRIAGE → view CCTV / latest frame → CONFIRM / FALSE → Publish advisory if confirmed.
              </div>
            </div>
          </div>
        )}

        {/* ── Shared: Advisory Map ── */}
        <div className="grid2">
          <div className="card">
            <h2 style={{ marginTop: 0 }}>Public Advisories</h2>
            <div className="small">Only confirmed &amp; published advisories appear here.</div>
            <hr />

            <div className="row" style={{ justifyContent: "space-between" }}>
              <div className="col" style={{ flex: 1 }}>
                <label className="small">Nearby filter radius (m)</label>
                <input
                  className="input"
                  type="number"
                  value={radiusM}
                  onChange={(e) => setRadiusM(Number(e.target.value))}
                />
              </div>
              <div className="col" style={{ flex: 1 }}>
                <div className="small">My location</div>
                <div className="badge">
                  {me ? `${me.lat.toFixed(5)}, ${me.lng.toFixed(5)}` : "Not shared"}
                </div>
              </div>
            </div>

            <hr />

            <div className="col">
              {nearby.length === 0 && <div className="small">No advisories near you.</div>}
              {nearby.map((a) => (
                <div key={a.id} className="card" style={{ padding: 12 }}>
                  <div className="row" style={{ justifyContent: "space-between" }}>
                    <b>{a.title}</b>
                    <span className="badge">{a.radiusM}m</span>
                  </div>
                  <div className="small" style={{ marginTop: 6 }}>{a.message}</div>
                  <div className="small" style={{ marginTop: 6, opacity: 0.65 }}>
                    Incident: {a.incidentId}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h2 style={{ marginTop: 0 }}>Map (OpenStreetMap)</h2>
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
