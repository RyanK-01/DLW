
//Live incident dashboard
//Shows all flagged incidents
//Lets responder choose which one to inspect

import React, { useEffect, useMemo, useState } from "react";
import { collection, onSnapshot, orderBy, query } from "firebase/firestore";
import { db } from "../firebase";
import type { Incident } from "../types/incident";
import { TopBar } from "./TopBar";
import { Link } from "react-router-dom";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000";

const ACTIVE_STATUSES = new Set(["NEW", "TRIAGED", "CONFIRMED"]);

type PingState = "idle" | "sending" | "sent" | "error";

function chip(status: string) {
  const style: React.CSSProperties = { padding: "4px 10px", border: "1px solid #ddd", borderRadius: 999, fontSize: 12 };
  if (status === "CONFIRMED") return <span style={{ ...style, background: "#e8fff1", borderColor: "#bde8cc" }}>CONFIRMED</span>;
  if (status === "FALSE_ALARM") return <span style={{ ...style, background: "#ffe8ea", borderColor: "#ffb3bc" }}>FALSE</span>;
  if (status === "TRIAGED") return <span style={{ ...style, background: "#fff4e5", borderColor: "#ffd8a8" }}>TRIAGED</span>;
  return <span style={style}>{status}</span>;
}

export function ResponderPage() {
  const [incs, setIncs] = useState<Incident[]>([]);
  const [minRisk, setMinRisk] = useState(0);
  const [pingStates, setPingStates] = useState<Record<string, PingState>>({});

  useEffect(() => {
    const q1 = query(collection(db, "incidents"), orderBy("updatedAt", "desc"));
    const unsub = onSnapshot(q1, (snap) => {
      const items: Incident[] = snap.docs.map((d) => ({ id: d.id, ...(d.data() as any) }));
      setIncs(items);
    });
    return () => unsub();
  }, []);

  const filtered = useMemo(() => incs.filter((i) => (i.riskScore ?? 0) >= minRisk), [incs, minRisk]);

  async function handlePing(e: React.MouseEvent, incidentId: string) {
    e.preventDefault();
    e.stopPropagation();

    setPingStates((prev) => ({ ...prev, [incidentId]: "sending" }));
    try {
      const res = await fetch(`${BACKEND_URL}/api/incidents/${incidentId}/ping`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setPingStates((prev) => ({ ...prev, [incidentId]: "sent" }));
      // Reset back to idle after 3 s
      setTimeout(() => setPingStates((prev) => ({ ...prev, [incidentId]: "idle" })), 3000);
    } catch {
      setPingStates((prev) => ({ ...prev, [incidentId]: "error" }));
      setTimeout(() => setPingStates((prev) => ({ ...prev, [incidentId]: "idle" })), 3000);
    }
  }

  function PingButton({ incident }: { incident: Incident }) {
    if (!ACTIVE_STATUSES.has(incident.status)) return null;
    const state = pingStates[incident.id] ?? "idle";
    return (
      <button
        className={`ping-btn${state === "sent" ? " ping-btn--sent" : ""}${state === "error" ? " ping-btn--error" : ""}`}
        disabled={state === "sending"}
        onClick={(e) => handlePing(e, incident.id)}
      >
        {state === "idle" && "🔔 Ping"}
        {state === "sending" && "Sending…"}
        {state === "sent" && "✓ Sent"}
        {state === "error" && "✗ Failed"}
      </button>
    );
  }

  return (
    <>
      <TopBar />
      <div style={{ minHeight: "calc(100vh - 57px)", background: "linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0e7490 100%)" }}>
      <div className="container">
        <div className="grid2">
          <div className="card">
            <h2 style={{ marginTop: 0 }}>Responder Queue</h2>
            <div className="small">Live updates via Firestore listener.</div>
            <hr />
            <label className="small">Minimum risk score</label>
            <input className="input" type="number" value={minRisk} onChange={(e) => setMinRisk(Number(e.target.value))} />
            <hr />

            <div className="col">
              {filtered.length === 0 && <div className="small">No incidents.</div>}

              {filtered.map((i) => (
                <Link key={i.id} to={`/incidents/${i.id}`} className="card" style={{ padding: 12 }}>
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
                  <div style={{ marginTop: 10 }}>
                    <PingButton incident={i} />
                  </div>
                </Link>
              ))}
            </div>
          </div>

          <div className="card">
            <h2 style={{ marginTop: 0 }}>Actions</h2>
            <div className="small">
              TRIAGE → view CCTV/latest frame → CONFIRM/FALSE → Publish advisory if confirmed.
            </div>
            <hr />
            <Link to="/incidents/new" className="button" style={{ display: "inline-block", textAlign: "center" }}>
              + Generate Report
            </Link>
          </div>
        </div>
      </div>
      </div>
    </>
  );
}