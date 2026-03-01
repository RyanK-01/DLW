
//Live incident dashboard
//Shows all flagged incidents
//Lets responder choose which one to inspect

import React, { useEffect, useMemo, useState } from "react";
import { collection, onSnapshot, orderBy, query } from "firebase/firestore";
import { db } from "../firebase";
import type { Incident } from "../types/incident";
import { TopBar } from "./TopBar";
import { Link } from "react-router-dom";

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

  useEffect(() => {
    const q1 = query(collection(db, "incidents"), orderBy("updatedAt", "desc"));
    const unsub = onSnapshot(q1, (snap) => {
      const items: Incident[] = snap.docs.map((d) => ({ id: d.id, ...(d.data() as any) }));
      setIncs(items);
    });
    return () => unsub();
  }, []);

  const filtered = useMemo(() => incs.filter((i) => (i.riskScore ?? 0) >= minRisk), [incs, minRisk]);

  return (
    <>
      <TopBar />
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
                <Link key={i.id} to={`/responder/incidents/${i.id}`} className="card" style={{ padding: 12 }}>
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
              TRIAGE → view CCTV/latest frame → CONFIRM/FALSE → Publish advisory if confirmed.
            </div>
          </div>
        </div>
      </div>
    </>
  );
}