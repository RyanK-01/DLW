
import React, { useEffect, useState } from "react";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "../firebase";
import { useAuth } from "../auth/AuthContext";
import { useNavigate } from "react-router-dom";
import { TopBar } from "./TopBar";

export function LoginPage() {
  const nav = useNavigate();
  const { user, role, loading } = useAuth();

  const [email, setEmail] = useState("");
  const [pw, setPw] = useState("");
  const [err, setErr] = useState("");

  useEffect(() => {
    if (!loading && user && role) {
      nav(role === "responder" || role === "admin" ? "/responder" : "/public", { replace: true });
    }
  }, [user, role, loading, nav]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr("");
    try {
      await signInWithEmailAndPassword(auth, email, pw);
    } catch (ex: any) {
      setErr(ex?.message ?? "Login failed");
    }
  }

  return (
    <>
      <TopBar />
      <div className="container">
        <div className="card" style={{ maxWidth: 440, margin: "30px auto" }}>
          <h2 style={{ marginTop: 0 }}>Login</h2>
          <div className="small">
            Shared login for public & responders. Role comes from Firestore `users/{`uid`}.role`.
          </div>
          <hr />
          <form onSubmit={onSubmit} className="col">
            <label>
              Email
              <input className="input" value={email} onChange={(e) => setEmail(e.target.value)} />
            </label>
            <label>
              Password
              <input className="input" type="password" value={pw} onChange={(e) => setPw(e.target.value)} />
            </label>
            <button className="button">Sign in</button>
            {err && <div style={{ color: "#b00020" }}>{err}</div>}
          </form>
        </div>
      </div>
    </>
  );
}