
import React from "react";
import { signOut } from "firebase/auth";
import { auth } from "../firebase";
import { useAuth } from "../auth/AuthContext";
import { Link } from "react-router-dom";

export function TopBar() {
  const { user, role } = useAuth();

  return (
    <div className="topbar">
      <div className="row" style={{ gap: 10 }}>
        <div className="brand">SafeWatch</div>
        <span className="badge">{role ?? "guest"}</span>
      </div>

      <div className="row">
        <Link className="badge" to="/public">Public</Link>
        <Link className="badge" to="/responder">Responder</Link>
        {!user ? (
          <Link className="button secondary" to="/login">Login</Link>
        ) : (
          <button className="button secondary" onClick={() => signOut(auth)}>Logout</button>
        )}
      </div>
    </div>
  );
}