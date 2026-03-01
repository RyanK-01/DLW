
import React, { useState } from "react";
import { signOut } from "firebase/auth";
import { auth } from "../firebase";
import { useAuth } from "../auth/AuthContext";
import { Link, useNavigate } from "react-router-dom";
import { ProfileModal } from "../auth/ProfileModal";

const box: React.CSSProperties = {
  padding: "7px 14px",
  border: "1px solid #ddd",
  borderRadius: 8,
  fontSize: 14,
  fontWeight: 500,
  background: "white",
  color: "#111",
  cursor: "pointer",
  lineHeight: 1,
  display: "inline-flex",
  alignItems: "center",
};

export function TopBar() {
  const { user, role } = useAuth();
  const nav = useNavigate();
  const [accessErr, setAccessErr] = useState(false);
  const [showProfile, setShowProfile] = useState(false);

  const roleLabel = role
    ? role.charAt(0).toUpperCase() + role.slice(1)
    : "Guest";

  function handleResponder() {
    if (role === "responder" || role === "admin") {
      setAccessErr(false);
      nav("/responder");
    } else {
      setAccessErr(true);
      setTimeout(() => setAccessErr(false), 3500);
    }
  }

  return (
    <>
      <div className="topbar">
        <div className="row" style={{ gap: 8 }}>
          <div className="brand">SafeWatch</div>
          <button
            style={box}
            onClick={() => user && setShowProfile(true)}
            title="Edit profile"
          >
            {roleLabel}
          </button>
        </div>

        <div className="row" style={{ gap: 8 }}>
          <Link style={box} to="/public">Public</Link>
          {user && (
            <button style={box} onClick={handleResponder}>
              Responder
            </button>
          )}
          {!user ? (
            <Link style={box} to="/login">Login</Link>
          ) : (
            <button style={box} onClick={() => signOut(auth)}>Logout</button>
          )}
        </div>
      </div>

      {showProfile && <ProfileModal onClose={() => setShowProfile(false)} />}

      {accessErr && (
        <div style={{
          position: "fixed",
          bottom: 24,
          left: "50%",
          transform: "translateX(-50%)",
          background: "#b00020",
          color: "white",
          padding: "12px 24px",
          borderRadius: 10,
          fontSize: 14,
          fontWeight: 500,
          boxShadow: "0 4px 16px rgba(0,0,0,0.18)",
          zIndex: 1000,
          whiteSpace: "nowrap",
        }}>
          You do not have access to the responder feature.
        </div>
      )}
    </>
  );
}