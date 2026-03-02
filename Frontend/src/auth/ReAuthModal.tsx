import React, { useState } from "react";
import { EmailAuthProvider, reauthenticateWithCredential } from "firebase/auth";
import { auth } from "../firebase";
import { useAuth } from "./AuthContext";
import { useNavigate } from "react-router-dom";

interface ReAuthModalProps {
  onClose: () => void;
}

export function ReAuthModal({ onClose }: ReAuthModalProps) {
  const { role } = useAuth();
  const nav = useNavigate();
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    const currentUser = auth.currentUser;
    if (!currentUser || !currentUser.email) {
      setError("No active session. Please log in again.");
      setLoading(false);
      return;
    }

    try {
      const credential = EmailAuthProvider.credential(currentUser.email, password);
      await reauthenticateWithCredential(currentUser, credential);

      // Auth passed — now check role
      if (role === "responder") {
        onClose();
        nav("/responder");
      } else {
        setError("You do not have access to the responder feature.");
      }
    } catch (ex: any) {
      if (
        ex?.code === "auth/wrong-password" ||
        ex?.code === "auth/invalid-credential"
      ) {
        setError("Incorrect password.");
      } else {
        setError("Authentication failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.45)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="card"
        style={{
          width: 340,
          padding: 28,
          borderRadius: 12,
          background: "#fff",
          boxShadow: "0 8px 32px rgba(0,0,0,0.18)",
        }}
      >
        <h3 style={{ marginTop: 0, marginBottom: 6 }}>Responder Access</h3>
        <div className="small" style={{ marginBottom: 20, opacity: 0.7 }}>
          Enter your password to verify your identity.
        </div>

        <form onSubmit={handleSubmit} className="col" style={{ gap: 14 }}>
          <input
            className="input"
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoFocus
            required
          />

          {error && (
            <div
              style={{
                background: "#ffe8ea",
                border: "1px solid #ffb3bc",
                borderRadius: 8,
                padding: "10px 14px",
                fontSize: 13,
                color: "#b00020",
              }}
            >
              {error}
            </div>
          )}

          <div className="row" style={{ gap: 10, marginTop: 4 }}>
            <button
              type="button"
              className="button secondary"
              style={{ flex: 1 }}
              onClick={onClose}
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="button"
              style={{ flex: 1 }}
              disabled={loading}
            >
              {loading ? "Verifying…" : "Verify"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
