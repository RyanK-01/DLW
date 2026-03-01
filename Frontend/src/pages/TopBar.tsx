
import React from "react";
import { signOut } from "firebase/auth";
import { auth } from "../firebase";
import { useAuth } from "../auth/AuthContext";
import { Link } from "react-router-dom";

export function TopBar() {
  const { user, role } = useAuth();

  return (
    <div className="topbar">
      <div className="row" style={{ gap: 12 }}>
        <svg width="36" height="40" viewBox="0 0 36 40" style={{ flexShrink: 0 }}>
          <path d="M18 2 L6 8 L6 16 Q6 25 18 35 Q30 25 30 16 L30 8 Z" fill="none" stroke="white" strokeWidth="2.5" strokeLinejoin="round"/>
          <path d="M18 2 L6 8 L6 16 Q6 25 18 35 Q18 25 18 16 L18 8 Z" fill="#1E3A5F" />
          <path d="M18 2 L30 8 L30 16 Q30 25 18 35 Q18 25 18 16 L18 8 Z" fill="#A8E6F0" />
        </svg>
        <div className="brand" style={{ fontSize: 20, margin: 0 }}>SafeWatch SG</div>
      </div>
    </div>
  );
}