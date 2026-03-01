
import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "./AuthContext";
import type { UserRole } from "../types/user";

export function RequireRole({ allow, children }: { allow: UserRole[]; children: React.ReactNode }) {
  const { user, role, loading } = useAuth();

  if (loading) return <div style={{ padding: 24 }}>Loading…</div>;
  if (!user) return <Navigate to="/login" replace />;
  if (!role || !allow.includes(role)) return <Navigate to="/public" replace />;
  return <>{children}</>;
}