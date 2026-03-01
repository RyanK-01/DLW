
import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import "./ui/styles.css";
import { AuthProvider } from "./auth/AuthContext";
import { LoginPage } from "./pages/LoginPage";
import { PublicPage } from "./pages/PublicPage";
import { ResponderPage } from "./pages/ResponderPage";
import { IncidentPage } from "./pages/IncidentPage";
import { RequireRole } from "./auth/RequireRole";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/public" element={<PublicPage />} />

          <Route
            path="/responder"
            element={
              <RequireRole allow={["responder", "admin"]}>
                <ResponderPage />
              </RequireRole>
            }
          />

          <Route
            path="/responder/incidents/:id"
            element={
              <RequireRole allow={["responder", "admin"]}>
                <IncidentPage />
              </RequireRole>
            }
          />

          <Route path="/" element={<Navigate to="/public" replace />} />
          <Route path="*" element={<Navigate to="/public" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}