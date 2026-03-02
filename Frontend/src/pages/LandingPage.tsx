import React, { useEffect, useState } from "react";
import { signInWithEmailAndPassword, createUserWithEmailAndPassword } from "firebase/auth";
import { doc, setDoc, serverTimestamp } from "firebase/firestore";
import { auth, db } from "../firebase";
import { useAuth } from "../auth/AuthContext";
import { useNavigate } from "react-router-dom";

type Tab = "login" | "signup";

const RESPONDER_DOMAIN = "@staff.safewatch.sg";

export function LandingPage() {
  const nav = useNavigate();
  const { user, role, loading } = useAuth();
  const [tab, setTab] = useState<Tab>("login");
  // Login state
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPw, setLoginPw] = useState("");
  const [loginErr, setLoginErr] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);

  // Sign up state
  const [name, setName] = useState("");
  const [signupEmail, setSignupEmail] = useState("");
  const [signupPw, setSignupPw] = useState("");
  const [signupPhone, setSignupPhone] = useState("");
  const [signupErr, setSignupErr] = useState("");
  const [signupLoading, setSignupLoading] = useState(false);

  useEffect(() => {
    if (!loading && user && role) {
      nav("/public", { replace: true });
    }
  }, [user, role, loading, nav]);

  async function onLogin(e: React.FormEvent) {
    e.preventDefault();
    setLoginErr("");
    setLoginLoading(true);
    try {
      await signInWithEmailAndPassword(auth, loginEmail.trim(), loginPw);
    } catch (ex: any) {
      setLoginErr(
        ex?.code === "auth/wrong-password" || ex?.code === "auth/invalid-credential"
          ? "Invalid email or password."
          : ex?.code === "auth/user-not-found"
          ? "No account found for this email. Please sign up."
          : ex?.message ?? "Login failed."
      );
    } finally {
      setLoginLoading(false);
    }
  }

  async function onSignup(e: React.FormEvent) {
    e.preventDefault();
    setSignupErr("");
    setSignupLoading(true);
    const email = signupEmail.trim().toLowerCase();
    const assignedRole = email.endsWith(RESPONDER_DOMAIN) ? "responder" : "public";
    // Normalise phone: prepend +65 if no country code
    let normPhone = signupPhone.trim().replace(/\s|-/g, "");
    if (normPhone && !normPhone.startsWith("+")) {
      normPhone = `+65${normPhone.replace(/^0+/, "")}`;
    }
    try {
      const credential = await createUserWithEmailAndPassword(auth, email, signupPw);
      const uid = credential.user.uid;

      await setDoc(doc(db, "users", uid), {
        username: name,
        email,
        role: assignedRole,
        phone: normPhone,
        createdAt: serverTimestamp(),
      });

      nav("/public", { replace: true });
    } catch (ex: any) {
      if (ex?.code === "auth/email-already-in-use") {
        setSignupErr("An account with this email already exists.");
      } else if (ex?.code === "auth/invalid-email") {
        setSignupErr("Please enter a valid email address.");
      } else if (ex?.code === "auth/weak-password") {
        setSignupErr("Password must be at least 6 characters.");
      } else {
        setSignupErr(ex?.message ?? "Registration failed.");
      }
    } finally {
      setSignupLoading(false);
    }
  }

  return (
    <div className="landing-root">
      {/* ── Hero ── */}
      <section className="landing-hero">
        <div className="landing-hero-inner">
          <div className="landing-logo">🛡️</div>
          <h1 className="landing-title">WELCOME TO SAFEWATCH</h1>
          <p className="landing-sub">
            The art of keeping you safe with the constant update of incidents and emergencies.
          </p>

          {/* Feature pills */}
          <div className="landing-pills">
            <span className="landing-pill">📡 Live Incident Feed</span>
            <span className="landing-pill">🔔 Real-Time Advisories</span>
            <span className="landing-pill">🤝 Community Safety</span>
          </div>
        </div>
      </section>

      {/* ── Auth card ── */}
      <div className="landing-card-wrap">
        <div className="landing-card">
          <div className="landing-tabs">
            <button
              className={`landing-tab${tab === "login" ? " active" : ""}`}
              onClick={() => setTab("login")}
              type="button"
            >
              Log In
            </button>
            <button
              className={`landing-tab${tab === "signup" ? " active" : ""}`}
              onClick={() => setTab("signup")}
              type="button"
            >
              Sign Up
            </button>
          </div>

          {/* ── Login form ── */}
          {tab === "login" && (
            <form onSubmit={onLogin} className="col" style={{ marginTop: 20 }}>
              <label className="landing-label">
                Email address
                <input
                  className="input"
                  type="email"
                  placeholder="you@example.com"
                  value={loginEmail}
                  onChange={(e) => setLoginEmail(e.target.value)}
                  required
                  autoComplete="email"
                />
              </label>
              <label className="landing-label">
                Password
                <input
                  className="input"
                  type="password"
                  placeholder="••••••••"
                  value={loginPw}
                  onChange={(e) => setLoginPw(e.target.value)}
                  required
                  autoComplete="current-password"
                />
              </label>
              {loginErr && <div className="landing-err">{loginErr}</div>}
              <button className="button landing-btn" disabled={loginLoading}>
                {loginLoading ? "Signing in…" : "Log In"}
              </button>
              <div className="landing-switch">
                Don't have an account?{" "}
                <button type="button" className="landing-link" onClick={() => setTab("signup")}>
                  Sign Up
                </button>
              </div>
            </form>
          )}

          {/* ── Sign up form ── */}
          {tab === "signup" && (
            <form onSubmit={onSignup} className="col" style={{ marginTop: 20 }}>
              <label className="landing-label">
                Full name
                <input
                  className="input"
                  placeholder="John Doe"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                  autoComplete="name"
                />
              </label>
              <label className="landing-label">
                Email address
                <input
                  className="input"
                  type="email"
                  placeholder="you@example.com"
                  value={signupEmail}
                  onChange={(e) => setSignupEmail(e.target.value)}
                  required
                  autoComplete="email"
                />
                {signupEmail.toLowerCase().endsWith("@staff.safewatch.sg") && (
                  <span className="small" style={{ color: "#2e7d32", marginTop: 4 }}>
                    ✓ Staff email — responder access will be granted.
                  </span>
                )}
              </label>
              <label className="landing-label">
                Mobile number
                <input
                  className="input"
                  type="tel"
                  placeholder="e.g. 91234567"
                  value={signupPhone}
                  onChange={(e) => setSignupPhone(e.target.value)}
                  required
                  autoComplete="tel"
                />
                {signupPhone && !signupPhone.startsWith("+") && (
                  <span className="small" style={{ color: "#0e7490", marginTop: 2 }}>
                    +65 will be added automatically
                  </span>
                )}
              </label>
              <label className="landing-label">
                Password
                <input
                  className="input"
                  type="password"
                  placeholder="Min. 6 characters"
                  value={signupPw}
                  onChange={(e) => setSignupPw(e.target.value)}
                  required
                  minLength={6}
                  autoComplete="new-password"
                />
              </label>
              {signupErr && <div className="landing-err">{signupErr}</div>}
              <button className="button landing-btn" disabled={signupLoading}>
                {signupLoading ? "Creating account…" : "Create Account"}
              </button>
              <div className="landing-switch">
                Already have an account?{" "}
                <button type="button" className="landing-link" onClick={() => setTab("login")}>
                  Log In
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
