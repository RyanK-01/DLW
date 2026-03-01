import React, { useEffect, useState } from "react";
import { signInWithEmailAndPassword, createUserWithEmailAndPassword } from "firebase/auth";
import { doc, setDoc, collection, query, where, getDocs, serverTimestamp } from "firebase/firestore";
import { auth, db } from "../firebase";
import { useAuth } from "../auth/AuthContext";
import { useNavigate } from "react-router-dom";

type Tab = "login" | "signup";

const REGIONS = [
  { label: "🇲🇾 +60", code: "60" },
  { label: "🇸🇬 +65", code: "65" },
];

function buildPhone(code: string, digits: string): string {
  return `+${code}${digits.replace(/\D/g, "")}`;
}

function phoneToEmail(fullPhone: string): string {
  return `${fullPhone.replace(/[^0-9]/g, "")}@safewatch.sg`;
}

export function LandingPage() {
  const nav = useNavigate();
  const { user, role, loading } = useAuth();
  const [tab, setTab] = useState<Tab>("login");

  // Login state
  const [loginRegion, setLoginRegion] = useState("60");
  const [loginDigits, setLoginDigits] = useState("");
  const [loginPw, setLoginPw] = useState("");
  const [loginErr, setLoginErr] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);

  // Sign up state
  const [name, setName] = useState("");
  const [signupRegion, setSignupRegion] = useState("60");
  const [signupDigits, setSignupDigits] = useState("");
  const [signupEmail, setSignupEmail] = useState("");
  const [signupPw, setSignupPw] = useState("");
  const [signupErr, setSignupErr] = useState("");
  const [signupLoading, setSignupLoading] = useState(false);

  useEffect(() => {
    if (!loading && user && role) {
      nav(role === "responder" || role === "admin" ? "/responder" : "/public", { replace: true });
    }
  }, [user, role, loading, nav]);

  async function onLogin(e: React.FormEvent) {
    e.preventDefault();
    setLoginErr("");
    setLoginLoading(true);
    const fullPhone = buildPhone(loginRegion, loginDigits);
    const email = phoneToEmail(fullPhone);
    try {
      // Step 1: Check responder collection first (phone + password match)
      const respSnap = await getDocs(
        query(
          collection(db, "responder"),
          where("phone", "==", fullPhone),
          where("password", "==", loginPw)
        )
      );

      if (!respSnap.empty) {
        // Responder matched — create Firebase Auth account on first login, then sign in
        try {
          const credential = await createUserWithEmailAndPassword(auth, email, loginPw);
          await setDoc(doc(db, "users", credential.user.uid), {
            username: respSnap.docs[0].data().name ?? fullPhone,
            phone: fullPhone,
            role: "responder",
            createdAt: serverTimestamp(),
          });
        } catch (createEx: any) {
          if (createEx?.code === "auth/email-already-in-use") {
            // Already created on a previous login — just sign in
            await signInWithEmailAndPassword(auth, email, loginPw);
          } else {
            throw createEx;
          }
        }
        return;
      }

      // Step 2: Not a responder — check users collection for public user
      const userSnap = await getDocs(
        query(collection(db, "users"), where("phone", "==", fullPhone))
      );

      if (userSnap.empty) {
        setLoginErr("Invalid phone number or password.");
        return;
      }

      // Public user found — sign in with Firebase Auth
      await signInWithEmailAndPassword(auth, email, loginPw);

    } catch (ex: any) {
      setLoginErr(ex?.code ? `[${ex.code}] ${ex.message}` : "Invalid phone number or password.");
    } finally {
      setLoginLoading(false);
    }
  }

  async function onSignup(e: React.FormEvent) {
    e.preventDefault();
    setSignupErr("");
    setSignupLoading(true);
    const fullPhone = buildPhone(signupRegion, signupDigits);
    const email = phoneToEmail(fullPhone);
    try {
      const credential = await createUserWithEmailAndPassword(auth, email, signupPw);
      const uid = credential.user.uid;

      // Check if phone matches a responder entry to assign role
      const respSnap = await getDocs(
        query(collection(db, "responder"), where("phone", "==", fullPhone))
      );
      const assignedRole = respSnap.empty ? "public" : "responder";

      await setDoc(doc(db, "users", uid), {
        username: name,
        phone: fullPhone,
        email: signupEmail,
        role: assignedRole,
        createdAt: serverTimestamp(),
      });

      nav("/public", { replace: true });
    } catch (ex: any) {
      if (ex?.code === "auth/email-already-in-use") {
        setSignupErr("An account with this phone number already exists.");
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
      <div className="landing-hero">
        <div className="landing-hero-inner">
          <div className="landing-logo">🛡️</div>
          <h1 className="landing-title">Welcome to SafeWatch SG</h1>
          <p className="landing-sub">
            AI-powered incident detection &amp; real-time public safety advisories.
          </p>
        </div>
      </div>

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
                Phone number
                <div className="phone-row">
                  <select
                    className="input phone-region"
                    value={loginRegion}
                    onChange={(e) => setLoginRegion(e.target.value)}
                  >
                    {REGIONS.map((r) => (
                      <option key={r.code} value={r.code}>{r.label}</option>
                    ))}
                  </select>
                  <input
                    className="input phone-digits"
                    placeholder="12345678"
                    value={loginDigits}
                    onChange={(e) => setLoginDigits(e.target.value)}
                    required
                    maxLength={9}
                    autoComplete="tel-national"
                  />
                </div>
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
                Phone number
                <div className="phone-row">
                  <select
                    className="input phone-region"
                    value={signupRegion}
                    onChange={(e) => setSignupRegion(e.target.value)}
                  >
                    {REGIONS.map((r) => (
                      <option key={r.code} value={r.code}>{r.label}</option>
                    ))}
                  </select>
                  <input
                    className="input phone-digits"
                    placeholder="12345678"
                    value={signupDigits}
                    onChange={(e) => setSignupDigits(e.target.value)}
                    required
                    maxLength={9}
                    autoComplete="tel-national"
                  />
                </div>
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
              </label>
              <label className="landing-label">
                Password
                <input
                  className="input"
                  type="password"
                  placeholder="••••••••"
                  value={signupPw}
                  onChange={(e) => setSignupPw(e.target.value)}
                  required
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
