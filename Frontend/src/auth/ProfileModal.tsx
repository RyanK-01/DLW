import React, { useEffect, useRef, useState } from "react";
import { doc, getDoc, updateDoc } from "firebase/firestore";
import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { db, auth, storage } from "../firebase";

interface Props {
  onClose: () => void;
}

const SEX_OPTIONS = [
  "Prefer not to say",
  "Male",
  "Female",
  "Non-binary",
  "Gay",
  "Lesbian",
  "Pansexual",
  "Other",
];

export function ProfileModal({ onClose }: Props) {
  const [name, setName]           = useState("");
  const [age, setAge]             = useState("");
  const [sex, setSex]             = useState(SEX_OPTIONS[0]);
  const [address, setAddress]     = useState("");
  const [photoURL, setPhotoURL]   = useState<string | null>(null);
  const [photoFile, setPhotoFile] = useState<File | null>(null);
  const [preview, setPreview]     = useState<string | null>(null);
  const [loading, setLoading]     = useState(true);
  const [saving, setSaving]       = useState(false);
  const [error, setError]         = useState("");
  const [success, setSuccess]     = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load existing profile from Firestore on open
  useEffect(() => {
    const uid = auth.currentUser?.uid;
    if (!uid) { setLoading(false); return; }
    getDoc(doc(db, "users", uid)).then((snap) => {
      if (snap.exists()) {
        const d = snap.data() as any;
        setName(d.username ?? "");
        setAge(d.age != null ? String(d.age) : "");
        setSex(d.sex ?? SEX_OPTIONS[0]);
        setAddress(d.address ?? "");
        setPhotoURL(d.photoURL ?? null);
      }
    }).finally(() => setLoading(false));
  }, []);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setPhotoFile(file);
    setPreview(URL.createObjectURL(file));
  }

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    setError(""); setSuccess(false); setSaving(true);
    const uid = auth.currentUser?.uid;
    if (!uid) { setError("Not logged in."); setSaving(false); return; }
    try {
      let finalPhotoURL = photoURL;

      if (photoFile) {
        const storageRef = ref(storage, `avatars/${uid}`);
        await uploadBytes(storageRef, photoFile);
        finalPhotoURL = await getDownloadURL(storageRef);
      }

      await updateDoc(doc(db, "users", uid), {
        username: name.trim(),
        age: age ? Number(age) : null,
        sex,
        address: address.trim(),
        ...(finalPhotoURL ? { photoURL: finalPhotoURL } : {}),
      });
      setSuccess(true);
      setTimeout(() => { setSuccess(false); onClose(); }, 1000);
    } catch (ex: any) {
      setError(ex?.message ?? "Failed to save.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.45)",
        display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000,
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="card"
        style={{ width: 380, padding: 28, borderRadius: 14, background: "#fff", boxShadow: "0 8px 32px rgba(0,0,0,0.18)" }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <h3 style={{ margin: 0 }}>My Profile</h3>
          <button onClick={onClose} style={{ background: "none", border: "none", fontSize: 20, cursor: "pointer", color: "#888", lineHeight: 1 }}>×</button>
        </div>

        {loading ? (
          <div className="small" style={{ textAlign: "center", padding: 20 }}>Loading…</div>
        ) : (
          <form onSubmit={handleSave} className="col" style={{ gap: 14 }}>

            <label className="landing-label" style={{ fontSize: 13, fontWeight: 500 }}>
              Full name
              <input
                className="input"
                placeholder="John Doe"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </label>

            <label className="landing-label" style={{ fontSize: 13, fontWeight: 500 }}>
              Age
              <input
                className="input"
                type="number"
                placeholder="e.g. 30"
                value={age}
                min={1} max={120}
                onChange={(e) => setAge(e.target.value)}
              />
            </label>

            <label className="landing-label" style={{ fontSize: 13, fontWeight: 500 }}>
              Sex
              <select
                className="input"
                value={sex}
                onChange={(e) => setSex(e.target.value)}
              >
                {SEX_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            </label>

            <label className="landing-label" style={{ fontSize: 13, fontWeight: 500 }}>
              Home address
              <input
                className="input"
                placeholder="123 Main St, Singapore"
                value={address}
                onChange={(e) => setAddress(e.target.value)}
              />
            </label>

            {error && (
              <div style={{ background: "#ffe8ea", border: "1px solid #ffb3bc", borderRadius: 8, padding: "10px 14px", fontSize: 13, color: "#b00020" }}>
                {error}
              </div>
            )}
            {success && (
              <div style={{ background: "#e8fff1", border: "1px solid #bde8cc", borderRadius: 8, padding: "10px 14px", fontSize: 13, color: "#1b5e20" }}>
                ✓ Profile saved.
              </div>
            )}

            <div className="row" style={{ gap: 10, marginTop: 4 }}>
              <button type="button" className="button secondary" style={{ flex: 1 }} onClick={onClose} disabled={saving}>
                Cancel
              </button>
              <button type="submit" className="button" style={{ flex: 1 }} disabled={saving}>
                {saving ? "Saving…" : "Save"}
              </button>
            </div>

          </form>
        )}
      </div>
    </div>
  );
}
