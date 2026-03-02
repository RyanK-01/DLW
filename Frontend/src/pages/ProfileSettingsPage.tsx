import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { doc, updateDoc, getDoc } from "firebase/firestore";
import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { db, storage } from "../firebase";

export function ProfileSettingsPage() {
  const { user, role } = useAuth();
  const nav = useNavigate();
  const [name, setName] = useState("");
  const [photoURL, setPhotoURL] = useState("");
  const [phone, setPhone] = useState("");
  const [uploading, setUploading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!user) return;
    const loadUserData = async () => {
      const userDoc = await getDoc(doc(db, "users", user.uid));
      if (userDoc.exists()) {
        const data = userDoc.data();
        setName(data.username || "");
        setPhotoURL(data.photoURL || "");
        setPhone(data.phone || "");
      }
    };
    loadUserData();
  }, [user]);

  const handlePhotoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !user) return;

    setUploading(true);
    try {
      const storageRef = ref(storage, `profiles/${user.uid}/${Date.now()}_${file.name}`);
      await uploadBytes(storageRef, file);
      const url = await getDownloadURL(storageRef);
      setPhotoURL(url);
      setMessage("Photo uploaded! Click Save to apply changes.");
    } catch (err: any) {
      setMessage("Failed to upload photo: " + err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleSave = async () => {
    if (!user) return;
    setSaving(true);
    try {
      // Normalise phone: add +65 if it's a bare SG number
      let normPhone = phone.trim().replace(/\s|-/g, "");
      if (normPhone && !normPhone.startsWith("+")) {
        normPhone = `+65${normPhone.replace(/^0+/, "")}`;
      }
      await updateDoc(doc(db, "users", user.uid), {
        username: name,
        photoURL: photoURL,
        ...(role === "responder" ? { phone: normPhone } : {}),
      });
      setMessage("Profile updated successfully!");
      setTimeout(() => nav("/public"), 1500);
    } catch (err: any) {
      setMessage("Failed to save: " + err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#fafafa", padding: "40px 20px" }}>
      <div className="container" style={{ maxWidth: 600 }}>
        <div className="card">
          <h1 style={{ marginTop: 0 }}>Profile Settings</h1>
          
          <div className="col" style={{ gap: 20 }}>
            <div className="col">
              <label className="small" style={{ fontWeight: 600 }}>Profile Picture</label>
              <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <div
                  style={{
                    width: 80,
                    height: 80,
                    borderRadius: "50%",
                    background: photoURL ? `url(${photoURL})` : "#ddd",
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    border: "2px solid #e6e6e6",
                  }}
                />
                <input
                  type="file"
                  accept="image/*"
                  onChange={handlePhotoUpload}
                  disabled={uploading}
                  style={{ fontSize: 13 }}
                />
              </div>
              {uploading && <div className="small">Uploading...</div>}
            </div>

            <div className="col">
              <label className="small" style={{ fontWeight: 600 }}>Name</label>
              <input
                className="input"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your name"
              />
            </div>

            {role === "responder" && (
              <div className="col">
                <label className="small" style={{ fontWeight: 600 }}>
                  Mobile Number
                  <span style={{ fontWeight: 400, color: "#888", marginLeft: 6 }}>
                    (used for incident SMS alerts)
                  </span>
                </label>
                <input
                  className="input"
                  type="tel"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  placeholder="e.g. 91234567 or +6591234567"
                />
                {phone && !phone.startsWith("+") && (
                  <span className="small" style={{ color: "#0e7490", marginTop: 2 }}>
                    +65 will be added automatically
                  </span>
                )}
              </div>
            )}

            {message && (
              <div
                style={{
                  padding: "10px 14px",
                  background: "#e3f2fd",
                  border: "1px solid #90caf9",
                  borderRadius: 8,
                  fontSize: 13,
                }}
              >
                {message}
              </div>
            )}

            <div className="row" style={{ gap: 12, marginTop: 10 }}>
              <button className="button" onClick={handleSave} disabled={saving}>
                {saving ? "Saving..." : "Save Changes"}
              </button>
              <button className="button secondary" onClick={() => nav("/public")}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
