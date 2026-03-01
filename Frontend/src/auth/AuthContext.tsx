
//to include firebase

import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { User } from "firebase/auth";
import { onAuthStateChanged } from "firebase/auth";
import { doc, onSnapshot } from "firebase/firestore";
import { auth, db } from "../firebase";
import type { UserRole } from "../types/user";

type AuthState = {
  user: User | null;
  role: UserRole | null;
  loading: boolean;
};

const Ctx = createContext<AuthState>({ user: null, role: null, loading: true });

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [role, setRole] = useState<UserRole | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let unsubRole: (() => void) | null = null;

    const unsubAuth = onAuthStateChanged(auth, (u) => {
      setUser(u);
      setRole(null);
      setLoading(true);

      // cleanup previous role listener
      if (unsubRole) {
        unsubRole();
        unsubRole = null;
      }

      if (!u) {
        setLoading(false);
        return;
      }

      const ref = doc(db, "users", u.uid);
      unsubRole = onSnapshot(
        ref,
        (snap) => {
          const data = snap.data() as any;
          setRole((data?.role ?? "public") as UserRole);
          setLoading(false);
        },
        () => {
          setRole("public");
          setLoading(false);
        }
      );
    });

    return () => {
      if (unsubRole) unsubRole();
      unsubAuth();
    };
  }, []);

  const value = useMemo(() => ({ user, role, loading }), [user, role, loading]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAuth() {
  return useContext(Ctx);
}