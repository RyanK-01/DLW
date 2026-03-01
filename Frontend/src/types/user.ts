
export type UserRole = "public" | "responder" | "admin";

export interface UserProfile {
  role: UserRole;
  displayName?: string;
}