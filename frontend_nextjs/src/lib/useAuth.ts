import { useSession, signOut } from "next-auth/react";
import { useEffect } from "react";

export function useAuth() {
  const { data: session, status } = useSession();

  const isAuthenticated = status === "authenticated" && !session?.error;
  const isLoading =
    status === "loading" || session?.error === "RefreshAccessTokenError";

  // Handle session errors globally
  useEffect(() => {
    if (session?.error === "RefreshAccessTokenError") {
      signOut({ callbackUrl: "/signin" });
    }
  }, [session?.error]);

  return {
    session,
    status,
    isAuthenticated,
    isLoading,
    accessToken: session?.accessToken,
  };
}
