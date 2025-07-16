import { useSession, signOut } from "next-auth/react";
import { useEffect } from "react";
import { usePathname } from "next/navigation";

export function useAuth() {
  const { data: session, status } = useSession();
  const pathname = usePathname();

  const isAuthenticated = status === "authenticated" && !session?.error;
  const isLoading =
    status === "loading" || session?.error === "RefreshAccessTokenError";

  // Handle session errors globally, but only redirect if not on landing page
  useEffect(() => {
    if (session?.error === "RefreshAccessTokenError") {
      // Don't redirect if user is on landing page or auth pages
      if (
        pathname === "/" ||
        pathname.startsWith("/signin") ||
        pathname.startsWith("/signup")
      ) {
        return;
      }
      signOut({ callbackUrl: "/signin" });
    }
  }, [session?.error, pathname]);

  return {
    session,
    status,
    isAuthenticated,
    isLoading,
    accessToken: session?.accessToken,
  };
}
