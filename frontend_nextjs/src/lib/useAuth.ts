import { useSession, signOut } from "next-auth/react";
import { useEffect } from "react";
import { usePathname } from "next/navigation";

export function useAuth() {
  const { data: session, status } = useSession();
  const pathname = usePathname();

  // Treat blacklisted tokens and sessions without accessToken as unauthenticated
  const isAuthenticated =
    status === "authenticated" &&
    !session?.error &&
    session?.accessToken &&
    session?.user;
  const isLoading = status === "loading";

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

    // Handle blacklisted tokens by clearing the session completely
    if (session?.error === "TokenBlacklisted") {
      // Clear the session without redirect for all pages
      signOut({ redirect: false });
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
