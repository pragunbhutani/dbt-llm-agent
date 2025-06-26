import { withAuth } from "next-auth/middleware";
import { NextResponse } from "next/server";

export default withAuth(
  function middleware(req) {
    const token = req.nextauth.token;
    const isAuth = !!token && !token.error; // Check that token exists and has no errors
    const isAuthPage =
      req.nextUrl.pathname.startsWith("/signin") ||
      req.nextUrl.pathname.startsWith("/signup");

    // If token has an error (like RefreshAccessTokenError), treat as unauthenticated
    if (token?.error === "RefreshAccessTokenError") {
      if (!isAuthPage) {
        let from = req.nextUrl.pathname;
        if (req.nextUrl.search) {
          from += req.nextUrl.search;
        }
        return NextResponse.redirect(
          new URL(`/signin?from=${encodeURIComponent(from)}`, req.url)
        );
      }
      return null;
    }

    if (isAuthPage) {
      if (isAuth) {
        return NextResponse.redirect(new URL("/dashboard", req.url));
      }
      return null;
    }

    if (!isAuth && req.nextUrl.pathname.startsWith("/dashboard")) {
      let from = req.nextUrl.pathname;
      if (req.nextUrl.search) {
        from += req.nextUrl.search;
      }

      return NextResponse.redirect(
        new URL(`/signin?from=${encodeURIComponent(from)}`, req.url)
      );
    }
  },
  {
    callbacks: {
      authorized: () => true, // Let the middleware handle redirect logic
    },
  }
);

export const config = {
  matcher: ["/dashboard/:path*", "/signin", "/signup"],
};
