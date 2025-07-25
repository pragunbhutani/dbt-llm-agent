import NextAuth, { User } from "next-auth";
import { JWT } from "next-auth/jwt";
import CredentialsProvider from "next-auth/providers/credentials";
import { jwtDecode } from "jwt-decode";

// Function to refresh the access token
async function refreshAccessToken(token: JWT): Promise<JWT> {
  try {
    // Use internal Docker service name for server-side requests
    const apiUrl =
      process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL;
    const response = await fetch(`${apiUrl}/api/token/refresh/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh: token.refreshToken }),
    });

    const refreshedTokens = await response.json();

    if (!response.ok) {
      throw refreshedTokens;
    }

    const newAccessToken = refreshedTokens.access;
    const decodedToken: { exp: number } = jwtDecode(newAccessToken);

    return {
      ...token,
      accessToken: newAccessToken,
      accessTokenExpiresAt: decodedToken.exp * 1000,
      // The refresh token might be rotated as well, but Django's default simple-jwt setup doesn't.
      // If it did, we would update it here:
      // refreshToken: refreshedTokens.refresh ?? token.refreshToken,
      error: undefined, // Clear any previous error
    };
  } catch (error) {
    console.error("Error refreshing access token", error);

    // Check if this is a blacklisted token error
    if (
      error &&
      typeof error === "object" &&
      "detail" in error &&
      (error.detail === "Token is blacklisted" ||
        (error as any).code === "token_not_valid")
    ) {
      // Return a token that signals complete logout
      return {
        error: "TokenBlacklisted" as const,
      };
    }

    // The refresh token is likely expired or invalid, so we need to sign out
    return {
      ...token,
      error: "RefreshAccessTokenError" as const,
    };
  }
}

const handler = NextAuth({
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "text" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials): Promise<User | null> {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        try {
          // Use internal Docker service name for server-side requests
          const apiUrl =
            process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL;

          // 1. Get Access and Refresh Tokens
          const tokenRes = await fetch(`${apiUrl}/api/token/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              email: credentials.email,
              password: credentials.password,
            }),
          });

          if (!tokenRes.ok) return null;
          const tokenData = await tokenRes.json();
          const accessToken = tokenData.access;
          const refreshToken = tokenData.refresh;

          if (!accessToken || !refreshToken) return null;

          // Decode token to get expiry time
          const decodedToken: { exp: number } = jwtDecode(accessToken);
          const accessTokenExpiresAt = decodedToken.exp * 1000;

          // 2. Get User Details
          const userRes = await fetch(`${apiUrl}/api/accounts/me/`, {
            headers: { Authorization: `Bearer ${accessToken}` },
          });

          if (!userRes.ok) return null;
          const userData = await userRes.json();

          // 3. Return a user object that includes all necessary details
          return {
            id: userData.id,
            email: userData.email,
            name: userData.first_name,
            accessToken,
            accessTokenExpiresAt,
            refreshToken,
          };
        } catch (error) {
          console.error("Authorize error:", error);
          return null;
        }
      },
    }),
  ],
  session: {
    strategy: "jwt",
  },
  callbacks: {
    // The 'user' object is the one returned from the 'authorize' callback
    async jwt({ token, user }) {
      // Initial sign in
      if (user) {
        token.id = user.id;
        token.accessToken = user.accessToken;
        token.accessTokenExpiresAt = user.accessTokenExpiresAt;
        token.refreshToken = user.refreshToken;
        return token;
      }

      // If token is blacklisted, clear the session completely
      if (token.error === "TokenBlacklisted") {
        return { error: "TokenBlacklisted" as const };
      }

      // Return previous token if the access token has not expired yet
      if (
        token.accessTokenExpiresAt &&
        Date.now() < token.accessTokenExpiresAt
      ) {
        return token;
      }

      // Access token has expired, try to update it
      return refreshAccessToken(token);
    },
    // The 'token' object is the one returned from the 'jwt' callback
    async session({ session, token }) {
      // If token is blacklisted or empty, return minimal empty session
      if (token.error === "TokenBlacklisted" || !token.accessToken) {
        return {
          expires: session.expires,
          user: undefined,
          accessToken: undefined,
          error: "TokenBlacklisted" as const,
        };
      }

      if (session.user && token.id) {
        session.user.id = token.id;
      }
      session.accessToken = token.accessToken;
      session.error = token.error;
      return session;
    },
  },
  pages: {
    signIn: "/signin",
  },
});

export { handler as GET, handler as POST };
