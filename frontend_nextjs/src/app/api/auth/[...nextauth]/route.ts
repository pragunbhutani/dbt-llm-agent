import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

const handler = NextAuth({
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "text" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        try {
          const res = await fetch(
            `${process.env.NEXT_PUBLIC_API_URL}/api/token/`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                email: credentials.email,
                password: credentials.password,
              }),
            }
          );

          if (!res.ok) {
            return null;
          }

          const data = await res.json();

          // The token response from simple-jwt doesn't contain the full user object.
          // We need another request to get user details, or we can decode the token
          // if we add user details to it. For now, let's assume we have what we need
          // or we can just return a user object with a placeholder. A better approach
          // would be to have an endpoint like /api/me/ that returns the user object.
          // Let's call the registration endpoint which returns user details. Not ideal.

          // Let's adjust the simplejwt view to return user data.

          // For now, if a token is returned, we consider it a success.
          // We'll decode the user from the token. But next-auth needs an object.
          if (data.access) {
            // A proper implementation would decode the JWT to get user info.
            // Or, the token endpoint could return the user object.
            // Let's assume the latter for simplicity. The user object is in data.user

            const userRes = await fetch(
              `${process.env.NEXT_PUBLIC_API_URL}/api/accounts/user/`,
              {
                headers: {
                  Authorization: `Bearer ${data.access}`,
                },
              }
            );

            if (!userRes.ok) {
              return null;
            }

            const user = await userRes.json();

            return {
              id: user.id,
              email: user.email,
              name: user.first_name,
            };
          }
          return null;
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
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.id as string;
      }
      return session;
    },
  },
  pages: {
    signIn: "/signin",
  },
});

export { handler as GET, handler as POST };
