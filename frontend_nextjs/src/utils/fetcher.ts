import { signOut } from "next-auth/react";

export const fetcher = async (
  url: string,
  token: string | undefined,
  options: { method?: string; body?: any } = {}
) => {
  if (!token) {
    throw new Error("Not authorized");
  }

  const { method = "GET", body = null } = options;

  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${url}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json", // Important for POST requests
    },
    body: body ? JSON.stringify(body) : null,
  });

  if (!res.ok) {
    // Handle authentication errors by redirecting to signin
    if (res.status === 401) {
      // Token is invalid/expired, trigger sign out
      signOut({ callbackUrl: "/signin" });
      const error = new Error("Authentication failed - redirecting to signin");
      (error as any).status = 401;
      throw error;
    }

    const error = new Error("An error occurred while fetching the data.");
    // Attach extra info to the error object.
    const errorInfo = await res.json().catch(() => ({})); // Handle cases where error is not JSON
    (error as any).info = errorInfo;
    (error as any).status = res.status;
    throw error;
  }
  // For 202 Accepted or 204 No Content, response body might be empty
  if (res.status === 202 || res.status === 204) {
    return null;
  }

  return res.json();
};
