import { getSession } from "next-auth/react";

const api = {
  async get(url: string) {
    const session = await getSession();
    const token = session?.accessToken;

    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${url}`, {
      headers: {
        "Content-Type": "application/json",
        ...(token && { Authorization: `Bearer ${token}` }),
      },
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    return response.json();
  },

  async put(url: string, data: any) {
    const session = await getSession();
    const token = session?.accessToken;

    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${url}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw { status: response.status, data: errorData };
    }
    return response.json();
  },

  async post(url: string, data: any) {
    const session = await getSession();
    const token = session?.accessToken;

    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${url}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw { status: response.status, data: errorData };
    }
    return response.json();
  },
};

export default api;
