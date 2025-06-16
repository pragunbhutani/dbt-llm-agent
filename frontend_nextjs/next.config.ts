import type { NextConfig } from "next";
import path from "path";
import dotenv from "dotenv";

if (process.env.NODE_ENV !== "production") {
  dotenv.config({ path: path.resolve(__dirname, "../.env") });
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  /* config options here */
  env: {
    // Provides a default for local development if the .env file is not set up.
    // In production, this should be set via your hosting provider's environment variables.
    NEXT_PUBLIC_API_URL:
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  },
};

export default nextConfig;
