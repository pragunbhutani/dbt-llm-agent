"use client";

import Link from "next/link";
import { useAuth } from "@/lib/useAuth";

export default function CtaSection() {
  const { isAuthenticated, isLoading } = useAuth();

  return (
    <div className="bg-white">
      <div className="mx-auto max-w-7xl py-16 sm:px-6 sm:py-24 lg:px-8">
        <div className="relative isolate overflow-hidden bg-gray-900 px-6 py-24 text-center shadow-2xl sm:rounded-3xl sm:px-16">
          <h2 className="text-4xl font-semibold tracking-tight text-balance text-white sm:text-5xl">
            Boost your data productivity today
          </h2>
          <p className="mx-auto mt-6 max-w-xl text-lg/8 text-pretty text-gray-300">
            Free your team from ad-hoc requests and let everyone explore data
            securely.
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            {isLoading ? null : isAuthenticated ? (
              <Link
                href="/dashboard"
                className="rounded-md bg-white px-3.5 py-2.5 text-sm font-semibold text-gray-900 shadow-xs hover:bg-gray-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
              >
                Go to dashboard
              </Link>
            ) : (
              <Link
                href="/signup"
                className="rounded-md bg-white px-3.5 py-2.5 text-sm font-semibold text-gray-900 shadow-xs hover:bg-gray-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
              >
                Get started
              </Link>
            )}
            {!isAuthenticated && (
              <a href="#pricing" className="text-sm/6 font-semibold text-white">
                Learn more<span aria-hidden="true">→</span>
              </a>
            )}
          </div>
          <svg
            viewBox="0 0 1024 1024"
            aria-hidden="true"
            className="absolute top-1/2 left-1/2 -z-10 size-256 -translate-x-1/2 mask-[radial-gradient(closest-side,white,transparent)]"
          >
            <circle
              r={512}
              cx={512}
              cy={512}
              fill="url(#ctaGradient)"
              fillOpacity="0.7"
            />
            <defs>
              <radialGradient id="ctaGradient">
                <stop stopColor="#7775D6" />
                <stop offset={1} stopColor="#E935C1" />
              </radialGradient>
            </defs>
          </svg>
        </div>
      </div>
    </div>
  );
}
