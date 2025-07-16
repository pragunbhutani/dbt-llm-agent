"use client";

import Link from "next/link";
import { useAuth } from "@/lib/useAuth";
import { Button } from "@/components/ui/button";
import { StarIcon } from "@heroicons/react/24/outline";
import { WaitlistModal } from "@/components/waitlist-modal";

export default function CtaSection() {
  const { isAuthenticated, isLoading } = useAuth();

  return (
    <div className="bg-white">
      <div className="mx-auto max-w-7xl py-16 sm:px-6 sm:py-24 lg:px-8">
        <div className="relative isolate overflow-hidden bg-gray-900 px-6 py-24 text-center shadow-2xl sm:rounded-3xl sm:px-16">
          <h2 className="text-4xl font-semibold tracking-tight text-balance text-white sm:text-5xl">
            Ready to stop being a human SQL compiler?
          </h2>
          <p className="mx-auto mt-6 max-w-xl text-lg/8 text-pretty text-gray-300">
            Join thousands of data engineers who&apos;ve reclaimed their time
            and become heroes instead of bottlenecks.
          </p>
          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
            {isLoading ? null : isAuthenticated ? (
              <Button
                size="lg"
                className="bg-white text-gray-900 hover:bg-gray-100"
              >
                <Link href="/dashboard">Go to Dashboard</Link>
              </Button>
            ) : (
              <>
                <WaitlistModal
                  trigger={
                    <Button
                      size="lg"
                      className="bg-white text-gray-900 hover:bg-gray-200"
                    >
                      Join Cloud Waitlist
                    </Button>
                  }
                />
                <Button
                  variant="outline"
                  size="lg"
                  className="border-white text-gray-900 hover:bg-gray-200"
                  asChild
                >
                  <Link
                    href="https://github.com/pragunbhutani/dbt-llm-agent"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <StarIcon className="w-4 h-4 mr-2" />
                    Start with Open Source
                  </Link>
                </Button>
              </>
            )}
          </div>
          <div className="mt-6 text-sm text-gray-400">
            <span className="font-semibold">Open source:</span> Free forever
            &nbsp;•&nbsp;
            <span className="font-semibold">Cloud:</span> $99/month (invite
            only)
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
