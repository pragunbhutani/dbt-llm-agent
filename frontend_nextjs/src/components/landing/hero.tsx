"use client";

import { useEffect, useState } from "react";
import { ChevronRightIcon } from "@heroicons/react/20/solid";
import Link from "next/link";
import { useAuth } from "@/lib/useAuth";

/**
 * Marketing hero section displayed on the public landing page ("/").
 *
 * Features:
 * 1. Fetches the latest tag from the dbt-llm-agent repo to surface in the "What's new" pill.
 * 2. Re-uses Tailwind template provided by the user for visual styling.
 * 3. Shows contextual CTA:
 *    • Authenticated users → "Go to dashboard" button.
 *    • Guests → "Get started" (signup) + secondary "Learn more" link.
 */
export default function LandingHero() {
  const { isAuthenticated, isLoading } = useAuth();
  const [latestTag, setLatestTag] = useState<string | null>(null);

  // Retrieve most recent Git tag once on mount.
  useEffect(() => {
    async function fetchLatestTag() {
      try {
        const res = await fetch(
          "https://api.github.com/repos/pragunbhutani/dbt-llm-agent/tags?per_page=1"
        );
        if (res.ok) {
          const json = (await res.json()) as Array<{ name: string }>;
          if (Array.isArray(json) && json.length > 0) {
            setLatestTag(json[0].name);
          }
        }
      } catch (err) {
        // Silently fail – we’ll just hide the tag.
        console.error("Unable to fetch latest tag", err);
      }
    }

    fetchLatestTag();
  }, []);

  return (
    <div className="relative isolate overflow-hidden bg-white">
      <svg
        aria-hidden="true"
        className="absolute inset-0 -z-10 size-full mask-[radial-gradient(100%_100%_at_top_right,white,transparent)] stroke-gray-200"
      >
        <defs>
          <pattern
            x="50%"
            y={-1}
            id="hero-pattern"
            width={200}
            height={200}
            patternUnits="userSpaceOnUse"
          >
            <path d="M.5 200V.5H200" fill="none" />
          </pattern>
        </defs>
        <rect
          fill="url(#hero-pattern)"
          width="100%"
          height="100%"
          strokeWidth={0}
        />
      </svg>
      <div className="mx-auto max-w-7xl px-6 pt-10 pb-24 sm:pb-32 lg:flex lg:px-8 lg:py-40">
        {/* Left column */}
        <div className="mx-auto max-w-2xl lg:mx-0 lg:shrink-0 lg:pt-8">
          {/* Logo placeholder – replace with real asset when available */}
          <img alt="Ragstar logo" src="/file.svg" className="h-11" />
          <div className="mt-24 sm:mt-32 lg:mt-16">
            <a
              href="https://github.com/pragunbhutani/dbt-llm-agent/releases"
              className="inline-flex space-x-6"
            >
              <span className="rounded-full bg-indigo-600/10 px-3 py-1 text-sm/6 font-semibold text-indigo-600 ring-1 ring-indigo-600/10 ring-inset">
                What's new
              </span>
              {latestTag && (
                <span className="inline-flex items-center space-x-2 text-sm/6 font-medium text-gray-600">
                  <span>{`Just shipped ${latestTag}`}</span>
                  <ChevronRightIcon
                    aria-hidden="true"
                    className="size-5 text-gray-400"
                  />
                </span>
              )}
            </a>
          </div>
          <h1 className="mt-10 text-5xl font-semibold tracking-tight text-pretty text-gray-900 sm:text-7xl">
            Turn questions into insights — instantly
          </h1>
          <p className="mt-8 text-lg font-medium text-pretty text-gray-500 sm:text-xl/8">
            Ragstar connects to your dbt project and warehouse so anyone can ask
            questions in plain English and get production-ready SQL, charts, and
            context in seconds.
          </p>
          <div className="mt-10 flex items-center gap-x-6">
            {isLoading ? null : isAuthenticated ? (
              <Link
                href="/dashboard"
                className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
              >
                Go to dashboard
              </Link>
            ) : (
              <>
                <Link
                  href="/signup"
                  className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
                >
                  Get started
                </Link>
                <Link
                  href="#features"
                  className="text-sm/6 font-semibold text-gray-900"
                >
                  Learn more <span aria-hidden="true">→</span>
                </Link>
              </>
            )}
          </div>
        </div>
        {/* Right column – screenshot */}
        <div className="mx-auto mt-16 flex max-w-2xl sm:mt-24 lg:mt-0 lg:mr-0 lg:ml-10 lg:max-w-none lg:flex-none xl:ml-32">
          <div className="max-w-3xl flex-none sm:max-w-5xl lg:max-w-none">
            <div className="-m-2 rounded-xl bg-gray-900/5 p-2 ring-1 ring-gray-900/10 ring-inset lg:-m-4 lg:rounded-2xl lg:p-4">
              <img
                alt="App screenshot"
                src="https://tailwindcss.com/plus-assets/img/component-images/project-app-screenshot.png"
                width={2432}
                height={1442}
                className="w-304 rounded-md shadow-2xl ring-1 ring-gray-900/10"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
