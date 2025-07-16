"use client";

import { useEffect, useState } from "react";
import { ChevronRightIcon, ArrowRightIcon } from "@heroicons/react/20/solid";
import { StarIcon, ScaleIcon } from "@heroicons/react/24/outline";
import Link from "next/link";
import { useAuth } from "@/lib/useAuth";
import { Button } from "@/components/ui/button";
import { Github } from "lucide-react";
import { WaitlistModal } from "@/components/waitlist-modal";

/**
 * Marketing hero section displayed on the public landing page ("/").
 */
export default function LandingHero() {
  const { isAuthenticated, isLoading } = useAuth();
  const [latestTag, setLatestTag] = useState<string | null>(null);
  const [starCount, setStarCount] = useState<number | null>(null);

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
        console.error("Unable to fetch latest tag", err);
      }
    }

    async function fetchStarCount() {
      try {
        const res = await fetch(
          "https://api.github.com/repos/pragunbhutani/dbt-llm-agent"
        );
        if (res.ok) {
          const json = (await res.json()) as { stargazers_count: number };
          setStarCount(json.stargazers_count);
        }
      } catch (err) {
        console.error("Unable to fetch star count", err);
      }
    }

    fetchLatestTag();
    fetchStarCount();
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
      <div className="mx-auto max-w-7xl px-6 pb-24 sm:pb-32 lg:flex lg:px-8 lg:py-24">
        {/* Left column */}
        <div className="mx-auto max-w-2xl lg:mx-0 lg:shrink-0 lg:pt-8">
          <div className="mt-24 sm:mt-32 lg:mt-16">
            <Link
              href="https://github.com/pragunbhutani/dbt-llm-agent"
              className="inline-flex items-center space-x-6"
              target="_blank"
              rel="noopener noreferrer"
            >
              <span className="rounded-full bg-indigo-600/10 px-3 py-1 text-sm/6 font-semibold text-indigo-600 ring-1 ring-indigo-600/10 ring-inset flex items-center gap-2">
                <ScaleIcon className="h-4 w-4" />
                MIT License
              </span>
              {/* GitHub Star Button */}
              <p
                className="inline-flex items-center rounded-full bg-gray-100 px-3 py-1 text-sm/6 font-medium text-gray-700 hover:bg-gray-200 transition"
                style={{ lineHeight: "1.5" }}
                aria-label="Star dbt-llm-agent on GitHub"
              >
                <StarIcon className="h-4 w-4 mr-1 text-yellow-500" />
                <span>Star</span>
                {typeof starCount === "number" && (
                  <span className="ml-2 font-semibold text-gray-900 tabular-nums">
                    {starCount?.toLocaleString()}
                  </span>
                )}
              </p>

              {latestTag && (
                <span className="inline-flex items-center space-x-2 text-sm/6 font-medium text-gray-600">
                  <span>{`Latest: ${latestTag}`}</span>
                  <ChevronRightIcon
                    aria-hidden="true"
                    className="size-5 text-gray-400"
                  />
                </span>
              )}
            </Link>
          </div>
          <h1 className="mt-10 text-5xl font-semibold tracking-tight text-pretty text-gray-900 sm:text-7xl">
            Stop spending your time writing repetitive SQL queries
          </h1>
          <p className="mt-8 text-lg font-medium text-pretty text-gray-500 sm:text-xl/8">
            Self-serve analytics that data engineers trust. Ragstar connects to
            your dbt projects and warehouse so anyone can ask questions in plain
            English — while you focus on building.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row items-start sm:items-center gap-4">
            <div className="flex items-center gap-x-4">
              {isLoading ? null : isAuthenticated ? (
                <Button size="lg" asChild>
                  <Link href="/dashboard">Go to Dashboard</Link>
                </Button>
              ) : (
                <WaitlistModal
                  trigger={<Button size="lg">Join Cloud Waitlist</Button>}
                />
              )}
              <Button variant="outline" size="lg" asChild>
                <Link
                  href="https://github.com/pragunbhutani/dbt-llm-agent"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Github className="h-4 w-4 mr-2" />
                  GitHub
                </Link>
              </Button>
            </div>
          </div>

          <div className="mt-8 flex items-center gap-6 text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-green-500"></div>
              <span>Open Source & Self-Hosted</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-blue-500"></div>
              <span>Cloud @ $99/month (waitlist)</span>
            </div>
          </div>
        </div>

        {/* Right column – demo */}
        <div className="mx-auto mt-16 flex max-w-2xl sm:mt-40 lg:mr-0 lg:ml-10 lg:max-w-none lg:flex-none xl:ml-24">
          <div className="max-w-3xl flex-none sm:max-w-5xl lg:max-w-none">
            <div className="-m-2 rounded-xl bg-gray-900/5 p-2 ring-1 ring-gray-900/10 ring-inset lg:-m-4 lg:rounded-2xl lg:p-4">
              <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl p-6 text-white max-w-2xl">
                <div className="space-y-4">
                  <div className="flex items-center space-x-2 text-green-400">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span className="text-sm">Connected to dbt Cloud</span>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-4">
                    <p className="text-sm text-gray-300 mb-2">
                      💼 Head of Product asks:
                    </p>
                    <p className="text-white">
                      &quot;What&apos;s driving the drop in user activation this
                      month?&quot;
                    </p>
                  </div>
                  <div className="bg-indigo-600 rounded-lg p-4">
                    <p className="text-sm text-indigo-200 mb-2">
                      🤖 Ragstar responds:
                    </p>
                    <p className="text-white text-sm mb-3">
                      &quot;User activation dropped 12% due to email
                      verification issues. Here&apos;s the funnel analysis and
                      the fix:&quot;
                    </p>
                    <div className="bg-indigo-700 rounded p-3 text-xs font-mono">
                      SELECT funnel_step, conversion_rate{"\n"}
                      FROM user_activation_funnel{"\n"}
                      WHERE created_at &gt;= &apos;2024-01-01&apos;...
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <span>⚡ Answered in 3.2s</span>
                    <span>✅ Using prod models</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
