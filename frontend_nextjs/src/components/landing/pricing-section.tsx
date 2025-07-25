"use client";

import { CheckIcon } from "@heroicons/react/20/solid";
import { StarIcon } from "@heroicons/react/24/outline";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { WaitlistModal } from "@/components/waitlist-modal";

const tiers = [
  {
    name: "Open Source",
    id: "tier-open-source",
    href: "https://github.com/pragunbhutani/dbt-llm-agent",
    priceMonthly: "$0",
    description:
      "Perfect for data teams who want full control and customization. Deploy anywhere, modify anything.",
    features: [
      "Complete source code (MIT license)",
      "Self-hosted deployment",
      "Unlimited dbt projects",
      "All core AI features",
      "Community support",
      "MCP server integration",
      "Docker & Kubernetes ready",
    ],
    ctaText: "View on GitHub",
    badge: "Most Popular",
    isExternal: true,
  },
  {
    name: "Ragstar Cloud",
    id: "tier-cloud",
    priceMonthly: "$99",
    description:
      "Fully managed cloud instance with premium features and priority support for teams who want to focus on building.",
    features: [
      "Fully managed hosting",
      "Priority support (24/7)",
      "Team collaboration tools",
      "Enterprise integrations",
    ],
    ctaText: "Join Waitlist",
    badge: "Invite Only",
    isWaitlist: true,
  },
];

export default function PricingSection() {
  return (
    <div id="pricing" className="isolate overflow-hidden bg-gray-900">
      <div className="mx-auto max-w-7xl px-6 pt-24 pb-96 text-center sm:pt-32 lg:px-8">
        <div className="mx-auto max-w-4xl">
          <h2 className="text-base/7 font-semibold text-indigo-400">Pricing</h2>
          <p className="mt-2 text-5xl font-semibold tracking-tight text-balance text-white sm:text-6xl">
            Start free, scale when ready
          </p>
        </div>
        <div className="relative mt-6">
          <p className="mx-auto max-w-2xl text-lg font-medium text-pretty text-gray-400 sm:text-xl/8">
            Begin with open source and upgrade to managed cloud when ready to
            scale. No vendor lock-in, MIT license.
          </p>
          <svg
            viewBox="0 0 1208 1024"
            className="absolute -top-10 left-1/2 -z-10 h-256 -translate-x-1/2 mask-[radial-gradient(closest-side,white,transparent)] sm:-top-12 md:-top-20 lg:-top-12 xl:top-0"
          >
            <ellipse
              cx={604}
              cy={512}
              rx={604}
              ry={512}
              fill="url(#6d1bd035-0dd1-437e-93fa-59d316231eb0)"
            />
            <defs>
              <radialGradient id="6d1bd035-0dd1-437e-93fa-59d316231eb0">
                <stop stopColor="#7775D6" />
                <stop offset={1} stopColor="#E935C1" />
              </radialGradient>
            </defs>
          </svg>
        </div>
      </div>
      <div className="flow-root bg-white pb-24 sm:pb-32">
        <div className="-mt-80">
          <div className="mx-auto max-w-7xl px-6 lg:px-8">
            <div className="mx-auto grid max-w-md grid-cols-1 gap-8 lg:max-w-4xl lg:grid-cols-2">
              {tiers.map((tier) => (
                <div
                  key={tier.id}
                  className="flex flex-col justify-between rounded-3xl bg-white p-8 shadow-xl ring-1 ring-gray-900/10 sm:p-10"
                >
                  <div>
                    {/* Badge */}
                    {tier.badge && (
                      <div className="mb-4">
                        <span
                          className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-medium ${
                            tier.badge === "Most Popular"
                              ? "bg-green-100 text-green-700"
                              : "bg-indigo-100 text-indigo-700"
                          }`}
                        >
                          {tier.badge === "Most Popular" && (
                            <StarIcon className="h-3 w-3" />
                          )}
                          {tier.badge}
                        </span>
                      </div>
                    )}

                    <h3
                      id={tier.id}
                      className="text-base/7 font-semibold text-indigo-600"
                    >
                      {tier.name}
                    </h3>
                    <div className="mt-4 flex items-baseline gap-x-2">
                      <span className="text-5xl font-semibold tracking-tight text-gray-900">
                        {tier.priceMonthly}
                      </span>
                      <span className="text-base/7 font-semibold text-gray-600">
                        /month
                      </span>
                    </div>
                    <p className="mt-6 text-base/7 text-gray-600">
                      {tier.description}
                    </p>
                    <ul
                      role="list"
                      className="mt-10 space-y-4 text-sm/6 text-gray-600"
                    >
                      {tier.features.map((feature) => (
                        <li key={feature} className="flex gap-x-3">
                          <CheckIcon
                            aria-hidden="true"
                            className="h-6 w-5 flex-none text-indigo-600"
                          />
                          {feature}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* CTA Button */}
                  {tier.isWaitlist ? (
                    <WaitlistModal
                      trigger={
                        <button className="mt-8 block w-full rounded-md bg-indigo-600 px-3.5 py-2 text-center text-sm/6 font-semibold text-white shadow-xs hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
                          {tier.ctaText}
                        </button>
                      }
                    />
                  ) : tier.isExternal && tier.href ? (
                    <Link
                      href={tier.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-describedby={tier.id}
                      className="mt-8 block rounded-md bg-indigo-600 px-3.5 py-2 text-center text-sm/6 font-semibold text-white shadow-xs hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
                    >
                      {tier.ctaText}
                    </Link>
                  ) : tier.href ? (
                    <Link
                      href={tier.href}
                      aria-describedby={tier.id}
                      className="mt-8 block rounded-md bg-indigo-600 px-3.5 py-2 text-center text-sm/6 font-semibold text-white shadow-xs hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
                    >
                      {tier.ctaText}
                    </Link>
                  ) : null}
                </div>
              ))}

              {/* Additional info card */}
              <div className="flex flex-col items-start gap-x-8 gap-y-6 rounded-3xl p-8 ring-1 ring-gray-900/10 sm:gap-y-10 sm:p-10 lg:col-span-2 lg:flex-row lg:items-center bg-gray-50">
                <div className="lg:min-w-0 lg:flex-1">
                  <h3 className="text-base/7 font-semibold text-indigo-600">
                    Cloud Waitlist
                  </h3>
                  <p className="mt-1 text-base/7 text-gray-600">
                    Cloud version is invite-only while we perfect the
                    experience. Join the waitlist to get early access and help
                    shape the product.
                  </p>
                </div>
                <WaitlistModal
                  trigger={
                    <button className="rounded-md px-3.5 py-2 text-sm/6 font-semibold text-indigo-600 ring-1 ring-indigo-200 ring-inset hover:ring-indigo-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
                      Join Waitlist <span aria-hidden="true">&rarr;</span>
                    </button>
                  }
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
