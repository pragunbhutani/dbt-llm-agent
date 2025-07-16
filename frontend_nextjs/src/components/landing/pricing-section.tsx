"use client";

import { CheckIcon } from "@heroicons/react/20/solid";
import { StarIcon } from "@heroicons/react/24/outline";
import Link from "next/link";
import { Button } from "@/components/ui/button";

// Pricing tiers configuration
const tiers = [
  {
    name: "Open Source",
    id: "tier-open-source",
    href: "https://github.com/pragunbhutani/dbt-llm-agent",
    priceMonthly: "$0",
    priceDescription: "Forever free",
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
    featured: false,
    ctaText: "View on GitHub",
    badge: "Most Popular",
  },
  {
    name: "Ragstar Cloud",
    id: "tier-cloud",
    href: "/signup",
    priceMonthly: "$99",
    priceDescription: "per month",
    description:
      "Fully managed cloud instance with premium features, enterprise security, and priority support.",
    features: [
      "Fully managed hosting",
      "Advanced analytics & insights",
      "Priority support (24/7)",
      "Team collaboration tools",
      "Enterprise integrations",
      "SOC2 compliance",
      "SLA guarantees",
      "Advanced governance controls",
    ],
    featured: true,
    ctaText: "Join Waitlist",
    badge: "Invite Only",
  },
] as const;

// Utility to merge class names safely
function classNames(...classes: (string | false | null | undefined)[]) {
  return classes.filter(Boolean).join(" ");
}

export default function PricingSection() {
  return (
    <div
      id="pricing"
      className="relative isolate bg-gray-50 px-6 py-16 sm:py-24 lg:px-8"
    >
      {/* Background shape */}
      <div
        aria-hidden="true"
        className="absolute inset-x-0 -top-3 -z-10 transform-gpu overflow-hidden px-36 blur-3xl"
      >
        <div
          style={{
            clipPath:
              "polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)",
          }}
          className="mx-auto aspect-[1155/678] w-[72rem] bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-30"
        />
      </div>

      {/* Section heading */}
      <div className="mx-auto max-w-4xl text-center">
        <h2 className="text-base font-semibold leading-7 text-indigo-600">
          Pricing
        </h2>
        <p className="mt-2 text-5xl font-semibold tracking-tight text-gray-900 sm:text-6xl">
          Start free, scale when ready
        </p>
      </div>
      <p className="mx-auto mt-6 max-w-2xl text-center text-lg text-gray-600 sm:text-xl">
        Begin with open source and upgrade to managed cloud when ready to scale.
      </p>

      {/* Pricing cards */}
      <div className="mx-auto mt-16 grid max-w-lg grid-cols-1 items-center gap-y-6 sm:mt-20 sm:gap-y-0 lg:max-w-4xl lg:grid-cols-2">
        {tiers.map((tier, tierIdx) => (
          <div
            key={tier.id}
            className={classNames(
              tier.featured
                ? "relative bg-gray-900 shadow-2xl"
                : "bg-white/60 sm:mx-8 lg:mx-0",
              tier.featured
                ? ""
                : tierIdx === 0
                ? "rounded-t-3xl sm:rounded-b-none lg:rounded-tr-none lg:rounded-bl-3xl"
                : "sm:rounded-t-none lg:rounded-tr-3xl lg:rounded-bl-none",
              "rounded-3xl p-8 ring-1 ring-gray-900/10 sm:p-10"
            )}
          >
            {/* Badge */}
            {tier.badge && (
              <div className="mb-4">
                <span
                  className={classNames(
                    tier.featured
                      ? "bg-indigo-500 text-white"
                      : "bg-green-100 text-green-700",
                    "inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-medium"
                  )}
                >
                  {!tier.featured && <StarIcon className="h-3 w-3" />}
                  {tier.badge}
                </span>
              </div>
            )}

            <h3
              id={tier.id}
              className={classNames(
                tier.featured ? "text-indigo-400" : "text-indigo-600",
                "text-base font-semibold leading-7"
              )}
            >
              {tier.name}
            </h3>
            <p className="mt-4 flex items-baseline gap-x-2">
              <span
                className={classNames(
                  tier.featured ? "text-white" : "text-gray-900",
                  "text-5xl font-semibold tracking-tight"
                )}
              >
                {tier.priceMonthly}
              </span>
              <span
                className={classNames(
                  tier.featured ? "text-gray-400" : "text-gray-500",
                  "text-base"
                )}
              >
                {tier.priceDescription}
              </span>
            </p>
            <p
              className={classNames(
                tier.featured ? "text-gray-300" : "text-gray-600",
                "mt-6 text-base leading-7"
              )}
            >
              {tier.description}
            </p>
            <ul
              role="list"
              className={classNames(
                tier.featured ? "text-gray-300" : "text-gray-600",
                "mt-8 space-y-3 text-sm leading-6 sm:mt-10"
              )}
            >
              {tier.features.map((feature) => (
                <li key={feature} className="flex gap-x-3">
                  <CheckIcon
                    aria-hidden="true"
                    className={classNames(
                      tier.featured ? "text-indigo-400" : "text-indigo-600",
                      "h-6 w-5 flex-none"
                    )}
                  />
                  {feature}
                </li>
              ))}
            </ul>

            {/* CTA */}
            <div className="mt-8 sm:mt-10">
              {tier.href.startsWith("http") ? (
                <Button
                  asChild
                  variant={tier.featured ? "default" : "outline"}
                  className="w-full"
                  size="lg"
                >
                  <Link
                    href={tier.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-describedby={tier.id}
                  >
                    {!tier.featured && <StarIcon className="h-4 w-4 mr-2" />}
                    {tier.ctaText}
                  </Link>
                </Button>
              ) : (
                <Button
                  asChild
                  variant={tier.featured ? "default" : "outline"}
                  className="w-full"
                  size="lg"
                >
                  <Link href={tier.href} aria-describedby={tier.id}>
                    {tier.ctaText}
                  </Link>
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Additional info */}
      <div className="mx-auto mt-16 max-w-2xl text-center">
        <p className="text-gray-600">
          <strong>Cloud version is invite-only</strong> while we perfect the
          experience. Join the waitlist to get early access and help shape the
          product.
        </p>
        <div className="mt-6 flex items-center justify-center gap-6 text-sm text-gray-500">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500"></div>
            <span>No vendor lock-in</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-blue-500"></div>
            <span>MIT license</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-purple-500"></div>
            <span>Enterprise ready</span>
          </div>
        </div>
      </div>
    </div>
  );
}
