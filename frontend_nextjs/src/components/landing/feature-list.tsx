"use client";

import {
  ArrowPathIcon,
  CloudArrowUpIcon,
  FingerPrintIcon,
  LockClosedIcon,
} from "@heroicons/react/24/outline";

const features = [
  {
    name: "AI-generated SQL",
    description:
      "Transform natural language questions into optimized SQL for Snowflake, BigQuery, Redshift, and Postgres.",
    icon: CloudArrowUpIcon,
  },
  {
    name: "Governance & Lineage",
    description:
      "Column-level lineage, tests, and exposures from your dbt project stay intact — no more shadow queries.",
    icon: LockClosedIcon,
  },
  {
    name: "Quick Iteration",
    description:
      "Iterate on prompts and models rapidly with versioned workflows and automatic testing.",
    icon: ArrowPathIcon,
  },
  {
    name: "Enterprise-grade Security",
    description:
      "Secure architecture with least-privilege warehouse credentials and encrypted secrets.",
    icon: FingerPrintIcon,
  },
] as const;

export default function FeatureList() {
  return (
    <div className="bg-gray-50 py-16 sm:py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl lg:text-center">
          <h2 className="text-base font-semibold leading-7 text-indigo-600">
            Powerful &amp; Secure
          </h2>
          <p className="mt-2 text-4xl font-semibold tracking-tight text-gray-900 sm:text-5xl lg:text-balance">
            Unlock your data with powerful, secure AI
          </p>
          <p className="mt-6 text-lg leading-8 text-gray-700">
            Ragstar combines the power of LLMs with the rigor of your existing
            dbt models and warehouse permissions.
          </p>
        </div>
        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-4xl">
          <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-10 lg:max-w-none lg:grid-cols-2 lg:gap-y-16">
            {features.map((feature) => (
              <div key={feature.name} className="relative pl-16">
                <dt className="text-base font-semibold leading-7 text-gray-900">
                  <div className="absolute top-0 left-0 flex size-10 items-center justify-center rounded-lg bg-indigo-600">
                    <feature.icon
                      aria-hidden="true"
                      className="size-6 text-white"
                    />
                  </div>
                  {feature.name}
                </dt>
                <dd className="mt-2 text-base leading-7 text-gray-600">
                  {feature.description}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </div>
  );
}
