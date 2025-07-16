"use client";

import React from "react";

const navigation = {
  main: [
    { name: "Features", href: "#features", external: false },
    { name: "Pricing", href: "#pricing", external: false },
    { name: "Installation", href: "#installation", external: false },
    {
      name: "Documentation",
      href: "https://github.com/pragunbhutani/dbt-llm-agent/blob/main/README.md",
      external: true,
    },
    {
      name: "Community",
      href: "https://github.com/pragunbhutani/dbt-llm-agent/discussions",
      external: true,
    },
  ],
  social: [
    {
      name: "X",
      href: "https://x.com/pragunbhutani",
      icon: (props: React.ComponentProps<"svg">) => (
        <svg fill="currentColor" viewBox="0 0 24 24" {...props}>
          <path d="M13.6823 10.6218L20.2391 3H18.6854L12.9921 9.61788L8.44486 3H3.2002L10.0765 13.0074L3.2002 21H4.75404L10.7663 14.0113L15.5685 21H20.8131L13.6819 10.6218H13.6823ZM11.5541 13.0956L10.8574 12.0991L5.31391 4.16971H7.70053L12.1742 10.5689L12.8709 11.5655L18.6861 19.8835H16.2995L11.5541 13.096V13.0956Z" />
        </svg>
      ),
    },
    {
      name: "GitHub",
      href: "https://github.com/pragunbhutani/dbt-llm-agent",
      icon: (props: React.ComponentProps<"svg">) => (
        <svg fill="currentColor" viewBox="0 0 24 24" {...props}>
          <path
            fillRule="evenodd"
            d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
            clipRule="evenodd"
          />
        </svg>
      ),
    },
  ],
} as const;

export default function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200">
      <div className="mx-auto max-w-7xl overflow-hidden px-6 py-20 sm:py-24 lg:px-8">
        <nav
          aria-label="Footer"
          className="-mb-6 flex flex-wrap justify-center gap-x-12 gap-y-3 text-sm leading-6"
        >
          {navigation.main.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className="text-gray-600 hover:text-gray-900"
              {...(item.external && {
                target: "_blank",
                rel: "noopener noreferrer",
              })}
            >
              {item.name}
            </a>
          ))}
        </nav>
        <div className="mt-16 flex justify-center gap-x-10">
          {navigation.social.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className="text-gray-600 hover:text-gray-800"
              target="_blank"
              rel="noopener noreferrer"
            >
              <span className="sr-only">{item.name}</span>
              <item.icon aria-hidden="true" className="h-6 w-6" />
            </a>
          ))}
        </div>

        <div className="mt-16 border-t border-gray-200 pt-8">
          <div className="text-center">
            <p className="text-sm leading-6 text-gray-600 mb-4">
              Open source AI data analyst for dbt teams
            </p>
            <div className="flex items-center justify-center gap-6 text-xs text-gray-500">
              <span>MIT License</span>
              <span>•</span>
              <span>Self-hostable</span>
              <span>•</span>
              <span>Enterprise ready</span>
            </div>
          </div>
        </div>

        <p className="mt-10 text-center text-sm leading-6 text-gray-600">
          &copy; {new Date().getFullYear()} Ragstar. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
