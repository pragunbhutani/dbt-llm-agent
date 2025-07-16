"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Terminal,
  Download,
  Play,
  CheckCircle,
  Copy,
  Container,
} from "lucide-react";

const installationSteps = [
  {
    step: 1,
    title: "Clone the Repository",
    description: "Get the latest version from GitHub",
    code: `git clone https://github.com/pragunbhutani/dbt-llm-agent.git
cd dbt-llm-agent`,
    icon: Download,
  },
  {
    step: 2,
    title: "Set Environment Variables",
    description: "Configure your environment variables",
    code: `cp .env.example .env
# Edit .env and fill in the required values`,
    icon: Terminal,
  },
  {
    step: 3,
    title: "Start with Docker",
    description: "Build and start all services",
    code: `docker compose up -d --build`,
    icon: Container,
  },
  {
    step: 4,
    title: "Run Database Migrations",
    description: "Initialize the database",
    code: `docker compose exec backend-python manage.py migrate`,
    icon: CheckCircle,
  },
];

const deploymentOptions = [
  {
    title: "Docker Compose",
    description: "Quick setup for local development",
    icon: Container,
    code: `docker compose up -d --build`,
  },
  {
    title: "Cloud Platforms",
    description: "Vercel + Railway + Supabase",
    icon: Terminal,
    code: `# Frontend: Vercel
# Backend: Railway
# Database: Supabase (+ pgvector)`,
  },
  {
    title: "Ragstar Cloud",
    description: "Fully managed cloud instance",
    icon: Play,
    code: `# Join the waitlist`,
  },
];

export default function Installation() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <section id="installation" className="px-4 bg-white">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Get Started in Minutes
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Ragstar is open source and ready to deploy. Choose your setup method
            and start analyzing your dbt projects today.
          </p>
        </div>

        {/* Installation Steps */}
        <div className="mb-16">
          <h3 className="text-2xl font-semibold text-gray-900 mb-8 text-center">
            Quick Start Installation
          </h3>
          <div className="grid gap-6">
            {installationSteps.map((step, index) => (
              <Card key={index} className="bg-white border-gray-200">
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-indigo-100 rounded-full flex items-center justify-center">
                      <step.icon className="w-5 h-5 text-indigo-600" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">
                        Step {step.step}: {step.title}
                      </CardTitle>
                      <p className="text-sm text-gray-600 mt-1">
                        {step.description}
                      </p>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="relative">
                    <pre className="bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto text-sm">
                      <code>{step.code}</code>
                    </pre>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute top-2 right-2 text-gray-400 hover:text-white"
                      onClick={() =>
                        copyToClipboard(step.code, `step-${step.step}`)
                      }
                    >
                      {copiedCode === `step-${step.step}` ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : (
                        <Copy className="w-4 h-4" />
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Deployment Options */}
        <div className="mb-16">
          <h3 className="text-2xl font-semibold text-gray-900 mb-8 text-center">
            Deployment Options
          </h3>
          <div className="grid md:grid-cols-3 gap-6">
            {deploymentOptions.map((option, index) => (
              <Card
                key={index}
                className="bg-white border-gray-200 hover:shadow-lg transition-shadow"
              >
                <CardHeader className="text-center">
                  <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <option.icon className="w-6 h-6 text-indigo-600" />
                  </div>
                  <CardTitle className="text-lg">{option.title}</CardTitle>
                  <p className="text-sm text-gray-600 mt-2">
                    {option.description}
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="relative">
                    <pre className="bg-gray-900 text-green-400 p-3 rounded text-sm">
                      <code>{option.code}</code>
                    </pre>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute top-2 right-2 text-gray-400 hover:text-white"
                      onClick={() =>
                        copyToClipboard(option.code, `deploy-${index}`)
                      }
                    >
                      {copiedCode === `deploy-${index}` ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : (
                        <Copy className="w-4 h-4" />
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Requirements & Support */}
        <div className="grid md:grid-cols-2 gap-8">
          <Card className="bg-white border-gray-200">
            <CardHeader>
              <CardTitle className="text-lg">Requirements</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-gray-600">
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  Python 3.11+
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  Node.js 18+
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  PostgreSQL 13+
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  Redis 6+
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  OpenAI API key or compatible LLM
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="bg-white border-gray-200">
            <CardHeader>
              <CardTitle className="text-lg">Need Help?</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  asChild
                >
                  <a
                    href="https://github.com/pragunbhutani/dbt-llm-agent/blob/main/README.md"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    📖 Read the Docs
                  </a>
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  asChild
                >
                  <a
                    href="https://github.com/pragunbhutani/dbt-llm-agent/issues"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    🐛 Report an Issue
                  </a>
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  asChild
                >
                  <a
                    href="https://github.com/pragunbhutani/dbt-llm-agent/discussions"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    💬 Community Support
                  </a>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}
