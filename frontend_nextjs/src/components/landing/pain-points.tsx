"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Clock,
  MessageSquare,
  Users,
  TrendingDown,
  AlertCircle,
  Search,
} from "lucide-react";

const painPoints = [
  {
    icon: Clock,
    title: "50% Time Drain",
    description:
      'Data engineers spend half their time answering questions like "What&apos;s our churn?" for the 20th time.',
    color: "text-red-500",
    stat: "50%",
  },
  {
    icon: MessageSquare,
    title: "Constant Interruptions",
    description:
      "Product needs segments, Marketing wants cohorts, Executives need board numbers.",
    color: "text-orange-500",
    stat: "10x/day",
  },
  {
    icon: Users,
    title: "Onboarding Nightmare",
    description: "New hires take months to understand dbt models.",
    color: "text-yellow-500",
    stat: "3+ months",
  },
  {
    icon: TrendingDown,
    title: "Innovation Stagnation",
    description:
      "While you explain the same joins, your team's impact is invisible to leadership.",
    color: "text-blue-500",
    stat: "↓ Impact",
  },
  {
    icon: AlertCircle,
    title: "Data Blind Spots",
    description:
      "You don't know if analysts are using the right tables or what questions they're asking.",
    color: "text-purple-500",
    stat: "Unknown",
  },
  {
    icon: Search,
    title: "Knowledge Silos",
    description:
      "Critical business logic lives in your head. When you're away, the company has to wait for answers.",
    color: "text-pink-500",
    stat: "1 person",
  },
];

export default function PainPoints() {
  return (
    <section className="py-12 px-4 bg-white">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Does this sound familiar?
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Your team is stuck writing repetitive SQL queries instead of driving
            impact.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 py-4">
          {painPoints.map((point, index) => (
            <Card
              key={index}
              className="bg-white border-gray-200 hover:shadow-lg transition-shadow"
            >
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <point.icon className={`w-8 h-8 ${point.color} mb-2`} />
                  <span className={`text-sm font-semibold ${point.color}`}>
                    {point.stat}
                  </span>
                </div>
                <CardTitle className="text-lg">{point.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 text-sm leading-relaxed">
                  {point.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center">
          <div className="inline-flex items-center justify-center px-6 py-3 border border-gray-300 rounded-lg bg-white">
            <div className="text-sm text-gray-600">
              <span className="font-semibold text-gray-900">
                The bottom line:
              </span>{" "}
              You&apos;re stuck writing repetitive SQL queries instead of
              driving impact.
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
