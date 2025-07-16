"use client";

import {
  GitBranch,
  Zap,
  BarChart3,
  Shield,
  Users,
  TrendingUp,
  Brain,
  Clock,
} from "lucide-react";

const benefits = [
  {
    icon: Brain,
    title: "AI That Understands Your Data",
    description:
      "Deep dbt integration means Ragstar knows your models, tests, and business logic. No explaining schemas.",
    color: "bg-blue-100 text-blue-600",
  },
  {
    icon: Clock,
    title: "Reclaim 20+ Hours Per Week",
    description:
      "Stop interruptions every 30 minutes. Stakeholders self-serve while you build real impact.",
    color: "bg-green-100 text-green-600",
  },
  {
    icon: TrendingUp,
    title: "Become the Hero, Not the Bottleneck",
    description:
      "Leadership sees instant data delivery. You become the engineer who enabled company-wide data literacy.",
    color: "bg-purple-100 text-purple-600",
  },
  {
    icon: Users,
    title: "Onboard New Hires in Days",
    description:
      "New engineers understand data models through natural language. Productive from day one.",
    color: "bg-orange-100 text-orange-600",
  },
];

const workflow = [
  {
    step: "01",
    title: "Connect Your dbt Project",
    description:
      "Connect to your dbt project and select the models that should be used for answering questions.",
    icon: GitBranch,
  },
  {
    step: "02",
    title: "Connect Integrations",
    description:
      "Connect to Slack and optionally, Snowflake and Metabase for enhanced functionality.",
    icon: Zap,
  },
  {
    step: "03",
    title: "Start Asking Questions",
    description:
      "Your team can now ask questions in natural language and get instant, accurate answers.",
    icon: BarChart3,
  },
];

export default function Solution() {
  return (
    <section className="py-24 px-4 bg-gray-50">
      <div className="container mx-auto max-w-6xl">
        {/* Main headline */}
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            It&apos;s time to automate the mundane
          </h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Self-serve analytics that understands your dbt projects, turns
            stakeholders into data analysts and lets you focus on building.
          </p>
        </div>

        {/* Benefits grid */}
        <div className="grid md:grid-cols-2 gap-8 mb-16">
          {benefits.map((benefit, index) => (
            <div key={index} className="flex items-start space-x-4">
              <div
                className={`w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 ${benefit.color}`}
              >
                <benefit.icon className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {benefit.title}
                </h3>
                <p className="text-gray-600">{benefit.description}</p>
              </div>
            </div>
          ))}
        </div>

        {/* How it works */}
        <div className="text-center py-8">
          <h3 className="text-2xl font-bold text-gray-900 mb-4">
            How it works
          </h3>
          <p className="text-lg text-gray-600">
            Set up in minutes, not weeks. Start seeing impact immediately.
          </p>
        </div>

        <div className="grid md:grid-cols-3 mt7-12">
          {workflow.map((step, index) => (
            <div key={index} className="text-center">
              <div className="w-16 h-16 bg-indigo-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <step.icon className="w-8 h-8 text-white" />
              </div>
              <div className="text-sm font-semibold text-indigo-600 mb-2">
                STEP {step.step}
              </div>
              <h4 className="text-lg font-semibold text-gray-900 mb-2">
                {step.title}
              </h4>
              <p className="text-gray-600 text-sm">{step.description}</p>
            </div>
          ))}
        </div>

        {/* Value proposition summary */}
        {/* <div className="mt-16 bg-gray-50 rounded-2xl p-8">
          <div className="text-center">
            <h3 className="text-2xl font-bold text-gray-900 mb-16">
              The result? You're no longer the bottleneck.
            </h3>
            <div className="grid md:grid-cols-3 gap-6 text-center">
              <div>
                <div className="text-2xl font-bold text-indigo-600 mb-2">
                  20+
                </div>
                <p className="text-gray-600">Hours saved per week</p>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600 mb-2">
                  90%
                </div>
                <p className="text-gray-600">Reduction in ad-hoc requests</p>
              </div>
              <div>
                <div className="text-2xl font-bold text-purple-600 mb-2">
                  10x
                </div>
                <p className="text-gray-600">Faster stakeholder insights</p>
              </div>
            </div>
          </div>
        </div> */}
      </div>
    </section>
  );
}
