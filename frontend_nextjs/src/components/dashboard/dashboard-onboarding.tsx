"use client";

import useSWR from "swr";
import { useAuth } from "@/lib/useAuth";
import { fetcher } from "@/utils/fetcher";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircleIcon, ClockIcon } from "@heroicons/react/24/outline";

interface OnboardingSteps {
  connect_dbt_project: boolean;
  train_knowledge_base: boolean;
  connect_to_slack: boolean;
  ask_first_question: boolean;
}

interface StatsData {
  dbt_projects_count: number;
  total_models_count: number;
  knowledge_base_count: number;
  onboarding_steps: OnboardingSteps;
}

const ONBOARDING_STEPS = [
  {
    key: "connect_dbt_project" as keyof OnboardingSteps,
    title: "Connect your dbt Project",
    description: "Add your first dbt project to get started",
  },
  {
    key: "train_knowledge_base" as keyof OnboardingSteps,
    title: "Train your Knowledge Base",
    description: "Let us learn about your data models",
  },
  {
    key: "connect_to_slack" as keyof OnboardingSteps,
    title: "Connect to Slack",
    description: "Enable Slack integration for easy access",
  },
  {
    key: "ask_first_question" as keyof OnboardingSteps,
    title: "Ask your first question",
    description: "Test the system with a data question",
  },
];

export function DashboardOnboarding() {
  const { accessToken, isAuthenticated } = useAuth();

  const { data: statsData } = useSWR<StatsData>(
    isAuthenticated && accessToken
      ? "/api/data_sources/dashboard-stats/"
      : null,
    (url: string) => fetcher(url, accessToken),
    { suspense: true }
  );

  if (!statsData) return null;

  const completedSteps = ONBOARDING_STEPS.filter(
    (s) => statsData.onboarding_steps[s.key]
  ).length;
  const totalSteps = ONBOARDING_STEPS.length;
  const allCompleted = completedSteps === totalSteps;

  if (allCompleted) {
    return (
      <Card className="mb-6 border-green-200 bg-green-50">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-green-800">
              🎉 You&apos;re all set up!
            </CardTitle>
            <Badge variant="secondary" className="bg-green-100 text-green-800">
              Complete
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-green-700">
            Your Ragstar setup is complete. You can now ask questions about your
            data through the dashboard or Slack.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="mb-6">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Getting Started</CardTitle>
          <Badge variant="outline">
            {completedSteps}/{totalSteps} completed
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground">
          Follow these steps to set up your Ragstar dashboard
        </p>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {ONBOARDING_STEPS.map((step, index) => {
            const isCompleted = statsData.onboarding_steps[step.key];
            return (
              <div key={step.key} className="flex items-start space-x-3">
                <div className="flex-shrink-0 mt-0.5">
                  {isCompleted ? (
                    <CheckCircleIcon className="h-5 w-5 text-green-500" />
                  ) : (
                    <ClockIcon className="h-5 w-5 text-gray-400" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p
                    className={`text-sm font-medium ${
                      isCompleted ? "text-green-700" : "text-gray-900"
                    }`}
                  >
                    {step.title}
                  </p>
                  <p
                    className={`text-sm ${
                      isCompleted ? "text-green-600" : "text-gray-500"
                    }`}
                  >
                    {step.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
