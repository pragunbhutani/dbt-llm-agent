"use client";

import useSWR from "swr";
import { useSession } from "next-auth/react";
import { fetcher } from "@/utils/fetcher";
import Link from "next/link";
import { CheckCircle2, Circle } from "lucide-react";
import { Card } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

function OnboardingStep({
  isComplete,
  children,
}: {
  isComplete: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center space-x-2">
      {isComplete ? (
        <CheckCircle2 className="h-5 w-5 text-green-500" />
      ) : (
        <Circle className="h-5 w-5 text-gray-400" />
      )}
      <span>{children}</span>
    </div>
  );
}

export function DashboardOnboarding() {
  const { data: session } = useSession();

  const { data: statsData } = useSWR(
    session?.accessToken ? "/api/data_sources/dashboard-stats/" : null,
    (url: string) => fetcher(url, session?.accessToken),
    { suspense: true }
  );

  const stats = statsData || {
    onboarding_steps: {
      connect_dbt_project: false,
      train_knowledge_base: false,
      connect_to_slack: false,
      ask_first_question: false,
    },
  };

  return (
    <Card className="mb-4">
      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="item-1" className="border-none">
          <AccordionTrigger className="px-6 py-2 hover:no-underline">
            <div className="text-left">
              <h3 className="text-lg font-semibold">Getting Started</h3>
              <p className="text-sm text-muted-foreground">
                Complete these steps to get your AI data analyst up and running.
              </p>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-6 pb-4">
            <div className="border-t pt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">
                    Initial Onboarding Checklist
                  </h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    Basic setup to get you started.
                  </p>
                  <ul className="space-y-2">
                    <li>
                      <OnboardingStep
                        isComplete={stats.onboarding_steps.connect_dbt_project}
                      >
                        Connect a dbt project
                      </OnboardingStep>
                    </li>
                    <li>
                      <OnboardingStep
                        isComplete={stats.onboarding_steps.train_knowledge_base}
                      >
                        Train your knowledge base
                      </OnboardingStep>
                    </li>
                    <li>
                      <OnboardingStep
                        isComplete={stats.onboarding_steps.connect_to_slack}
                      >
                        Connect to Slack
                      </OnboardingStep>
                    </li>
                    <li>
                      <OnboardingStep
                        isComplete={stats.onboarding_steps.ask_first_question}
                      >
                        Ask your first question
                      </OnboardingStep>
                    </li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Optional Next Steps</h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    Further enhance your Ragstar experience with these optional
                    integrations.
                  </p>
                  <ul className="space-y-2">
                    <li>
                      <Link
                        href="/dashboard/settings"
                        className="text-primary hover:underline"
                      >
                        Configure your LLM API keys or models
                      </Link>
                    </li>
                    <li>
                      <Link
                        href="/dashboard/integrations"
                        className="text-primary hover:underline"
                      >
                        Connect to Snowflake
                      </Link>
                    </li>
                    <li>
                      <Link
                        href="/dashboard/integrations"
                        className="text-primary hover:underline"
                      >
                        Connect to Metabase
                      </Link>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </Card>
  );
}
