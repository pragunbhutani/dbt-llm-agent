"use client";

import useSWR from "swr";
import { useSession } from "next-auth/react";
import { fetcher } from "@/utils/fetcher";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function DashboardStats() {
  const { data: session } = useSession();

  const { data: statsData } = useSWR(
    session?.accessToken ? "/api/data_sources/dashboard-stats/" : null,
    (url: string) => fetcher(url, session?.accessToken),
    { suspense: true }
  );

  const stats = statsData || {
    dbt_projects_count: 0,
    total_models_count: 0,
    knowledge_base_count: 0,
    onboarding_steps: {
      connect_dbt_project: false,
      train_knowledge_base: false,
      connect_to_slack: false,
      ask_first_question: false,
    },
  };

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-4">
      <Card>
        <CardHeader>
          <CardTitle>dbt Projects</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.dbt_projects_count}</div>
          <p className="text-xs text-muted-foreground">Connected Projects</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Total Models</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.total_models_count}</div>
          <p className="text-xs text-muted-foreground">
            Models available for use
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Knowledge Base</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{stats.knowledge_base_count}</div>
          <p className="text-xs text-muted-foreground">
            Models in knowledge base
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
