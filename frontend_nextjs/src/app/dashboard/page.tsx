"use client";

import Heading from "@/components/heading";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { AddProjectModal } from "@/components/add-project-modal";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import useSWR from "swr";
import { useSession } from "next-auth/react";
import { CheckCircle2, Circle, Loader2 } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { fetcher } from "@/utils/fetcher";

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

export default function DashboardPage() {
  const { data: session } = useSession();

  const { data: statsData, error: statsError } = useSWR(
    session?.accessToken ? "/api/data_sources/dashboard-stats/" : null,
    (url: string) => fetcher(url, session?.accessToken)
  );
  const {
    data: projectsData,
    error: projectsError,
    mutate: mutateProjects,
  } = useSWR(
    session?.accessToken ? "/api/data_sources/projects/" : null,
    (url: string) => fetcher(url, session?.accessToken)
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

  const projects = projectsData || [];
  const isLoadingStats = !statsData && !statsError && !!session?.accessToken;
  const isLoadingProjects =
    !projectsData && !projectsError && !!session?.accessToken;

  const [actioningProject, setActioningProject] = useState<{
    id: string;
    action: "refresh" | "delete";
  } | null>(null);

  const handleRefresh = async (projectId: string) => {
    if (!session?.accessToken) return;
    setActioningProject({ id: projectId, action: "refresh" });
    try {
      await fetcher(
        `/api/data_sources/projects/${projectId}/refresh/`,
        session.accessToken,
        { method: "POST" }
      );
      mutateProjects();
    } catch (error) {
      console.error("Failed to refresh project", error);
      // TODO: Add user-facing error feedback
    } finally {
      setActioningProject(null);
    }
  };

  const handleDelete = async (projectId: string) => {
    if (!session?.accessToken) return;
    setActioningProject({ id: projectId, action: "delete" });
    try {
      await fetcher(
        `/api/data_sources/projects/${projectId}/`,
        session.accessToken,
        { method: "DELETE" }
      );
      await mutateProjects();
    } catch (error) {
      console.error("Failed to delete project", error);
      // TODO: Add user-facing error feedback
    } finally {
      setActioningProject(null);
    }
  };

  return (
    <>
      <div className="flex h-16 items-center justify-between border-b border-gray-200 px-4">
        <Heading
          title="Dashboard"
          subtitle="Here's an overview of your Ragstar setup."
        />
        <AddProjectModal>
          <Button>Add dbt Project</Button>
        </AddProjectModal>
      </div>

      <div className="p-4">
        <Card className="mb-4">
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="item-1" className="border-none">
              <AccordionTrigger className="px-6 py-2 hover:no-underline">
                <div className="text-left">
                  <h3 className="text-lg font-semibold">Getting Started</h3>
                  <p className="text-sm text-muted-foreground">
                    Complete these steps to get your AI data analyst up and
                    running.
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
                            isComplete={
                              stats.onboarding_steps.connect_dbt_project
                            }
                          >
                            Connect a dbt project
                          </OnboardingStep>
                        </li>
                        <li>
                          <OnboardingStep
                            isComplete={
                              stats.onboarding_steps.train_knowledge_base
                            }
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
                            isComplete={
                              stats.onboarding_steps.ask_first_question
                            }
                          >
                            Ask your first question
                          </OnboardingStep>
                        </li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">
                        Optional Next Steps
                      </h4>
                      <p className="text-sm text-muted-foreground mb-2">
                        Further enhance your Ragstar experience with these
                        optional integrations.
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

        {/* Stat Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-4">
          <Card>
            <CardHeader>
              <CardTitle>dbt Projects</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {isLoadingStats ? (
                  <Loader2 className="h-6 w-6 animate-spin" />
                ) : (
                  stats.dbt_projects_count
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                Connected Projects
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Total Models</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {isLoadingStats ? (
                  <Loader2 className="h-6 w-6 animate-spin" />
                ) : (
                  stats.total_models_count
                )}
              </div>
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
              <div className="text-2xl font-bold">
                {isLoadingStats ? (
                  <Loader2 className="h-6 w-6 animate-spin" />
                ) : (
                  stats.knowledge_base_count
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                Models in knowledge base
              </p>
            </CardContent>
          </Card>
        </div>

        {/* dbt Projects Table */}
        <Card>
          <CardHeader>
            <CardTitle>Connected dbt Projects</CardTitle>
            <CardDescription>
              Manage your connected dbt projects and their status.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Project Name</TableHead>
                  <TableHead>Connection Type</TableHead>
                  <TableHead>Models Synced</TableHead>
                  <TableHead>Last Refreshed</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoadingProjects ? (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center">
                      <Loader2 className="h-6 w-6 animate-spin mx-auto" />
                    </TableCell>
                  </TableRow>
                ) : projects.length > 0 ? (
                  projects.map((project: any) => (
                    <TableRow key={project.id}>
                      <TableCell className="font-medium">
                        {project.name}
                      </TableCell>
                      <TableCell>{project.connection_type}</TableCell>
                      <TableCell>N/A</TableCell>
                      <TableCell>
                        {new Date(project.updated_at).toLocaleDateString()}
                      </TableCell>
                      <TableCell>
                        <div className="flex space-x-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleRefresh(project.id)}
                            disabled={actioningProject?.id === project.id}
                          >
                            {actioningProject?.id === project.id &&
                            actioningProject?.action === "refresh" ? (
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            ) : null}
                            Refresh
                          </Button>
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => handleDelete(project.id)}
                            disabled={actioningProject?.id === project.id}
                          >
                            {actioningProject?.id === project.id &&
                            actioningProject?.action === "delete" ? (
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            ) : null}
                            Delete
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center">
                      No dbt projects found.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
