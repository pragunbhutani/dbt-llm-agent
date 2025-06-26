"use client";

import { Button } from "@/components/ui/button";
import { AddProjectModal } from "@/components/add-project-modal";
import PageLayout from "@/components/layout/page-layout";
import { SuspenseWrapper } from "@/components/suspense-wrapper";
import { DashboardOnboarding } from "@/components/dashboard/dashboard-onboarding";
import { DashboardStats } from "@/components/dashboard/dashboard-stats";
import { DashboardProjects } from "@/components/dashboard/dashboard-projects";

export default function DashboardPage() {
  const headerActions = (
    <AddProjectModal>
      <Button>Add dbt Project</Button>
    </AddProjectModal>
  );

  return (
    <PageLayout
      title="Dashboard"
      subtitle="Here's an overview of your Ragstar setup."
      actions={headerActions}
    >
      <SuspenseWrapper loadingText="Loading onboarding...">
        <DashboardOnboarding />
      </SuspenseWrapper>

      <SuspenseWrapper loadingText="Loading statistics...">
        <DashboardStats />
      </SuspenseWrapper>

      <SuspenseWrapper loadingText="Loading projects...">
        <DashboardProjects />
      </SuspenseWrapper>
    </PageLayout>
  );
}
