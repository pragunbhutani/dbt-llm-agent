"use client";

import PageLayout from "@/components/layout/page-layout";
import { SuspenseWrapper } from "@/components/suspense-wrapper";
import { SettingsContent } from "@/components/settings/settings-content";

export default function SettingsPage() {
  return (
    <PageLayout
      title="Settings"
      subtitle="Manage your workspace and LLM configurations."
    >
      <SuspenseWrapper loadingText="Loading settings...">
        <SettingsContent />
      </SuspenseWrapper>
    </PageLayout>
  );
}
