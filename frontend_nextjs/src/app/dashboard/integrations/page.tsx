"use client";

import PageLayout from "@/components/layout/page-layout";
import { SuspenseWrapper } from "@/components/suspense-wrapper";
import { IntegrationsContent } from "@/components/integrations/integrations-content";

export default function IntegrationsPage() {
  return (
    <PageLayout
      title="Integrations"
      subtitle="Connect your data sources and tools to enhance ragstar's capabilities."
    >
      <SuspenseWrapper loadingText="Loading integrations...">
        <IntegrationsContent />
      </SuspenseWrapper>
    </PageLayout>
  );
}
