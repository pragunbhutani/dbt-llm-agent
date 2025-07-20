"use client";

import React from "react";
import DbtModelsTable from "./dbt-models-table";
import PageLayout from "@/components/layout/page-layout";
import { SuspenseWrapper } from "@/components/suspense-wrapper";

export default function KnowledgeBasePage() {
  return (
    <PageLayout
      title="Knowledge Base"
      subtitle="Manage your dbt models to use with RAGStar AI."
    >
      <SuspenseWrapper loadingText="Loading dbt models...">
        <DbtModelsTable />
      </SuspenseWrapper>
    </PageLayout>
  );
}
