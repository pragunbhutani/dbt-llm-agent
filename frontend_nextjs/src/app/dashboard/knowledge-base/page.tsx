import React from "react";
import DbtModelsTable from "./dbt-models-table";
import PageLayout from "@/components/layout/page-layout";
import { Breadcrumb } from "@/components/ui/breadcrumb";

export default function KnowledgeBasePage() {
  return (
    <PageLayout
      title="Knowledge Base"
      subtitle="Manage your dbt models to use with RAGStar AI."
    >
      <div className="space-y-6">
        {/* <Breadcrumb items={[{ label: "Knowledge Base" }]} className="mb-4" /> */}
        <DbtModelsTable />
      </div>
    </PageLayout>
  );
}
