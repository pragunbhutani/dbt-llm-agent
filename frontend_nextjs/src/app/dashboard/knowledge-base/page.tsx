import React from "react";
import Heading from "@/components/heading";
import DbtModelsTable from "./dbt-models-table";

export default function KnowledgeBasePage() {
  return (
    <>
      <div className="flex h-16 items-center border-b border-gray-200 px-4">
        <Heading
          title="Knowledge Base"
          subtitle="Manage your dbt models to use with RAGStar AI."
        />
      </div>
      <div className="p-4">
        <DbtModelsTable />
      </div>
    </>
  );
}
