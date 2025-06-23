import React from "react";
import ConversationsTable from "@/components/conversations/conversations-table";
import PageLayout from "@/components/layout/page-layout";
import { Breadcrumb } from "@/components/ui/breadcrumb";

export default function PastConversationsPage() {
  return (
    <PageLayout
      title="Past Conversations"
      subtitle="View and manage your past conversations and interactions."
    >
      <div className="space-y-6">
        <Breadcrumb items={[{ label: "Conversations" }]} className="mb-4" />
        <ConversationsTable />
      </div>
    </PageLayout>
  );
}
