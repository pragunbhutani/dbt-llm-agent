import React from "react";
import ConversationsTable from "@/components/conversations/conversations-table";
import PageLayout from "@/components/layout/page-layout";
import { SuspenseWrapper } from "@/components/suspense-wrapper";

export default function PastConversationsPage() {
  return (
    <PageLayout
      title="Past Conversations"
      subtitle="View and manage your past conversations and interactions."
    >
      <SuspenseWrapper loadingText="Loading conversations...">
        <ConversationsTable />
      </SuspenseWrapper>
    </PageLayout>
  );
}
