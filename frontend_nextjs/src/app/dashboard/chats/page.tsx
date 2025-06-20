import React from "react";
import Heading from "@/components/heading";
import ConversationsTable from "@/components/conversations/conversations-table";

export default function PastConversationsPage() {
  return (
    <>
      <div className="flex h-16 items-center border-b border-gray-200 px-4">
        <Heading
          title="Past Conversations"
          subtitle="View and manage your past conversations and interactions."
        />
      </div>
      <div className="p-4">
        <ConversationsTable />
      </div>
    </>
  );
}
