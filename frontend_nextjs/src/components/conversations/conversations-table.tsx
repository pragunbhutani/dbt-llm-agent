"use client";

import { DataTable } from "@/components/data-table/data-table";
import { getColumns } from "./columns";
import useSWR from "swr";
import { useSession } from "next-auth/react";
import { fetcher } from "@/utils/fetcher";
import { useState } from "react";
import { toast } from "sonner";
import { ConversationListItem } from "@/types/conversations";

export default function ConversationsTable() {
  const { data: session } = useSession();
  const [isDeleting, setIsDeleting] = useState(false);

  const apiURL = session?.accessToken ? `/api/workflows/conversations/` : null;
  const {
    data: conversations,
    error,
    mutate,
  } = useSWR<ConversationListItem[]>(
    apiURL,
    (url: string) => fetcher(url, session?.accessToken),
    { suspense: true }
  );

  const handleDeleteConversation = async (conversationId: number) => {
    if (!session?.accessToken || !conversations || isDeleting) return;

    const confirmed = window.confirm(
      "Are you sure you want to delete this conversation? This action cannot be undone."
    );

    if (!confirmed) return;

    setIsDeleting(true);
    const originalConversations = conversations;

    // Optimistic UI update - remove the conversation
    mutate(
      conversations.filter((conv) => conv.id !== conversationId),
      false // do not revalidate immediately
    );

    try {
      await fetcher(
        `/api/workflows/conversations/${conversationId}/`,
        session.accessToken,
        { method: "DELETE" }
      );

      toast.success("Conversation deleted successfully");
      // Re-fetch data from the API to get the final state
      mutate();
    } catch (error) {
      console.error("Failed to delete conversation", error);
      toast.error("Failed to delete conversation. Please try again.");
      // Revert on error
      mutate(originalConversations, false);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleBulkDelete = async (selectedRows: ConversationListItem[]) => {
    if (!session?.accessToken || !conversations || isDeleting) return;

    const selectedConversationIds = selectedRows.map((row) => row.id);

    if (selectedConversationIds.length === 0) {
      toast.error("Please select at least one conversation to delete.");
      return;
    }

    const confirmed = window.confirm(
      `Are you sure you want to delete ${selectedConversationIds.length} conversation(s)? This action cannot be undone.`
    );

    if (!confirmed) return;

    setIsDeleting(true);
    const originalConversations = conversations;

    // Optimistic UI update - remove the selected conversations
    mutate(
      conversations.filter(
        (conv) => !selectedConversationIds.includes(conv.id)
      ),
      false
    );

    try {
      // Delete conversations one by one
      await Promise.all(
        selectedConversationIds.map((id) =>
          fetcher(`/api/workflows/conversations/${id}/`, session.accessToken, {
            method: "DELETE",
          })
        )
      );

      toast.success(
        `Deleted ${selectedConversationIds.length} conversations successfully`
      );
      // Re-fetch data from the API to get the final state
      mutate();
    } catch (error) {
      console.error("Failed to delete conversations", error);
      toast.error("Failed to delete conversations. Please try again.");
      // Revert on error
      mutate(originalConversations, false);
    } finally {
      setIsDeleting(false);
    }
  };

  if (error) return <div>Failed to load conversations</div>;
  if (!conversations) return <div>No conversations available</div>;

  const columns = getColumns({ handleDeleteConversation });
  const initialColumnVisibility = {
    trigger: false,
    channel: false,
  };

  const filterOptions = [
    { value: "initial_question", label: "Question" },
    { value: "channel", label: "Channel" },
    { value: "user_id", label: "User" },
  ];

  const bulkActions = [
    {
      key: "delete",
      label: "Delete Selected",
      variant: "destructive" as const,
    },
  ];

  const handleBulkAction = async (
    action: string,
    selectedRows: ConversationListItem[]
  ) => {
    if (action === "delete") {
      await handleBulkDelete(selectedRows);
    }
  };

  return (
    <DataTable
      columns={columns}
      data={conversations}
      initialColumnVisibility={initialColumnVisibility}
      filterOptions={filterOptions}
      bulkActions={bulkActions}
      onBulkAction={handleBulkAction}
    />
  );
}
