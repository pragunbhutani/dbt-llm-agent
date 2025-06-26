"use client";

import { DataTable } from "@/components/data-table/data-table";
import { getColumns, DbtModel } from "./columns";
import useSWR from "swr";
import { useSession } from "next-auth/react";
import { fetcher } from "@/utils/fetcher";
import { useState } from "react";
import { toast } from "sonner";

export default function DbtModelsTable() {
  const { data: session } = useSession();
  const [isProcessing, setIsProcessing] = useState(false);

  const apiURL = session?.accessToken ? `/api/knowledge_base/models/` : null;
  const {
    data: models,
    error,
    mutate,
  } = useSWR<DbtModel[]>(
    apiURL,
    (url: string) => fetcher(url, session?.accessToken),
    { suspense: true }
  );

  const handleToggleAnswering = async (
    modelId: string,
    currentStatus: DbtModel["answering_status"]
  ) => {
    if (!session?.accessToken || !models) return;

    const originalModels = models;

    // Optimistic UI update
    mutate(
      models.map((model) => {
        if (model.id !== modelId) return model;
        // If it's 'No', we assume it will be 'Training'. If 'Yes', it becomes 'No'.
        const newStatus = currentStatus === "No" ? "Training" : "No";
        return { ...model, answering_status: newStatus };
      }),
      false // do not revalidate immediately
    );

    try {
      await fetcher(
        `/api/knowledge_base/models/${modelId}/toggle-answering-status/`,
        session.accessToken,
        { method: "POST" }
      );
      // Re-fetch data from the API to get the final state
      mutate();
    } catch (error) {
      console.error("Failed to toggle answering status", error);
      // Revert on error
      mutate(originalModels, false);
    }
  };

  const handleRefreshModel = async (modelId: string) => {
    if (!session?.accessToken || !models) return;

    const originalModels = models;

    // Optimistic UI update - set to Training
    mutate(
      models.map((model) => {
        if (model.id !== modelId) return model;
        return { ...model, answering_status: "Training" as const };
      }),
      false // do not revalidate immediately
    );

    try {
      await fetcher(
        `/api/knowledge_base/models/${modelId}/refresh/`,
        session.accessToken,
        { method: "POST" }
      );
      toast.success("Model refresh started");
      // Re-fetch data from the API to get the final state
      mutate();
    } catch (error) {
      console.error("Failed to refresh model", error);
      toast.error("Failed to refresh model. Please try again.");
      // Revert on error
      mutate(originalModels, false);
    }
  };

  const handleBulkAction = async (
    action: "enable" | "disable" | "refresh",
    selectedRows: DbtModel[]
  ) => {
    if (!session?.accessToken || !models || isProcessing) return;

    const selectedModelIds = selectedRows.map((row) => row.id);

    if (selectedModelIds.length === 0) {
      toast.error("Please select at least one model to perform bulk actions.");
      return;
    }

    setIsProcessing(true);
    const originalModels = models;

    // Optimistic UI update for bulk action
    if (action === "enable" || action === "refresh") {
      mutate(
        models.map((model) => {
          if (!selectedModelIds.includes(model.id)) return model;
          // Set to Training for models being enabled or refreshed
          return { ...model, answering_status: "Training" as const };
        }),
        false
      );
    }

    try {
      let endpoint: string;
      let body: any;
      let successMessage: string;

      if (action === "refresh") {
        endpoint = `/api/knowledge_base/models/bulk-refresh/`;
        body = { model_ids: selectedModelIds };
        successMessage = `Refreshed ${selectedModelIds.length} models`;
      } else {
        endpoint = `/api/knowledge_base/models/bulk-toggle-answering-status/`;
        body = { model_ids: selectedModelIds, enable: action === "enable" };
        successMessage = `${action === "enable" ? "Enabled" : "Disabled"} ${
          selectedModelIds.length
        } models for answering`;
      }

      const response = await fetcher(endpoint, session.accessToken, {
        method: "POST",
        body,
      });

      toast.success(response?.status || successMessage);

      // Re-fetch data from the API to get the final state
      mutate();
    } catch (error) {
      console.error("Failed to perform bulk action", error);
      toast.error("Failed to update models. Please try again.");
      // Revert on error
      mutate(originalModels, false);
    } finally {
      setIsProcessing(false);
    }
  };

  // Remove loading checks since we're using suspense
  if (error) return <div>Failed to load models</div>;
  if (!models) return <div>No data available</div>;

  const columns = getColumns({ handleToggleAnswering, handleRefreshModel });
  const initialColumnVisibility = {
    path: false,
    tags: false,
  };

  return (
    <DataTable
      columns={columns}
      data={models}
      initialColumnVisibility={initialColumnVisibility}
      onBulkAction={handleBulkAction}
    />
  );
}
