"use client";

import useSWR from "swr";
import { useState } from "react";
import { toast } from "sonner";
import { useAuth } from "@/lib/useAuth";
import { fetcher } from "@/utils/fetcher";
import { DataTable } from "@/components/data-table/data-table";
import { getColumns, DbtModel } from "./columns";

export default function DbtModelsTable() {
  const { accessToken, isAuthenticated } = useAuth();
  const [isProcessing, setIsProcessing] = useState(false);

  const {
    data: models,
    error,
    mutate,
  } = useSWR<DbtModel[]>(
    isAuthenticated && accessToken ? "/api/knowledge_base/models/" : null,
    (url: string) => fetcher(url, accessToken),
    { suspense: true }
  );

  const handleToggleAnswering = async (
    modelId: string,
    currentStatus: DbtModel["answering_status"]
  ) => {
    if (!accessToken || !models) return;

    const originalModels = models;
    const updatedModels = models.map((model) =>
      model.id === modelId
        ? { ...model, answering_status: "Training" as const }
        : model
    );

    try {
      // Optimistically update the UI
      mutate(updatedModels, false);

      await fetcher(
        `/api/knowledge_base/models/${modelId}/toggle-answering-status/`,
        accessToken,
        { method: "POST" }
      );

      // Refresh to get the actual status from server
      mutate();
      toast.success("Model status updated successfully");
    } catch (error) {
      // Revert on error
      mutate(originalModels, false);
      toast.error("Failed to update model status");
      console.error("Error toggling answering status:", error);
    }
  };

  const handleRefreshModel = async (modelId: string) => {
    if (!accessToken || !models) return;

    const originalModels = models;
    const updatedModels = models.map((model) =>
      model.id === modelId
        ? { ...model, answering_status: "Training" as const }
        : model
    );

    try {
      // Optimistically update the UI
      mutate(updatedModels, false);

      await fetcher(
        `/api/knowledge_base/models/${modelId}/refresh/`,
        accessToken,
        { method: "POST" }
      );

      // Refresh to get the actual status from server
      mutate();
      toast.success("Model refreshed successfully");
    } catch (error) {
      // Revert on error
      mutate(originalModels, false);
      toast.error("Failed to refresh model");
      console.error("Error refreshing model:", error);
    }
  };

  const handleBulkAction = async (action: string, selectedRows: DbtModel[]) => {
    if (!accessToken || !models || isProcessing) return;

    const selectedModelIds = selectedRows.map((row) => row.id);
    setIsProcessing(true);

    try {
      let endpoint = "";
      let body = {};

      if (action === "enable" || action === "disable") {
        endpoint = "/api/knowledge_base/models/bulk-toggle-answering/";
        body = {
          model_ids: selectedModelIds,
          enable: action === "enable",
        };
      } else if (action === "refresh") {
        endpoint = "/api/knowledge_base/models/bulk-refresh/";
        body = {
          model_ids: selectedModelIds,
        };
      }

      const response = await fetcher(endpoint, accessToken, {
        method: "POST",
        body,
      });

      mutate();
      toast.success(`${action} action completed successfully`);
    } catch (error) {
      toast.error(`Failed to ${action} selected models`);
      console.error(`Error during bulk ${action}:`, error);
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
