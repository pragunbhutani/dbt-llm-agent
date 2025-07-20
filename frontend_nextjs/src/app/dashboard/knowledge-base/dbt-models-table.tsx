"use client";

import useSWR from "swr";
import { useState } from "react";
import { toast } from "sonner";
import { useAuth } from "@/lib/useAuth";
import { fetcher } from "@/utils/fetcher";
import { DataTable } from "@/components/data-table/data-table";
import { getColumns, DbtModel } from "./columns";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";

export default function DbtModelsTable({
  initialProjectId,
}: {
  initialProjectId?: string;
}) {
  const { accessToken, isAuthenticated } = useAuth();
  const [isProcessing, setIsProcessing] = useState(false);
  const [projectId, setProjectId] = useState<string>(initialProjectId || "all");

  // Fetch list of projects for dropdown
  const { data: projects } = useSWR<{ id: number; name: string }[]>(
    isAuthenticated && accessToken ? "/api/data_sources/projects/" : null,
    (url: string) => fetcher(url, accessToken)
  );

  const apiPath = projectId === "all" ? "" : `?project=${projectId}`;

  const {
    data: models,
    error,
    mutate,
  } = useSWR<DbtModel[]>(
    isAuthenticated && accessToken
      ? `/api/knowledge_base/models/${apiPath}`
      : null,
    (url: string) => fetcher(url, accessToken),
    { suspense: true }
  );

  // UI component for project filter to be rendered inside the table toolbar
  const projectFilterNode =
    projects && projects.length > 0 ? (
      <div className="w-48">
        <Select value={projectId} onValueChange={setProjectId}>
          <SelectTrigger id="project-filter" className="h-9">
            <SelectValue placeholder="All projects" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All projects</SelectItem>
            {projects.map((p) => (
              <SelectItem key={p.id} value={p.id.toString()}>
                {p.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    ) : null;

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

  // Sort models to show those enabled for answering (Yes) first
  const sortedModels = [...models].sort((a, b) => {
    if (a.answering_status === "Yes" && b.answering_status !== "Yes") {
      return -1;
    }
    if (a.answering_status !== "Yes" && b.answering_status === "Yes") {
      return 1;
    }
    // If both have same status or neither is 'Yes', preserve original order (by updated_at)
    return 0;
  });

  const columns = getColumns({ handleToggleAnswering, handleRefreshModel });
  const initialColumnVisibility = {
    path: false,
    tags: false,
  };

  return (
    <DataTable
      columns={columns}
      data={sortedModels}
      initialColumnVisibility={initialColumnVisibility}
      onBulkAction={handleBulkAction}
      leadingComponents={projectFilterNode}
    />
  );
}
