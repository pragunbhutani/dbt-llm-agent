"use client";

import React from "react";
import { useParams } from "next/navigation";
import useSWR from "swr";
import PageLayout from "@/components/layout/page-layout";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { SuspenseWrapper } from "@/components/suspense-wrapper";
import { useAuth } from "@/lib/useAuth";
import { fetcher } from "@/utils/fetcher";

interface ModelDetail {
  id: string;
  name: string;
  yml_description: string | null;
  yml_columns: Record<string, any> | null;
  interpreted_description: string | null;
  interpreted_columns: Record<string, string> | null;
}

function ModelDocumentation({ model }: { model: ModelDetail }) {
  const renderColumnsTable = (columns: Record<string, any> | any[]) => (
    <div className="border rounded-lg overflow-hidden mt-4">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Column
            </th>
            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Description
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {(Array.isArray(columns) ? columns : Object.entries(columns)).map(
            (entry: any, index: number) => {
              const [colName, details] = Array.isArray(columns)
                ? [entry?.name || `col_${index}`, entry]
                : entry;

              const description =
                typeof details === "string"
                  ? details
                  : (details as any)?.description || "";

              return (
                <tr key={colName}>
                  <td className="px-4 py-2 whitespace-nowrap text-sm font-medium text-gray-900">
                    {colName}
                  </td>
                  <td className="px-4 py-2 whitespace-pre-wrap text-sm text-gray-700">
                    {description}
                  </td>
                </tr>
              );
            }
          )}
        </tbody>
      </table>
    </div>
  );

  return (
    <div className="space-y-8">
      {/* YML Documentation */}
      <section>
        <h2 className="text-lg font-semibold mb-2">YML Documentation</h2>
        {model.yml_description ? (
          <p className="text-gray-800 whitespace-pre-wrap">
            {model.yml_description}
          </p>
        ) : (
          <p className="text-gray-500 italic">No YML description available.</p>
        )}
        {model.yml_columns &&
          Object.keys(model.yml_columns).length > 0 &&
          renderColumnsTable(model.yml_columns)}
      </section>

      {/* Interpreted Documentation */}
      <section>
        <h2 className="text-lg font-semibold mb-2">
          Interpreted Documentation
        </h2>
        {model.interpreted_description ? (
          <p className="text-gray-800 whitespace-pre-wrap">
            {model.interpreted_description}
          </p>
        ) : (
          <p className="text-gray-500 italic">
            No interpreted description available.
          </p>
        )}
        {model.interpreted_columns &&
          Object.keys(model.interpreted_columns).length > 0 &&
          renderColumnsTable(model.interpreted_columns)}
      </section>
    </div>
  );
}

export default function ModelDetailsPage() {
  const params = useParams();
  const { accessToken, isAuthenticated } = useAuth();
  const modelId = params.id as string;

  // Only render the content when authentication is ready
  const shouldRender = isAuthenticated && accessToken && modelId;

  return (
    <PageLayout title="Model Details" subtitle="Model Documentation">
      {shouldRender ? (
        <SuspenseWrapper loadingText="Loading model documentation...">
          <ModelDetailsContent
            modelId={modelId}
            accessToken={accessToken as string}
          />
        </SuspenseWrapper>
      ) : (
        <div className="p-4">Loading authentication...</div>
      )}
    </PageLayout>
  );
}

function ModelDetailsContent({
  modelId,
  accessToken,
}: {
  modelId: string;
  accessToken: string;
}) {
  const { data: model, error } = useSWR<ModelDetail>(
    `/api/knowledge_base/models/${modelId}/`,
    (url: string) => fetcher(url, accessToken),
    { suspense: true }
  );

  if (error) {
    return (
      <div className="p-4 text-red-600">Failed to load model details.</div>
    );
  }

  if (!model) {
    return <div className="p-4">Loading model details...</div>;
  }

  return (
    <>
      <Breadcrumb
        items={[
          { label: "Knowledge Base", href: "/dashboard/knowledge-base" },
          { label: model.name },
        ]}
        className="mb-4"
      />
      <ModelDocumentation model={model} />
    </>
  );
}
