"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import PageLayout from "@/components/layout/page-layout";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { IntegrationConfigModal } from "@/components/integrations/integration-config-modal";
import { fetcher } from "@/utils/fetcher";
import useSWR from "swr";

interface Integration {
  key: string;
  name: string;
  description: string;
  integration_type: "inbound" | "outbound";
  icon_url?: string;
  documentation_url?: string;
  is_active: boolean;
}

interface IntegrationStatus {
  id: number | null;
  key: string;
  name: string;
  integration_type: "inbound" | "outbound";
  is_enabled: boolean;
  is_configured: boolean;
  connection_status: "connected" | "error" | "not_configured" | "unknown";
  last_tested_at?: string;
  tools_count: number;
}

export default function IntegrationsPage() {
  const { data: session } = useSession();
  const router = useRouter();
  const [selectedIntegration, setSelectedIntegration] =
    useState<IntegrationStatus | null>(null);
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(false);

  // Fetch integration status
  const {
    data: integrationStatus,
    error,
    mutate,
  } = useSWR(
    session?.accessToken
      ? "/api/integrations/organisation-integrations/status/"
      : null,
    (url) => fetcher(url, session!.accessToken as string)
  );

  const [testingIntegration, setTestingIntegration] = useState<number | null>(
    null
  );

  const handleTestConnection = async (integrationId: number) => {
    if (!session?.accessToken) return;

    setTestingIntegration(integrationId);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/integrations/organisation-integrations/${integrationId}/test_connection/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${session.accessToken}`,
            "Content-Type": "application/json",
          },
        }
      );

      if (response.ok) {
        const result = await response.json();
        console.log("Test connection result:", result);
        mutate(); // Refresh the data

        // Show success message
        // TODO: Add toast notification here
      } else {
        const errorData = await response.json();
        console.error("Test failed:", errorData);
        // TODO: Add error toast notification here
      }
    } catch (error) {
      console.error("Connection test failed:", error);
      // TODO: Add error toast notification here
    } finally {
      setTestingIntegration(null);
    }
  };

  const handleToggleIntegration = async (integrationId: number) => {
    if (!session?.accessToken) return;

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/integrations/organisation-integrations/${integrationId}/toggle_enable/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${session.accessToken}`,
            "Content-Type": "application/json",
          },
        }
      );

      if (response.ok) {
        mutate(); // Refresh the data
      }
    } catch (error) {
      console.error("Toggle failed:", error);
    }
  };

  const handleConfigureIntegration = (integration: IntegrationStatus) => {
    // For Slack, navigate to dedicated setup page
    if (integration.key === "slack") {
      router.push("/dashboard/integrations/slack");
      return;
    }

    // For MCP, navigate to dedicated setup page
    if (integration.key === "mcp") {
      router.push("/dashboard/integrations/mcp");
      return;
    }

    setSelectedIntegration(integration);
    setIsConfigModalOpen(true);
  };

  const handleConfigSuccess = () => {
    mutate(); // Refresh the data after successful configuration
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "connected":
        return "bg-green-100 text-green-800 border-green-200";
      case "error":
        return "bg-red-100 text-red-800 border-red-200";
      case "not_configured":
        return "bg-gray-100 text-gray-800 border-gray-200";
      default:
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case "connected":
        return "Connected";
      case "error":
        return "Error";
      case "not_configured":
        return "Not Configured";
      default:
        return "Unknown";
    }
  };

  const inboundIntegrations =
    integrationStatus?.filter(
      (i: IntegrationStatus) => i.integration_type === "inbound"
    ) || [];
  const outboundIntegrations =
    integrationStatus?.filter(
      (i: IntegrationStatus) => i.integration_type === "outbound"
    ) || [];

  if (error) {
    return (
      <PageLayout
        title="Integrations"
        subtitle="Error loading integrations. Please try again."
      >
        <div className="p-4" />
      </PageLayout>
    );
  }

  return (
    <PageLayout
      title="Integrations"
      subtitle="Connect your data sources and tools to enhance ragstar's capabilities."
    >
      {selectedIntegration && (
        <IntegrationConfigModal
          integration={selectedIntegration}
          isOpen={isConfigModalOpen}
          onClose={() => {
            setIsConfigModalOpen(false);
            setSelectedIntegration(null);
          }}
          onSuccess={handleConfigSuccess}
        />
      )}

      <div className="">
        {/* <Breadcrumb items={[{ label: "Integrations" }]} className="mb-4" /> */}

        <Tabs defaultValue="all" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="all">All Integrations</TabsTrigger>
            <TabsTrigger value="inbound">
              Inbound ({inboundIntegrations.length})
            </TabsTrigger>
            <TabsTrigger value="outbound">
              Outbound ({outboundIntegrations.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="all" className="space-y-6 px-2">
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Inbound Integrations
                </h3>
                <p className="text-sm text-gray-600 mb-4">
                  Services that send requests to ragstar (e.g., Slack, MCP
                  clients)
                </p>
                <IntegrationGrid
                  integrations={inboundIntegrations}
                  onTestConnection={handleTestConnection}
                  onToggle={handleToggleIntegration}
                  onConfigure={handleConfigureIntegration}
                  testingIntegration={testingIntegration}
                />
              </div>

              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Outbound Integrations
                </h3>
                <p className="text-sm text-gray-600 mb-4">
                  Services that ragstar connects to for data and functionality
                  (e.g., Snowflake, Metabase)
                </p>
                <IntegrationGrid
                  integrations={outboundIntegrations}
                  onTestConnection={handleTestConnection}
                  onToggle={handleToggleIntegration}
                  onConfigure={handleConfigureIntegration}
                  testingIntegration={testingIntegration}
                />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="inbound">
            <IntegrationGrid
              integrations={inboundIntegrations}
              onTestConnection={handleTestConnection}
              onToggle={handleToggleIntegration}
              onConfigure={handleConfigureIntegration}
              testingIntegration={testingIntegration}
            />
          </TabsContent>

          <TabsContent value="outbound">
            <IntegrationGrid
              integrations={outboundIntegrations}
              onTestConnection={handleTestConnection}
              onToggle={handleToggleIntegration}
              onConfigure={handleConfigureIntegration}
              testingIntegration={testingIntegration}
            />
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}

function IntegrationGrid({
  integrations,
  onTestConnection,
  onToggle,
  onConfigure,
  testingIntegration,
}: {
  integrations: IntegrationStatus[];
  onTestConnection: (id: number) => void;
  onToggle: (id: number) => void;
  onConfigure: (integration: IntegrationStatus) => void;
  testingIntegration: number | null;
}) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "connected":
        return "bg-green-100 text-green-800 border-green-200";
      case "error":
        return "bg-red-100 text-red-800 border-red-200";
      case "not_configured":
        return "bg-gray-100 text-gray-800 border-gray-200";
      default:
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case "connected":
        return "Connected";
      case "error":
        return "Error";
      case "not_configured":
        return "Not Configured";
      default:
        return "Unknown";
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {integrations.map((integration) => (
        <Card key={integration.key} className="relative">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">{integration.name}</CardTitle>
              <Badge
                variant="outline"
                className={`text-xs ${getStatusColor(
                  integration.connection_status
                )}`}
              >
                {getStatusText(integration.connection_status)}
              </Badge>
            </div>
            <CardDescription className="text-sm">
              {getIntegrationDescription(integration.key)}
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Status:</span>
              <span
                className={
                  integration.is_enabled ? "text-green-600" : "text-gray-400"
                }
              >
                {integration.is_enabled ? "Enabled" : "Disabled"}
              </span>
            </div>

            <div className="flex gap-2 pt-2">
              <Button
                variant="outline"
                size="sm"
                className="flex-1"
                disabled={integration.id === null}
                onClick={() => integration.id && onToggle(integration.id)}
              >
                {integration.is_enabled ? "Disable" : "Enable"}
              </Button>

              {integration.is_configured && integration.id && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onTestConnection(integration.id!)}
                  disabled={testingIntegration === integration.id}
                >
                  {testingIntegration === integration.id ? (
                    <>
                      <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                      Testing
                    </>
                  ) : (
                    "Test"
                  )}
                </Button>
              )}

              <Button
                variant="default"
                size="sm"
                onClick={() => onConfigure(integration)}
              >
                {integration.key === "slack" ? "Setup" : "Configure"}
              </Button>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function getIntegrationDescription(key: string): string {
  const descriptions: Record<string, string> = {
    slack:
      "Connect your Slack workspace to interact with ragstar through mentions and DMs.",
    snowflake:
      "Connect to your Snowflake data warehouse to run queries and analyze data.",
    metabase:
      "Integrate with Metabase to create dashboards and visualizations.",
    mcp: "Expose ragstar as a remote MCP server for Claude, OpenAI, and other LLM applications.",
  };

  return (
    descriptions[key] ||
    "Third-party integration to extend ragstar's capabilities."
  );
}
