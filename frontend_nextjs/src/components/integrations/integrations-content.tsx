"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
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

export function IntegrationsContent() {
  const { data: session } = useSession();
  const router = useRouter();
  const [selectedIntegration, setSelectedIntegration] =
    useState<IntegrationStatus | null>(null);
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(false);
  const [testingIntegration, setTestingIntegration] = useState<number | null>(
    null
  );
  const [togglingIntegration, setTogglingIntegration] = useState<number | null>(
    null
  );

  // Fetch integration status with Suspense
  const {
    data: integrationStatus,
    error,
    mutate,
  } = useSWR(
    session?.accessToken
      ? "/api/integrations/organisation-integrations/status/"
      : null,
    (url) => fetcher(url, session!.accessToken as string),
    { suspense: true }
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
      } else {
        const errorData = await response.json();
        console.error("Test failed:", errorData);
      }
    } catch (error) {
      console.error("Connection test failed:", error);
    } finally {
      setTestingIntegration(null);
    }
  };

  const handleToggleIntegration = async (integrationId: number) => {
    if (!session?.accessToken) return;

    setTogglingIntegration(integrationId);

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
    } finally {
      setTogglingIntegration(null);
    }
  };

  const handleConfigureIntegration = (integration: IntegrationStatus) => {
    // For Slack, navigate to dedicated setup page
    if (integration.key === "slack") {
      router.push("/dashboard/integrations/slack");
      return;
    }

    setSelectedIntegration(integration);
    setIsConfigModalOpen(true);
  };

  const handleConfigSuccess = () => {
    mutate(); // Refresh the data after successful configuration
  };

  const handleRefreshLink = async (integrationId: number) => {
    if (!session?.accessToken) return;
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/integrations/organisation-integrations/${integrationId}/refresh_link/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${session.accessToken}`,
          },
        }
      );
      const data = await res.json();
      if (data.authorization_url) {
        window.location.href = data.authorization_url;
      }
    } catch (e) {
      console.error(e);
    }
  };

  const inboundIntegrations =
    integrationStatus?.filter(
      (i: IntegrationStatus) =>
        i.integration_type === "inbound" && i.key !== "mcp"
    ) || [];
  const outboundIntegrations =
    integrationStatus?.filter(
      (i: IntegrationStatus) => i.integration_type === "outbound"
    ) || [];

  if (error) {
    return <div>Error loading integrations. Please try again.</div>;
  }

  return (
    <>
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
                togglingIntegration={togglingIntegration}
                onRefreshLink={handleRefreshLink}
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
                togglingIntegration={togglingIntegration}
                onRefreshLink={handleRefreshLink}
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
            togglingIntegration={togglingIntegration}
            onRefreshLink={handleRefreshLink}
          />
        </TabsContent>

        <TabsContent value="outbound">
          <IntegrationGrid
            integrations={outboundIntegrations}
            onTestConnection={handleTestConnection}
            onToggle={handleToggleIntegration}
            onConfigure={handleConfigureIntegration}
            testingIntegration={testingIntegration}
            togglingIntegration={togglingIntegration}
            onRefreshLink={handleRefreshLink}
          />
        </TabsContent>
      </Tabs>
    </>
  );
}

// Helper components
function IntegrationGrid({
  integrations,
  onTestConnection,
  onToggle,
  onConfigure,
  testingIntegration,
  togglingIntegration,
  onRefreshLink,
}: {
  integrations: IntegrationStatus[];
  onTestConnection: (id: number) => void;
  onToggle: (id: number) => void;
  onConfigure: (integration: IntegrationStatus) => void;
  testingIntegration: number | null;
  togglingIntegration: number | null;
  onRefreshLink: (id: number) => void;
}) {
  const getEnabledBadgeColor = (enabled: boolean) => {
    return enabled
      ? "bg-green-100 text-green-800 border-green-200"
      : "bg-gray-100 text-gray-800 border-gray-200";
  };

  const getEnabledBadgeText = (enabled: boolean) =>
    enabled ? "Enabled" : "Disabled";

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {integrations.map((integration) => {
        const isToggling =
          togglingIntegration !== null &&
          togglingIntegration === integration.id;

        return (
          <Card key={integration.key} className="relative">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{integration.name}</CardTitle>
                <Badge
                  variant="outline"
                  className={`text-xs ${getEnabledBadgeColor(
                    integration.is_enabled
                  )}`}
                >
                  {getEnabledBadgeText(integration.is_enabled)}
                </Badge>
                {integration.key === "github" && integration.is_enabled && (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => onRefreshLink(integration.id!)}
                  >
                    Refresh
                  </Button>
                )}
              </div>
              <CardDescription className="text-sm">
                {getIntegrationDescription(integration.key)}
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Configured:</span>
                <span
                  className={
                    integration.is_configured
                      ? "text-green-600"
                      : "text-gray-400"
                  }
                >
                  {integration.is_configured ? "Yes" : "No"}
                </span>
              </div>

              <div className="flex gap-2 pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  disabled={integration.id === null || isToggling}
                  onClick={() => integration.id && onToggle(integration.id)}
                >
                  {isToggling ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      {integration.is_enabled ? "Disabling..." : "Enabling..."}
                    </>
                  ) : (
                    <>{integration.is_enabled ? "Disable" : "Enable"}</>
                  )}
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
        );
      })}
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
