"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import {
  Copy,
  CheckCircle,
  AlertCircle,
  Loader2,
  ExternalLink,
  Server,
  Shield,
  Zap,
} from "lucide-react";
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
import { Alert, AlertDescription } from "@/components/ui/alert";
import { toast } from "sonner";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import PageLayout from "@/components/layout/page-layout";

interface MCPIntegrationStatus {
  id: number | null;
  is_enabled: boolean;
  is_configured: boolean;
  connection_status: "connected" | "error" | "not_configured" | "unknown";
  configuration: {
    server_url?: string;
    auth_provider?: string;
  };
  credentials: {
    client_count?: number;
    active_connections?: number;
  };
  last_tested_at?: string;
}

export default function MCPIntegrationPage() {
  const { data: session } = useSession();
  const [mcpStatus, setMcpStatus] = useState<MCPIntegrationStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"setup" | "claude" | "openai">(
    "setup"
  );

  // Get URLs
  const getMCPServerUrl = () => {
    if (typeof window !== "undefined") {
      const protocol = window.location.protocol;
      const hostname = window.location.hostname;
      const port = window.location.port;
      const baseUrl = `${protocol}//${hostname}${port ? `:${port}` : ""}`;
      return `${baseUrl}/mcp`;
    }
    return "https://your-ragstar-domain.com/mcp";
  };

  const getAuthServerUrl = () => {
    if (typeof window !== "undefined") {
      const protocol = window.location.protocol;
      const hostname = window.location.hostname;
      const port = window.location.port;
      return `${protocol}//${hostname}${port ? `:${port}` : ""}`;
    }
    return "https://your-ragstar-domain.com";
  };

  const mcpServerUrl = getMCPServerUrl();
  const authServerUrl = getAuthServerUrl();
  const metadataUrl = `${authServerUrl}/.well-known/oauth-authorization-server`;

  // Copy to clipboard function
  const copyToClipboard = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
      toast.success("Copied to clipboard!");
      setTimeout(() => setCopied(null), 2000);
    } catch (error) {
      toast.error("Failed to copy to clipboard");
    }
  };

  // Fetch MCP integration status
  const fetchMCPStatus = async () => {
    if (!session?.accessToken) return;

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/integrations/organisation-integrations/status/`,
        {
          headers: {
            Authorization: `Bearer ${session.accessToken}`,
          },
        }
      );

      if (response.ok) {
        const integrations = await response.json();
        const mcpIntegration = integrations.find((i: any) => i.key === "mcp");
        setMcpStatus(mcpIntegration || null);
      }
    } catch (error) {
      console.error("Failed to fetch MCP status:", error);
    }
  };

  // Enable MCP server
  const handleEnableMCP = async () => {
    if (!session?.accessToken) return;

    setIsLoading(true);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/integrations/organisation-integrations/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${session.accessToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            integration_key: "mcp",
            is_enabled: true,
            configuration: {
              server_url: mcpServerUrl,
              auth_provider: "oauth2",
            },
          }),
        }
      );

      if (response.ok) {
        toast.success("MCP Server enabled successfully!");
        fetchMCPStatus();
        setActiveTab("claude");
      } else {
        const errorData = await response.json();
        toast.error(errorData.error || "Failed to enable MCP server");
      }
    } catch (error) {
      toast.error("Failed to enable MCP server");
    } finally {
      setIsLoading(false);
    }
  };

  // Disable MCP server
  const handleDisableMCP = async () => {
    if (!session?.accessToken || !mcpStatus?.id) return;

    setIsLoading(true);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/integrations/organisation-integrations/${mcpStatus.id}/toggle_enable/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${session.accessToken}`,
          },
        }
      );

      if (response.ok) {
        toast.success("MCP Server disabled");
        fetchMCPStatus();
      } else {
        toast.error("Failed to disable MCP server");
      }
    } catch (error) {
      toast.error("Failed to disable MCP server");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchMCPStatus();
  }, [session]);

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
        return "Active";
      case "error":
        return "Error";
      case "not_configured":
        return "Not Configured";
      default:
        return "Unknown";
    }
  };

  const headerActions = (
    <div className="flex items-center gap-3">
      {mcpStatus && (
        <Badge
          variant="outline"
          className={`text-xs ${getStatusColor(mcpStatus.connection_status)}`}
        >
          {getStatusText(mcpStatus.connection_status)}
        </Badge>
      )}
      {mcpStatus?.is_enabled && (
        <Button
          variant="outline"
          onClick={handleDisableMCP}
          disabled={isLoading}
          size="sm"
        >
          {isLoading && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
          Disable MCP Server
        </Button>
      )}
    </div>
  );

  return (
    <PageLayout
      title="MCP Server Setup"
      subtitle="Enable your ragstar instance as a remote MCP server for AI applications"
      actions={headerActions}
    >
      <div className="space-y-6">
        {/* Breadcrumbs */}
        <Breadcrumb
          items={[
            { label: "Integrations", href: "/dashboard/integrations" },
            { label: "MCP Server" },
          ]}
          className="mb-4"
        />

        {/* Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Server className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    Server Status
                  </p>
                  <p className="text-xs text-gray-600">
                    {mcpStatus?.is_enabled ? "Active" : "Disabled"}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-green-100 rounded-lg">
                  <Shield className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">Security</p>
                  <p className="text-xs text-gray-600">OAuth 2.1 + PKCE</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <Zap className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    Connections
                  </p>
                  <p className="text-xs text-gray-600">
                    {mcpStatus?.credentials?.active_connections || 0} active
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content */}
        <Tabs
          value={activeTab}
          onValueChange={(value) => setActiveTab(value as any)}
        >
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="setup">Setup</TabsTrigger>
            <TabsTrigger value="claude" disabled={!mcpStatus?.is_enabled}>
              Claude.ai
            </TabsTrigger>
            <TabsTrigger value="openai" disabled={!mcpStatus?.is_enabled}>
              OpenAI
            </TabsTrigger>
          </TabsList>

          <TabsContent value="setup" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>What is MCP?</CardTitle>
                <CardDescription>
                  The Model Context Protocol (MCP) allows AI applications to
                  securely connect to your ragstar instance
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-medium text-blue-900 mb-2">
                    How it works:
                  </h4>
                  <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
                    <li>
                      AI applications (like Claude.ai) can discover and connect
                      to your ragstar MCP server
                    </li>
                    <li>
                      Users authorize the connection through secure OAuth 2.1
                      authentication
                    </li>
                    <li>
                      AI can then access your dbt models, data sources, and
                      analytics tools
                    </li>
                    <li>
                      All data remains in your control - you can revoke access
                      anytime
                    </li>
                  </ul>
                </div>

                {!mcpStatus?.is_enabled ? (
                  <div className="space-y-4">
                    <Alert>
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        Your MCP server is currently disabled. Enable it to
                        allow AI applications to connect.
                      </AlertDescription>
                    </Alert>

                    <div className="flex justify-center">
                      <Button
                        onClick={handleEnableMCP}
                        disabled={isLoading}
                        size="lg"
                      >
                        {isLoading && (
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        )}
                        Enable MCP Server
                      </Button>
                    </div>
                  </div>
                ) : (
                  <Alert>
                    <CheckCircle className="h-4 w-4" />
                    <AlertDescription>
                      MCP Server is active! You can now connect AI applications
                      using the instructions in the tabs above.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {mcpStatus?.is_enabled && (
              <Card>
                <CardHeader>
                  <CardTitle>Server Configuration</CardTitle>
                  <CardDescription>
                    Connection details for your ragstar MCP server
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      MCP Server URL
                    </label>
                    <div className="flex items-center space-x-2">
                      <code className="flex-1 px-3 py-2 bg-gray-50 border border-gray-200 rounded-md text-sm font-mono">
                        {mcpServerUrl}
                      </code>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() =>
                          copyToClipboard(mcpServerUrl, "server-url")
                        }
                      >
                        <Copy className="h-4 w-4 mr-2" />
                        {copied === "server-url" ? "Copied!" : "Copy"}
                      </Button>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      OAuth Discovery URL
                    </label>
                    <div className="flex items-center space-x-2">
                      <code className="flex-1 px-3 py-2 bg-gray-50 border border-gray-200 rounded-md text-sm font-mono">
                        {metadataUrl}
                      </code>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() =>
                          copyToClipboard(metadataUrl, "metadata-url")
                        }
                      >
                        <Copy className="h-4 w-4 mr-2" />
                        {copied === "metadata-url" ? "Copied!" : "Copy"}
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="claude" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <div className="w-6 h-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded"></div>
                  Connect Claude.ai
                </CardTitle>
                <CardDescription>
                  Step-by-step instructions to connect Claude.ai to your ragstar
                  MCP server
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <h4 className="font-medium text-purple-900 mb-3">
                      Connection Steps:
                    </h4>
                    <ol className="text-sm text-purple-800 space-y-3 list-decimal list-inside">
                      <li>
                        <strong>Open Claude.ai</strong> in your web browser and
                        sign in to your account
                      </li>
                      <li>
                        <strong>Navigate to Settings</strong> →{" "}
                        <strong>Integrations</strong> or{" "}
                        <strong>Connected Apps</strong> section
                      </li>
                      <li>
                        <strong>Add New Integration</strong> and select{" "}
                        <strong>MCP Server</strong> or{" "}
                        <strong>Remote Server</strong>
                      </li>
                      <li>
                        <strong>Enter Server URL:</strong>
                        <div className="mt-2 flex items-center space-x-2">
                          <code className="flex-1 px-2 py-1 bg-purple-100 rounded text-xs font-mono">
                            {mcpServerUrl}
                          </code>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() =>
                              copyToClipboard(mcpServerUrl, "claude-url")
                            }
                          >
                            <Copy className="h-3 w-3" />
                          </Button>
                        </div>
                      </li>
                      <li>
                        <strong>Authorize Connection:</strong> Claude will
                        redirect you to ragstar for authentication
                      </li>
                      <li>
                        <strong>Sign in to ragstar</strong> and grant Claude.ai
                        the requested permissions
                      </li>
                      <li>
                        <strong>Complete Setup:</strong> Return to Claude.ai -
                        ragstar tools should now be available!
                      </li>
                    </ol>
                  </div>

                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 className="font-medium text-blue-900 mb-2">
                      💡 Usage Tips:
                    </h4>
                    <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
                      <li>
                        Ask Claude: "What dbt models do I have?" to see your
                        analytics models
                      </li>
                      <li>
                        Try: "Show me my data sources" to explore connected
                        databases
                      </li>
                      <li>
                        Query: "Help me understand this dbt model" and reference
                        specific models
                      </li>
                      <li>
                        Use: "Generate SQL for my data question" to get query
                        assistance
                      </li>
                    </ul>
                  </div>

                  <Alert>
                    <ExternalLink className="h-4 w-4" />
                    <AlertDescription>
                      If you don't see MCP/Remote Server options in Claude.ai,
                      make sure you're using a Claude.ai plan that supports
                      integrations, or check Anthropic's documentation for
                      updates.
                    </AlertDescription>
                  </Alert>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="openai" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <div className="w-6 h-6 bg-black rounded flex items-center justify-center">
                    <div className="w-3 h-3 bg-white rounded-full"></div>
                  </div>
                  Connect OpenAI
                </CardTitle>
                <CardDescription>
                  Instructions for connecting OpenAI applications to your
                  ragstar MCP server
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      OpenAI's MCP support is still evolving. These instructions
                      apply to MCP-compatible OpenAI applications.
                    </AlertDescription>
                  </Alert>

                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <h4 className="font-medium text-green-900 mb-3">
                      Connection Steps:
                    </h4>
                    <ol className="text-sm text-green-800 space-y-3 list-decimal list-inside">
                      <li>
                        <strong>Install MCP-compatible client</strong> such as
                        OpenAI Desktop app with MCP support
                      </li>
                      <li>
                        <strong>Open the application</strong> and navigate to
                        Settings → Integrations or Extensions
                      </li>
                      <li>
                        <strong>Add Remote MCP Server</strong> or similar option
                      </li>
                      <li>
                        <strong>Configure Connection:</strong>
                        <div className="mt-2 space-y-2">
                          <div>
                            <span className="text-xs font-medium">
                              Server URL:
                            </span>
                            <div className="flex items-center space-x-2 mt-1">
                              <code className="flex-1 px-2 py-1 bg-green-100 rounded text-xs font-mono">
                                {mcpServerUrl}
                              </code>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() =>
                                  copyToClipboard(mcpServerUrl, "openai-server")
                                }
                              >
                                <Copy className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                          <div>
                            <span className="text-xs font-medium">
                              Auth Discovery URL:
                            </span>
                            <div className="flex items-center space-x-2 mt-1">
                              <code className="flex-1 px-2 py-1 bg-green-100 rounded text-xs font-mono">
                                {metadataUrl}
                              </code>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() =>
                                  copyToClipboard(metadataUrl, "openai-auth")
                                }
                              >
                                <Copy className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                        </div>
                      </li>
                      <li>
                        <strong>Complete OAuth Flow:</strong> Authenticate when
                        prompted
                      </li>
                      <li>
                        <strong>Verify Connection:</strong> ragstar tools should
                        appear in available tools
                      </li>
                    </ol>
                  </div>

                  <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                    <h4 className="font-medium text-amber-900 mb-2">
                      📝 Important Notes:
                    </h4>
                    <ul className="text-sm text-amber-800 space-y-1 list-disc list-inside">
                      <li>
                        OpenAI's MCP integration is actively being developed
                      </li>
                      <li>
                        Check OpenAI's latest documentation for current
                        connection methods
                      </li>
                      <li>
                        Some OpenAI applications may require specific MCP client
                        libraries
                      </li>
                      <li>
                        Contact OpenAI support if you encounter integration
                        issues
                      </li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Security Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Security & Privacy
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="font-medium text-gray-900">
                  🔒 Security Features
                </h4>
                <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside">
                  <li>OAuth 2.1 with PKCE authentication</li>
                  <li>Encrypted token storage</li>
                  <li>Automatic token rotation</li>
                  <li>Scope-based access control</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-gray-900">🛡️ Your Control</h4>
                <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside">
                  <li>Revoke access anytime</li>
                  <li>Monitor active connections</li>
                  <li>Data stays in your instance</li>
                  <li>Audit logs for all access</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageLayout>
  );
}
