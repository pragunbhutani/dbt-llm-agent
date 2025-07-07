"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import {
  Copy,
  CheckCircle,
  AlertCircle,
  ExternalLink,
  Info,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";

import { toast } from "sonner";
import { Breadcrumb } from "@/components/ui/breadcrumb";
import PageLayout from "@/components/layout/page-layout";
import { fetcher } from "@/utils/fetcher";
import useSWR from "swr";

interface SlackIntegrationStatus {
  id: number | null;
  is_configured: boolean;
  connection_status: "connected" | "error" | "not_configured" | "unknown";
  last_tested_at?: string;
}

export default function SlackIntegrationPage() {
  const { data: session } = useSession();
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Form data
  const [customUrl, setCustomUrl] = useState("");
  const [botToken, setBotToken] = useState("");
  const [signingSecret, setSigningSecret] = useState("");
  const [appToken, setAppToken] = useState("");

  // Get current integration status
  const { data: integrationStatus, mutate: refreshStatus } = useSWR(
    session?.accessToken
      ? "/api/integrations/organisation-integrations/status/"
      : null,
    (url) => fetcher(url, session!.accessToken as string)
  );

  const slackIntegration = integrationStatus?.find(
    (i: any) => i.key === "slack"
  ) as SlackIntegrationStatus | undefined;

  // Generate API URLs
  const generateApiUrls = () => {
    const baseUrl =
      customUrl ||
      process.env.NEXT_PUBLIC_API_URL ||
      (typeof window !== "undefined"
        ? `${window.location.protocol}//${window.location.host}`
        : "");

    return {
      eventsUrl: `${baseUrl}/api/integrations/slack/events/`,
      shortcutsUrl: `${baseUrl}/api/integrations/slack/shortcuts/`,
    };
  };

  // Generate the Slack app manifest with dynamic URLs
  const generateSlackManifest = () => {
    const urls = generateApiUrls();

    return {
      display_information: {
        name: "Ragstar - AI Data Assistant",
        description:
          "Ragstar is an AI data assistant that allows you to ask questions about your data and get answers back.",
        background_color: "#2F6EAF",
      },
      features: {
        bot_user: {
          display_name: "ragstar",
          always_online: true,
        },
        shortcuts: [
          {
            name: "Run Query (csv)",
            type: "message",
            callback_id: "execute_query",
            description:
              "Executes the query and returns the result as CSV to the same thread",
          },
          {
            name: "Create Metabase Query",
            type: "message",
            callback_id: "create_metabase_query",
            description:
              "Saves this message as a SQL query in Metabase and posts a link to it.",
          },
        ],
      },
      oauth_config: {
        scopes: {
          bot: [
            "commands",
            "team:read",
            "im:history",
            "app_mentions:read",
            "chat:write",
            "chat:write.public",
            "groups:history",
            "channels:history",
            "users:read",
            "files:read",
            "files:write",
            "reactions:read",
          ],
        },
      },
      settings: {
        interactivity: {
          is_enabled: true,
          request_url: urls.shortcutsUrl,
        },
        org_deploy_enabled: false,
        socket_mode_enabled: false,
        token_rotation_enabled: false,
        event_subscriptions: {
          request_url: urls.eventsUrl,
          bot_events: [
            "app_mention",
            "message.im",
            "reaction_added",
            "reaction_removed",
          ],
        },
      },
    };
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success("Copied to clipboard!");
    } catch (err) {
      toast.error("Failed to copy to clipboard");
    }
  };

  const handleTokenSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!session?.accessToken || !botToken.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/integrations/slack/setup/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${session.accessToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            bot_token: botToken.trim(),
            signing_secret: signingSecret.trim(),
            app_token: appToken.trim(),
          }),
        }
      );

      if (!response.ok) {
        let errorMessage = "Configuration failed";
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorData.detail || errorMessage;
        } catch {}
        throw new Error(errorMessage);
      }

      await refreshStatus();
      toast.success("Slack integration configured successfully!");
      setCurrentStep(3); // Move to success step
    } catch (err: any) {
      setError(err.message);
      toast.error("Failed to configure Slack integration");
    } finally {
      setIsLoading(false);
    }
  };

  const testConnection = async () => {
    if (!session?.accessToken || !slackIntegration?.id) return;

    setIsLoading(true);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/integrations/organisation-integrations/${slackIntegration.id}/test_connection/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${session.accessToken}`,
            "Content-Type": "application/json",
          },
        }
      );

      if (response.ok) {
        await refreshStatus();
        toast.success("Connection test successful!");
      } else {
        const errorData = await response.json();
        toast.error(
          `Connection test failed: ${errorData.detail || "Unknown error"}`
        );
      }
    } catch (error) {
      toast.error("Connection test failed");
    } finally {
      setIsLoading(false);
    }
  };

  const urls = generateApiUrls();
  const manifest = generateSlackManifest();

  const breadcrumbItems = [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Integrations", href: "/dashboard/integrations" },
    { label: "Slack Setup" },
  ];

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
    <PageLayout
      title="Slack Integration"
      subtitle="Connect ragstar to Slack to interact with your data directly from your workspace."
      actions={
        slackIntegration ? (
          <Badge
            variant="outline"
            className={`text-xs ${getStatusColor(
              slackIntegration.connection_status
            )}`}
          >
            {getStatusText(slackIntegration.connection_status)}
          </Badge>
        ) : null
      }
    >
      <div className="space-y-6">
        <Breadcrumb
          items={[
            { label: "Integrations", href: "/dashboard/integrations" },
            { label: "Slack" },
          ]}
          className="mb-4"
        />

        {/* Integration Active Banner */}
        {slackIntegration?.is_configured && (
          <Card className="border-green-200 bg-green-50">
            <CardHeader>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-600" />
                <CardTitle className="text-green-800">
                  Integration Active
                </CardTitle>
              </div>
              <CardDescription className="text-green-700">
                Your Slack integration is configured and ready to use.
                {slackIntegration.last_tested_at && (
                  <span className="block mt-1">
                    Last tested:{" "}
                    {new Date(slackIntegration.last_tested_at).toLocaleString()}
                  </span>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Button onClick={testConnection} disabled={isLoading}>
                  Test Connection
                </Button>
                <Button variant="outline" onClick={() => setCurrentStep(2)}>
                  Reconfigure Tokens
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Main Content */}
        <Tabs value={currentStep.toString()} className="w-full">
          <TabsList className="grid grid-cols-3 w-full">
            <TabsTrigger value="1">1. Setup & Create App</TabsTrigger>
            <TabsTrigger value="2">2. Configure Tokens</TabsTrigger>
            <TabsTrigger value="3">3. Complete</TabsTrigger>
          </TabsList>

          <TabsContent value="1" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Configure URLs & Create Slack App</CardTitle>
                <CardDescription>
                  Set up the URLs that Slack will use to communicate with
                  Ragstar, then create your Slack app using the generated
                  manifest.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* URL Configuration */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="custom-url">
                      Custom Base URL (Optional)
                    </Label>
                    <Input
                      id="custom-url"
                      placeholder="https://your-domain.com or https://abc123.ngrok-free.app"
                      value={customUrl}
                      onChange={(e) => setCustomUrl(e.target.value)}
                    />
                    <p className="text-sm text-gray-600">
                      Leave empty to use the default URL. Use this for local
                      development with ngrok/localtunnel or if you&apos;re
                      hosting Ragstar on a custom domain.
                    </p>
                  </div>

                  <div className="space-y-3">
                    <h4 className="font-medium">Generated URLs:</h4>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 p-3 bg-gray-50 rounded-md">
                        <code className="flex-1 text-sm">{urls.eventsUrl}</code>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => copyToClipboard(urls.eventsUrl)}
                        >
                          <Copy className="h-4 w-4" />
                        </Button>
                      </div>
                      <div className="flex items-center gap-2 p-3 bg-gray-50 rounded-md">
                        <code className="flex-1 text-sm">
                          {urls.shortcutsUrl}
                        </code>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => copyToClipboard(urls.shortcutsUrl)}
                        >
                          <Copy className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Slack App Creation */}
                <div className="space-y-4 pt-6 border-t">
                  <h3 className="text-lg font-medium">Create Your Slack App</h3>

                  <div className="flex items-start gap-3 p-4 border border-blue-200 bg-blue-50 rounded-md">
                    <Info className="h-5 w-5 text-blue-600 mt-0.5" />
                    <div>
                      <h4 className="font-medium text-blue-800">
                        What you&apos;ll need to do:
                      </h4>
                      <ol className="list-decimal list-inside text-sm text-blue-700 mt-2 space-y-1">
                        <li>Copy the manifest below (preserves formatting)</li>
                        <li>
                          Click &quot;Create App on Slack&quot; to open Slack
                          API
                        </li>
                        <li>Choose &quot;From an app manifest&quot;</li>
                        <li>Paste the manifest and create the app</li>
                        <li>Install the app to your workspace</li>
                        <li>Copy the tokens back here in the next step</li>
                      </ol>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label>Slack App Manifest</Label>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() =>
                          copyToClipboard(JSON.stringify(manifest, null, 2))
                        }
                      >
                        <Copy className="h-4 w-4 mr-2" />
                        Copy Manifest
                      </Button>
                    </div>
                    <textarea
                      readOnly
                      value={JSON.stringify(manifest, null, 2)}
                      className="flex min-h-64 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 font-mono"
                    />
                  </div>

                  <div className="flex gap-2">
                    <Button asChild>
                      <a
                        href="https://api.slack.com/apps?new_app=1"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Create App on Slack
                        <ExternalLink className="h-4 w-4 ml-2" />
                      </a>
                    </Button>
                  </div>
                </div>

                <div className="flex justify-end pt-4">
                  <Button onClick={() => setCurrentStep(2)}>
                    Next: Configure Tokens
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="2" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Configure Integration Tokens</CardTitle>
                <CardDescription>
                  Enter the tokens from your Slack app to complete the
                  integration setup.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-md p-3 mb-4">
                    <div className="flex items-center gap-2">
                      <AlertCircle className="h-4 w-4 text-red-600" />
                      <p className="text-sm text-red-800">{error}</p>
                    </div>
                  </div>
                )}

                <form onSubmit={handleTokenSubmit} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="bot-token">Bot User OAuth Token *</Label>
                    <Input
                      id="bot-token"
                      type="password"
                      placeholder="xoxb-..."
                      value={botToken}
                      onChange={(e) => setBotToken(e.target.value)}
                      required
                    />
                    <p className="text-sm text-gray-600">
                      Found in OAuth &amp; Permissions → Bot User OAuth Token
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="signing-secret">Signing Secret *</Label>
                    <Input
                      id="signing-secret"
                      type="password"
                      placeholder="..."
                      value={signingSecret}
                      onChange={(e) => setSigningSecret(e.target.value)}
                      required
                    />
                    <p className="text-sm text-gray-600">
                      Found in Basic Information → Signing Secret
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="app-token">App Token (Optional)</Label>
                    <Input
                      id="app-token"
                      type="password"
                      placeholder="xapp-..."
                      value={appToken}
                      onChange={(e) => setAppToken(e.target.value)}
                    />
                    <p className="text-sm text-gray-600">
                      Only needed for Socket Mode. Found in Basic Information →
                      App-Level Tokens
                    </p>
                  </div>

                  <div className="flex gap-2 pt-4">
                    <Button type="submit" disabled={isLoading}>
                      {isLoading ? "Configuring..." : "Configure Integration"}
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setCurrentStep(1)}
                    >
                      Back
                    </Button>
                  </div>
                </form>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="3" className="space-y-6">
            <Card className="border-green-200">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-6 w-6 text-green-600" />
                  <CardTitle className="text-green-800">
                    Setup Complete!
                  </CardTitle>
                </div>
                <CardDescription>
                  Your Slack integration has been successfully configured.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <h4 className="font-medium">What&apos;s Next?</h4>
                  <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                    <li>Invite the Ragstar bot to your Slack channels</li>
                    <li>Start asking questions using @ragstar mentions</li>
                    <li>Use message shortcuts for quick actions</li>
                    <li>
                      Test the integration by sending a direct message to the
                      bot
                    </li>
                  </ul>
                </div>

                <div className="flex gap-2 pt-4">
                  <Button
                    onClick={() => router.push("/dashboard/integrations")}
                  >
                    Back to Integrations
                  </Button>
                  <Button
                    variant="outline"
                    onClick={testConnection}
                    disabled={isLoading}
                  >
                    Test Connection
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </PageLayout>
  );
}
