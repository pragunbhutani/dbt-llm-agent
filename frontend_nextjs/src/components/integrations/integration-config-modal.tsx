import { useState } from "react";
import { useSession } from "next-auth/react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";
import { fetcher } from "@/utils/fetcher";

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

interface IntegrationConfigModalProps {
  integration: IntegrationStatus;
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export function IntegrationConfigModal({
  integration,
  isOpen,
  onClose,
  onSuccess,
}: IntegrationConfigModalProps) {
  const { data: session } = useSession();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const renderConfigurationForm = () => {
    switch (integration.key) {
      case "slack":
        return (
          <SlackConfigForm
            integration={integration}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
            setError={setError}
            onSuccess={onSuccess}
            onClose={onClose}
          />
        );
      case "snowflake":
        return (
          <SnowflakeConfigForm
            integration={integration}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
            setError={setError}
            onSuccess={onSuccess}
            onClose={onClose}
          />
        );
      case "metabase":
        return (
          <MetabaseConfigForm
            integration={integration}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
            setError={setError}
            onSuccess={onSuccess}
            onClose={onClose}
          />
        );
      case "mcp":
        return (
          <MCPConfigForm
            integration={integration}
            isLoading={isLoading}
            setIsLoading={setIsLoading}
            setError={setError}
            onSuccess={onSuccess}
            onClose={onClose}
          />
        );
      default:
        return (
          <div className="text-center py-8">
            <p className="text-gray-500">
              Configuration form for {integration.name} coming soon...
            </p>
          </div>
        );
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[800px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <DialogTitle className="text-xl">
              Configure {integration.name}
            </DialogTitle>
            <Badge
              variant="outline"
              className={`text-xs ${getStatusColor(
                integration.connection_status
              )}`}
            >
              {integration.connection_status.replace("_", " ").toUpperCase()}
            </Badge>
          </div>
          <DialogDescription>
            Set up your {integration.name} integration to{" "}
            {integration.integration_type === "inbound"
              ? "receive requests from"
              : "connect to"}{" "}
            this service.
          </DialogDescription>
        </DialogHeader>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-3">
            <p className="text-sm text-red-800">{error}</p>
          </div>
        )}

        {renderConfigurationForm()}
      </DialogContent>
    </Dialog>
  );
}

interface ConfigFormProps {
  integration: IntegrationStatus;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  onSuccess: () => void;
  onClose: () => void;
}

function SlackConfigForm({
  integration,
  isLoading,
  setIsLoading,
  setError,
  onSuccess,
  onClose,
}: ConfigFormProps) {
  const { data: session } = useSession();
  const [botToken, setBotToken] = useState("");
  const [signingSecret, setSigningSecret] = useState("");
  const [appToken, setAppToken] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
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
          // Try to get response text first
          const responseText = await response.text();

          // Check if it's HTML (likely a 404 page from Next.js)
          if (responseText.includes("<!DOCTYPE")) {
            errorMessage = `API endpoint not found (${response.status}). Check if the backend is running.`;
          } else {
            // Try to parse as JSON
            try {
              const errorData = JSON.parse(responseText);
              errorMessage =
                errorData.error ||
                `HTTP ${response.status}: ${response.statusText}`;
            } catch (jsonError) {
              // Not JSON, use the text as-is or default message
              errorMessage =
                responseText ||
                `HTTP ${response.status}: ${response.statusText}`;
            }
          }
        } catch (textError) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      console.log("Slack setup successful:", result);

      onSuccess();
      onClose();
    } catch (error) {
      console.error("Slack setup error:", error);
      setError(error instanceof Error ? error.message : "Configuration failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-4">
        <div>
          <label
            htmlFor="bot-token"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Bot User OAuth Token <span className="text-red-500">*</span>
          </label>
          <input
            id="bot-token"
            type="password"
            value={botToken}
            onChange={(e) => setBotToken(e.target.value)}
            placeholder="xoxb-..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            required
            disabled={isLoading}
          />
          <p className="text-xs text-gray-500 mt-1">
            The Bot User OAuth Token from your Slack app configuration
          </p>
        </div>

        <div>
          <label
            htmlFor="signing-secret"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Signing Secret
          </label>
          <input
            id="signing-secret"
            type="password"
            value={signingSecret}
            onChange={(e) => setSigningSecret(e.target.value)}
            placeholder="Enter signing secret..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
          />
          <p className="text-xs text-gray-500 mt-1">
            Used to verify requests from Slack (optional but recommended)
          </p>
        </div>

        <div>
          <label
            htmlFor="app-token"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            App-Level Token
          </label>
          <input
            id="app-token"
            type="password"
            value={appToken}
            onChange={(e) => setAppToken(e.target.value)}
            placeholder="xapp-..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
          />
          <p className="text-xs text-gray-500 mt-1">
            App-level token for Socket Mode (optional)
          </p>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
        <h4 className="text-sm font-medium text-blue-900 mb-1">
          Setup Instructions
        </h4>
        <ol className="text-xs text-blue-800 space-y-1 list-decimal list-inside">
          <li>Go to your Slack app configuration at api.slack.com</li>
          <li>Navigate to &quot;OAuth &amp; Permissions&quot; section</li>
          <li>Copy the &quot;Bot User OAuth Token&quot; (starts with xoxb-)</li>
          <li>
            Optionally, copy the &quot;Signing Secret&quot; from &quot;Basic
            Information&quot;
          </li>
        </ol>
      </div>

      <div className="flex justify-end gap-3 pt-4">
        <Button
          type="button"
          variant="outline"
          onClick={onClose}
          disabled={isLoading}
        >
          Cancel
        </Button>
        <Button type="submit" disabled={isLoading || !botToken.trim()}>
          {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Save
        </Button>
      </div>
    </form>
  );
}

// Snowflake configuration form
function SnowflakeConfigForm({
  integration,
  isLoading,
  setIsLoading,
  setError,
  onSuccess,
  onClose,
}: ConfigFormProps) {
  const { data: session } = useSession();
  const [account, setAccount] = useState("");
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [warehouse, setWarehouse] = useState("");
  const [database, setDatabase] = useState("");
  const [schema, setSchema] = useState("PUBLIC");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (
      !session?.accessToken ||
      !account.trim() ||
      !user.trim() ||
      !password.trim() ||
      !warehouse.trim()
    )
      return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/integrations/snowflake/setup/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${session.accessToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            account: account.trim(),
            user: user.trim(),
            password: password.trim(),
            warehouse: warehouse.trim(),
            database: database.trim() || null,
            schema: database.trim() ? schema.trim() || "PUBLIC" : null,
          }),
        }
      );

      if (!response.ok) {
        let errorMessage = "Configuration failed";
        try {
          const responseText = await response.text();
          if (responseText.includes("<!DOCTYPE")) {
            errorMessage = `API endpoint not found (${response.status}). Check if the backend is running.`;
          } else {
            try {
              const errorData = JSON.parse(responseText);
              errorMessage =
                errorData.error ||
                `HTTP ${response.status}: ${response.statusText}`;
            } catch (jsonError) {
              errorMessage =
                responseText ||
                `HTTP ${response.status}: ${response.statusText}`;
            }
          }
        } catch (textError) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      console.log("Snowflake setup successful:", result);

      onSuccess();
      onClose();
    } catch (error) {
      console.error("Snowflake setup error:", error);
      setError(error instanceof Error ? error.message : "Configuration failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex flex-col md:flex-row gap-6">
        <div className="space-y-4 flex-1">
          <div>
            <label
              htmlFor="account"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Account <span className="text-red-500">*</span>
            </label>
            <input
              id="account"
              type="text"
              value={account}
              onChange={(e) => setAccount(e.target.value)}
              placeholder="your-account-name"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
              disabled={isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">
              Your Snowflake account identifier (e.g.
              mycompany.snowflakecomputing.com)
            </p>
          </div>

          <div>
            <label
              htmlFor="user"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Username <span className="text-red-500">*</span>
            </label>
            <input
              id="user"
              type="text"
              value={user}
              onChange={(e) => setUser(e.target.value)}
              placeholder="your-username"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
              disabled={isLoading}
            />
          </div>

          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Password <span className="text-red-500">*</span>
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
              disabled={isLoading}
            />
          </div>

          <div>
            <label
              htmlFor="warehouse"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Warehouse <span className="text-red-500">*</span>
            </label>
            <input
              id="warehouse"
              type="text"
              value={warehouse}
              onChange={(e) => setWarehouse(e.target.value)}
              placeholder="COMPUTE_WH"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
              disabled={isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">
              The warehouse to use for queries
            </p>
          </div>

          <div>
            <label
              htmlFor="database"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Database (Optional)
            </label>
            <input
              id="database"
              type="text"
              value={database}
              onChange={(e) => setDatabase(e.target.value)}
              placeholder="YOUR_DATABASE"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">
              Optional default database to use (can be specified later in
              queries)
            </p>
          </div>

          <div>
            <label
              htmlFor="schema"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Schema
            </label>
            <input
              id="schema"
              type="text"
              value={schema}
              onChange={(e) => setSchema(e.target.value)}
              placeholder="PUBLIC"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">
              Optional default schema to use when database is specified
              (defaults to PUBLIC)
            </p>
          </div>
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-md p-3 md:w-1/3 overflow-y-auto max-h-[60vh]">
          <h4 className="text-sm font-medium text-blue-900 mb-1">
            Setup Instructions
          </h4>
          <ol className="text-xs text-blue-800 space-y-1 list-decimal list-inside">
            <li>Log into your Snowflake account</li>
            <li>Create a user with appropriate permissions for data access</li>
            <li>
              Note your account identifier (visible in your Snowflake URL)
            </li>
            <li>Enter the warehouse, database, and schema you want to use</li>
          </ol>
        </div>
      </div>

      <div className="flex justify-end gap-3 pt-4">
        <Button
          type="button"
          variant="outline"
          onClick={onClose}
          disabled={isLoading}
        >
          Cancel
        </Button>
        <Button
          type="submit"
          disabled={
            isLoading ||
            !account.trim() ||
            !user.trim() ||
            !password.trim() ||
            !warehouse.trim()
          }
        >
          {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Save
        </Button>
      </div>
    </form>
  );
}

function MetabaseConfigForm({
  integration,
  isLoading,
  setIsLoading,
  setError,
  onSuccess,
  onClose,
}: ConfigFormProps) {
  const { data: session } = useSession();
  const [url, setUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [databaseId, setDatabaseId] = useState("1");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (
      !session?.accessToken ||
      !url.trim() ||
      !apiKey.trim() ||
      !databaseId.trim()
    )
      return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/integrations/metabase/setup/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${session.accessToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            url: url.trim(),
            api_key: apiKey.trim(),
            database_id: parseInt(databaseId.trim()),
          }),
        }
      );

      if (!response.ok) {
        let errorMessage = "Configuration failed";
        try {
          const responseText = await response.text();
          if (responseText.includes("<!DOCTYPE")) {
            errorMessage = `API endpoint not found (${response.status}). Check if the backend is running.`;
          } else {
            try {
              const errorData = JSON.parse(responseText);
              errorMessage =
                errorData.error ||
                `HTTP ${response.status}: ${response.statusText}`;
            } catch (jsonError) {
              errorMessage =
                responseText ||
                `HTTP ${response.status}: ${response.statusText}`;
            }
          }
        } catch (textError) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      console.log("Metabase setup successful:", result);

      onSuccess();
      onClose();
    } catch (error) {
      console.error("Metabase setup error:", error);
      setError(error instanceof Error ? error.message : "Configuration failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-4">
        <div>
          <label
            htmlFor="url"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Metabase URL <span className="text-red-500">*</span>
          </label>
          <input
            id="url"
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://your-metabase.example.com"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            required
            disabled={isLoading}
          />
          <p className="text-xs text-gray-500 mt-1">
            The URL of your Metabase instance
          </p>
        </div>

        <div>
          <label
            htmlFor="api-key"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            API Key <span className="text-red-500">*</span>
          </label>
          <input
            id="api-key"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Enter your API key..."
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            required
            disabled={isLoading}
          />
          <p className="text-xs text-gray-500 mt-1">
            Your Metabase API key from Admin Settings &gt; API Keys
          </p>
        </div>

        <div>
          <label
            htmlFor="database-id"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Database ID <span className="text-red-500">*</span>
          </label>
          <input
            id="database-id"
            type="number"
            value={databaseId}
            onChange={(e) => setDatabaseId(e.target.value)}
            placeholder="1"
            min="1"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            required
            disabled={isLoading}
          />
          <p className="text-xs text-gray-500 mt-1">
            The ID of the database you want to use (check the URL when viewing a
            database in Metabase)
          </p>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
        <h4 className="text-sm font-medium text-blue-900 mb-1">
          Setup Instructions
        </h4>
        <ol className="text-xs text-blue-800 space-y-1 list-decimal list-inside">
          <li>Log into your Metabase instance as an admin</li>
          <li>Go to Admin Settings &gt; API Keys</li>
          <li>Create a new API key with appropriate permissions</li>
          <li>Note the database ID from the URL when viewing your database</li>
        </ol>
      </div>

      <div className="flex justify-end gap-3 pt-4">
        <Button
          type="button"
          variant="outline"
          onClick={onClose}
          disabled={isLoading}
        >
          Cancel
        </Button>
        <Button
          type="submit"
          disabled={
            isLoading || !url.trim() || !apiKey.trim() || !databaseId.trim()
          }
        >
          {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Save
        </Button>
      </div>
    </form>
  );
}

function MCPConfigForm({
  integration,
  isLoading,
  setIsLoading,
  setError,
  onSuccess,
  onClose,
}: ConfigFormProps) {
  const { data: session } = useSession();
  const [activeTab, setActiveTab] = useState<"claude" | "openai">("claude");
  const [serverUrl, setServerUrl] = useState("");
  const [copied, setCopied] = useState<string | null>(null);

  // Get the current domain for the MCP server URL
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

  const copyToClipboard = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
      setTimeout(() => setCopied(null), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  const handleEnableIntegration = async () => {
    if (!session?.accessToken) return;

    setIsLoading(true);
    setError(null);

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
              server_url: getMCPServerUrl(),
              auth_provider: activeTab,
            },
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to enable MCP server");
      }

      onSuccess();
      onClose();
    } catch (error) {
      console.error("MCP setup error:", error);
      setError(error instanceof Error ? error.message : "Setup failed");
    } finally {
      setIsLoading(false);
    }
  };

  const mcpServerUrl = getMCPServerUrl();
  const authServerUrl =
    typeof window !== "undefined"
      ? `${window.location.protocol}//${window.location.hostname}${
          window.location.port ? `:${window.location.port}` : ""
        }`
      : "https://your-ragstar-domain.com";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
        <h4 className="text-sm font-medium text-blue-900 mb-2">
          🚀 Enable Remote MCP Server
        </h4>
        <p className="text-sm text-blue-800">
          Make your ragstar instance available as a remote MCP server that AI
          applications like Claude.ai and OpenAI can connect to securely.
        </p>
      </div>

      {/* Tab Selection */}
      <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
        <button
          type="button"
          onClick={() => setActiveTab("claude")}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
            activeTab === "claude"
              ? "bg-white text-blue-700 shadow-sm"
              : "text-gray-600 hover:text-gray-900"
          }`}
        >
          Claude.ai
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("openai")}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
            activeTab === "openai"
              ? "bg-white text-blue-700 shadow-sm"
              : "text-gray-600 hover:text-gray-900"
          }`}
        >
          OpenAI
        </button>
      </div>

      {/* Server Information */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            MCP Server URL
          </label>
          <div className="flex items-center space-x-2">
            <code className="flex-1 px-3 py-2 bg-gray-50 border border-gray-200 rounded-md text-sm font-mono">
              {mcpServerUrl}
            </code>
            <button
              type="button"
              onClick={() => copyToClipboard(mcpServerUrl, "server-url")}
              className="px-3 py-2 text-sm text-gray-600 hover:text-gray-900 border border-gray-200 rounded-md hover:bg-gray-50"
            >
              {copied === "server-url" ? "Copied!" : "Copy"}
            </button>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Authorization Server URL
          </label>
          <div className="flex items-center space-x-2">
            <code className="flex-1 px-3 py-2 bg-gray-50 border border-gray-200 rounded-md text-sm font-mono">
              {authServerUrl}
            </code>
            <button
              type="button"
              onClick={() => copyToClipboard(authServerUrl, "auth-url")}
              className="px-3 py-2 text-sm text-gray-600 hover:text-gray-900 border border-gray-200 rounded-md hover:bg-gray-50"
            >
              {copied === "auth-url" ? "Copied!" : "Copy"}
            </button>
          </div>
        </div>
      </div>

      {/* Instructions */}
      {activeTab === "claude" && (
        <div className="bg-purple-50 border border-purple-200 rounded-md p-4">
          <h4 className="text-sm font-medium text-purple-900 mb-3">
            📋 How to connect Claude.ai to your ragstar MCP server
          </h4>
          <ol className="text-sm text-purple-800 space-y-2 list-decimal list-inside">
            <li>
              <strong>Open Claude.ai</strong> in your web browser and sign in
            </li>
            <li>
              <strong>Go to Settings</strong> → <strong>Integrations</strong> or{" "}
              <strong>Connected Apps</strong>
            </li>
            <li>
              <strong>Add MCP Server</strong> or{" "}
              <strong>Connect Remote Server</strong>
            </li>
            <li>
              <strong>Enter the MCP Server URL:</strong>
              <br />
              <code className="ml-4 text-xs bg-purple-100 px-2 py-1 rounded">
                {mcpServerUrl}
              </code>
            </li>
            <li>
              <strong>Claude will redirect you</strong> to this ragstar instance
              for authorization
            </li>
            <li>
              <strong>Sign in to ragstar</strong> and authorize Claude.ai to
              access your data
            </li>
            <li>
              <strong>Return to Claude.ai</strong> - you should now see ragstar
              tools available!
            </li>
          </ol>
          <div className="mt-3 p-3 bg-purple-100 rounded-md">
            <p className="text-xs text-purple-700">
              <strong>💡 Tip:</strong> Once connected, you can ask Claude
              questions like &quot;Show me my dbt models&quot; or &quot;What
              data sources do I have?&quot; and Claude will use ragstar&apos;s
              tools to answer.
            </p>
          </div>
        </div>
      )}

      {activeTab === "openai" && (
        <div className="bg-green-50 border border-green-200 rounded-md p-4">
          <h4 className="text-sm font-medium text-green-900 mb-3">
            📋 How to connect OpenAI to your ragstar MCP server
          </h4>
          <ol className="text-sm text-green-800 space-y-2 list-decimal list-inside">
            <li>
              <strong>Install MCP-compatible OpenAI client</strong> (like the
              OpenAI Desktop app with MCP support)
            </li>
            <li>
              <strong>Open the application</strong> and go to Settings →
              Integrations
            </li>
            <li>
              <strong>Add Remote MCP Server</strong>
            </li>
            <li>
              <strong>Configure the server connection:</strong>
              <br />
              <span className="ml-4 text-xs">
                Server URL:{" "}
                <code className="bg-green-100 px-2 py-1 rounded">
                  {mcpServerUrl}
                </code>
                <br />
                Auth URL:{" "}
                <code className="bg-green-100 px-2 py-1 rounded">
                  {authServerUrl}
                </code>
              </span>
            </li>
            <li>
              <strong>Complete OAuth authorization</strong> when prompted
            </li>
            <li>
              <strong>Verify connection</strong> - ragstar tools should appear
              in your available tools
            </li>
          </ol>
          <div className="mt-3 p-3 bg-green-100 rounded-md">
            <p className="text-xs text-green-700">
              <strong>📝 Note:</strong> OpenAI&apos;s MCP integration is still
              evolving. Check OpenAI&apos;s documentation for the latest
              connection instructions.
            </p>
          </div>
        </div>
      )}

      {/* Security Notice */}
      <div className="bg-amber-50 border border-amber-200 rounded-md p-4">
        <h4 className="text-sm font-medium text-amber-900 mb-2">
          🔒 Security & Permissions
        </h4>
        <ul className="text-sm text-amber-800 space-y-1 list-disc list-inside">
          <li>
            Your ragstar data remains secure - AI applications only get access
            after you authorize them
          </li>
          <li>You can revoke access at any time from the integrations page</li>
          <li>All connections use OAuth 2.1 with PKCE for maximum security</li>
          <li>
            AI applications can only access tools and data you explicitly grant
            permission for
          </li>
        </ul>
      </div>

      {/* Enable Button */}
      <div className="flex justify-end gap-3 pt-4">
        <Button
          type="button"
          variant="outline"
          onClick={onClose}
          disabled={isLoading}
        >
          Cancel
        </Button>
        <Button
          type="button"
          onClick={handleEnableIntegration}
          disabled={isLoading}
        >
          {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Enable MCP Server
        </Button>
      </div>
    </div>
  );
}
