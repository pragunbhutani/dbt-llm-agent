"use client";

import { useEffect, useState } from "react";
import { useSession, signOut } from "next-auth/react";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import api from "@/lib/api";
import { toast } from "sonner";
import { CHAT_MODELS, EMBEDDING_MODELS } from "@/lib/llm_models";
import { cn } from "@/lib/utils";
import useSWR from "swr";
import { fetcher } from "@/utils/fetcher";

export function SettingsContent() {
  const { data: session } = useSession();
  const [settings, setSettings] = useState({
    llm_openai_api_key: "",
    llm_anthropic_api_key: "",
    llm_google_api_key: "",
    llm_chat_provider: "",
    llm_chat_model: "",
    llm_embeddings_provider: "",
    llm_embeddings_model: "",
  });
  const [availableChatModels, setAvailableChatModels] = useState<
    { public_name: string; api_name: string }[]
  >([]);
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState<
    { public_name: string; api_name: string }[]
  >([]);

  // Use SWR with suspense for settings data
  const { data: settingsData } = useSWR(
    session?.accessToken ? "/api/accounts/settings/" : null,
    (url: string) => fetcher(url, session?.accessToken),
    { suspense: true }
  );

  useEffect(() => {
    if (session?.error === "RefreshAccessTokenError") {
      toast.error("Your session has expired. Please sign in again.");
      signOut({ callbackUrl: "/signin" });
    }

    if (settingsData) {
      setSettings({
        ...settingsData,
        llm_chat_provider: settingsData.llm_chat_provider || "",
        llm_chat_model: settingsData.llm_chat_model || "",
        llm_embeddings_provider: settingsData.llm_embeddings_provider || "",
        llm_embeddings_model: settingsData.llm_embeddings_model || "",
      });

      if (settingsData.llm_chat_provider) {
        setAvailableChatModels(
          CHAT_MODELS[settingsData.llm_chat_provider] || []
        );
      }
      if (settingsData.llm_embeddings_provider) {
        setAvailableEmbeddingModels(
          EMBEDDING_MODELS[settingsData.llm_embeddings_provider] || []
        );
      }
    }
  }, [session, settingsData]);

  const handleChange = (name: string, value: string) => {
    if (name === "llm_chat_provider") {
      setSettings((prev) => ({
        ...prev,
        llm_chat_provider: value,
        llm_chat_model: "",
      }));
      setAvailableChatModels(CHAT_MODELS[value] || []);
    } else if (name === "llm_embeddings_provider") {
      setSettings((prev) => ({
        ...prev,
        llm_embeddings_provider: value,
        llm_embeddings_model: "",
      }));
      setAvailableEmbeddingModels(EMBEDDING_MODELS[value] || []);
    } else {
      setSettings((prev) => ({ ...prev, [name]: value }));
    }
  };

  const handleSave = async () => {
    const promise = async () => {
      const payload: Partial<typeof settings> = { ...settings };
      if (!payload.llm_openai_api_key) {
        delete payload.llm_openai_api_key;
      }
      if (!payload.llm_anthropic_api_key) {
        delete payload.llm_anthropic_api_key;
      }
      if (!payload.llm_google_api_key) {
        delete payload.llm_google_api_key;
      }
      await api.put("/api/accounts/settings/", payload);
    };

    toast.promise(promise(), {
      loading: "Saving settings...",
      success: "Settings saved successfully!",
      error: (error: any) => {
        console.error(error);
        if (error.response && error.response.status === 400) {
          const errorMessages = Object.entries(error.response.data)
            .map(([key, value]: [string, any]) => `${key}: ${value.join(", ")}`)
            .join("\n");
          return `Please correct the following errors:\n${errorMessages}`;
        }
        return "Failed to save settings.";
      },
    });
  };

  return (
    <Tabs defaultValue="llm">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="llm">LLM Configuration</TabsTrigger>
        <TabsTrigger value="workspace">Workspace</TabsTrigger>
      </TabsList>
      <TabsContent value="llm">
        <Card>
          <CardHeader>
            <CardTitle>LLM Configuration</CardTitle>
            <CardDescription>
              Manage API keys and select models for chat and embeddings.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>API Keys</CardTitle>
                <CardDescription>
                  Provide API keys for the LLM providers you want to use.
                  Existing keys are not displayed for security.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="openai-api-key">OpenAI API Key</Label>
                  <Input
                    id="openai-api-key"
                    type="password"
                    placeholder="Enter new key to update..."
                    value={settings.llm_openai_api_key ?? ""}
                    onChange={(e) =>
                      handleChange("llm_openai_api_key", e.target.value)
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="anthropic-api-key">Anthropic API Key</Label>
                  <Input
                    id="anthropic-api-key"
                    type="password"
                    placeholder="Enter new key to update..."
                    value={settings.llm_anthropic_api_key ?? ""}
                    onChange={(e) =>
                      handleChange("llm_anthropic_api_key", e.target.value)
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="google-api-key">Google API Key</Label>
                  <Input
                    id="google-api-key"
                    type="password"
                    placeholder="Enter new key to update..."
                    value={settings.llm_google_api_key ?? ""}
                    onChange={(e) =>
                      handleChange("llm_google_api_key", e.target.value)
                    }
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Model Selection</CardTitle>
                <CardDescription>
                  Choose your preferred models for chat and embeddings.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="chat-provider">Chat Provider</Label>
                    <Select
                      value={settings.llm_chat_provider}
                      onValueChange={(value) =>
                        handleChange("llm_chat_provider", value)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select provider" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="openai">OpenAI</SelectItem>
                        <SelectItem value="anthropic">Anthropic</SelectItem>
                        <SelectItem value="google">Google</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="chat-model">Chat Model</Label>
                    <Select
                      value={settings.llm_chat_model}
                      onValueChange={(value) =>
                        handleChange("llm_chat_model", value)
                      }
                      disabled={!settings.llm_chat_provider}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        {availableChatModels.map((model) => (
                          <SelectItem
                            key={model.api_name}
                            value={model.api_name}
                          >
                            {model.public_name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="embeddings-provider">
                      Embeddings Provider
                    </Label>
                    <Select
                      value={settings.llm_embeddings_provider}
                      onValueChange={(value) =>
                        handleChange("llm_embeddings_provider", value)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select provider" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="openai">OpenAI</SelectItem>
                        <SelectItem value="google">Google</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="embeddings-model">Embeddings Model</Label>
                    <Select
                      value={settings.llm_embeddings_model}
                      onValueChange={(value) =>
                        handleChange("llm_embeddings_model", value)
                      }
                      disabled={!settings.llm_embeddings_provider}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        {availableEmbeddingModels.map((model) => (
                          <SelectItem
                            key={model.api_name}
                            value={model.api_name}
                          >
                            {model.public_name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="flex justify-end">
              <Button onClick={handleSave} className="px-8">
                Save Settings
              </Button>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
      <TabsContent value="workspace">
        <Card>
          <CardHeader>
            <CardTitle>Workspace Settings</CardTitle>
            <CardDescription>
              Manage your workspace configuration and preferences.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              Workspace settings will be available in a future update.
            </p>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  );
}
