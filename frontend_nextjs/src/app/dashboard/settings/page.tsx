"use client";

import { useEffect, useState } from "react";
import { useSession, signOut } from "next-auth/react";
import Heading from "@/components/heading";
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

export default function SettingsPage() {
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
  const [isLoading, setIsLoading] = useState(true);
  const [availableChatModels, setAvailableChatModels] = useState<
    { public_name: string; api_name: string }[]
  >([]);
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState<
    { public_name: string; api_name: string }[]
  >([]);

  useEffect(() => {
    if (session?.error === "RefreshAccessTokenError") {
      toast.error("Your session has expired. Please sign in again.");
      signOut({ callbackUrl: "/signin" });
    }

    const fetchSettings = async () => {
      try {
        const data = await api.get("/api/accounts/settings/");
        setSettings({
          ...data,
          llm_chat_provider: data.llm_chat_provider || "",
          llm_chat_model: data.llm_chat_model || "",
          llm_embeddings_provider: data.llm_embeddings_provider || "",
          llm_embeddings_model: data.llm_embeddings_model || "",
        });

        if (data.llm_chat_provider) {
          setAvailableChatModels(CHAT_MODELS[data.llm_chat_provider] || []);
        }
        if (data.llm_embeddings_provider) {
          setAvailableEmbeddingModels(
            EMBEDDING_MODELS[data.llm_embeddings_provider] || []
          );
        }
      } catch (error) {
        toast.error("Failed to load settings.");
        console.error(error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchSettings();
  }, [session]);

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

  if (isLoading) {
    return <div>Loading...</div>;
  }

  return (
    <>
      <div className="flex h-16 items-center border-b border-gray-200 px-4">
        <Heading
          title="Settings"
          subtitle="Manage your workspace and LLM configurations."
        />
      </div>
      <div className="p-4">
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
                      <Label htmlFor="anthropic-api-key">
                        Anthropic API Key
                      </Label>
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
                    <CardTitle>Chat Model</CardTitle>
                    <CardDescription>
                      Select the primary model for chat-based interactions.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="grid gap-6 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="chat-provider">Provider</Label>
                      <Select
                        value={settings.llm_chat_provider}
                        onValueChange={(value) =>
                          handleChange("llm_chat_provider", value)
                        }
                      >
                        <SelectTrigger id="chat-provider">
                          <SelectValue placeholder="Select a provider" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="openai">OpenAI</SelectItem>
                          <SelectItem value="anthropic">Anthropic</SelectItem>
                          <SelectItem value="google">Google</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="chat-model">Model</Label>
                      <Select
                        value={settings.llm_chat_model || undefined}
                        onValueChange={(value) =>
                          handleChange("llm_chat_model", value)
                        }
                        disabled={!settings.llm_chat_provider}
                      >
                        <SelectTrigger
                          id="chat-model"
                          className={cn({
                            "text-muted-foreground": !settings.llm_chat_model,
                          })}
                        >
                          <SelectValue placeholder="Select a model" />
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
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Embedding Model</CardTitle>
                    <CardDescription>
                      Select the model for creating vector embeddings for your
                      knowledge base.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="grid gap-6 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="embedding-provider">Provider</Label>
                      <Select
                        value={settings.llm_embeddings_provider}
                        onValueChange={(value) =>
                          handleChange("llm_embeddings_provider", value)
                        }
                      >
                        <SelectTrigger id="embedding-provider">
                          <SelectValue placeholder="Select a provider" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="openai">OpenAI</SelectItem>
                          <SelectItem value="google">Google</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="embedding-model">Model</Label>
                      <Select
                        value={settings.llm_embeddings_model || undefined}
                        onValueChange={(value) =>
                          handleChange("llm_embeddings_model", value)
                        }
                        disabled={!settings.llm_embeddings_provider}
                      >
                        <SelectTrigger
                          id="embedding-model"
                          className={cn({
                            "text-muted-foreground":
                              !settings.llm_embeddings_model,
                          })}
                        >
                          <SelectValue placeholder="Select a model" />
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
                  </CardContent>
                </Card>
                <div className="flex justify-end">
                  <Button onClick={handleSave}>Save Settings</Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="workspace">
            <Card>
              <CardHeader>
                <CardTitle>Workspace Settings</CardTitle>
                <CardDescription>
                  Manage your workspace settings here.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p>Workspace settings content will go here.</p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </>
  );
}
