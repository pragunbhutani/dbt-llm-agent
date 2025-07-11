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
import useSWR from "swr";
import { fetcher } from "@/utils/fetcher";

export function SettingsContent() {
  const { data: session } = useSession();

  /**
   * Two pieces of state:
   * 1. `settings` – holds the *editable* form values.
   * 2. `existing` – booleans indicating whether a given secret is already set so we can
   *    inform the user without pre-filling inputs (which would re-save masked values).
   */
  const [settings, setSettings] = useState({
    llm_openai_api_key: "",
    llm_anthropic_api_key: "",
    llm_google_api_key: "",
    llm_chat_provider: "",
    llm_chat_model: "",
    llm_embeddings_provider: "",
    llm_embeddings_model: "",
  });

  const [existing, setExisting] = useState({
    openai: false,
    anthropic: false,
    google: false,
  });

  const [availableChatModels, setAvailableChatModels] = useState<
    { public_name: string; api_name: string }[]
  >([]);
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState<
    { public_name: string; api_name: string }[]
  >([]);

  // Tracks whether user has entered/edited a key (to switch input to revealed text)
  const [editedKeys, setEditedKeys] = useState<{
    [k in "openai" | "anthropic" | "google"]: boolean;
  }>({
    openai: false,
    anthropic: false,
    google: false,
  });

  // Fetch organisation settings (SWR with suspense)
  const { data: settingsData } = useSWR(
    session?.accessToken ? "/api/accounts/settings/" : null,
    (url: string) => fetcher(url, session?.accessToken),
    { suspense: true }
  );

  // --- Effect: populate form once data is fetched ---
  useEffect(() => {
    if (session?.error === "RefreshAccessTokenError") {
      toast.error("Your session has expired. Please sign in again.");
      signOut({ callbackUrl: "/signin" });
    }

    if (settingsData) {
      // Prefill masked keys *as returned by backend* so user can reveal them.
      setSettings((prev) => ({
        ...prev,
        llm_openai_api_key: settingsData.llm_openai_api_key || "",
        llm_anthropic_api_key: settingsData.llm_anthropic_api_key || "",
        llm_google_api_key: settingsData.llm_google_api_key || "",
        llm_chat_provider: settingsData.llm_chat_provider || "",
        llm_chat_model: settingsData.llm_chat_model || "",
        llm_embeddings_provider: settingsData.llm_embeddings_provider || "",
        llm_embeddings_model: settingsData.llm_embeddings_model || "",
      }));

      // Track whether secrets are already configured
      setExisting({
        openai: Boolean(settingsData.llm_openai_api_key),
        anthropic: Boolean(settingsData.llm_anthropic_api_key),
        google: Boolean(settingsData.llm_google_api_key),
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

  // --- Handlers ---
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

      // Mark field as edited so we switch to revealed text
      if (name === "llm_openai_api_key")
        setEditedKeys((e) => ({ ...e, openai: true }));
      if (name === "llm_anthropic_api_key")
        setEditedKeys((e) => ({ ...e, anthropic: true }));
      if (name === "llm_google_api_key")
        setEditedKeys((e) => ({ ...e, google: true }));
    }
  };

  // Helper to detect masked strings (contain * representing hidden chars)
  const isMasked = (val: string) => val.includes("*");

  /** Save only API keys that have been entered and are not masked. */
  const handleSaveKeys = async () => {
    const payload: Partial<typeof settings> = {};
    (
      [
        ["llm_openai_api_key", "openai"],
        ["llm_anthropic_api_key", "anthropic"],
        ["llm_google_api_key", "google"],
      ] as [keyof typeof settings, "openai" | "anthropic" | "google"][]
    ).forEach(([field]) => {
      const val = settings[field];
      if (val && !isMasked(val)) {
        payload[field] = val;
      }
    });

    if (Object.keys(payload).length === 0) {
      toast.message("No new keys to save.");
      return;
    }

    const promise = api.put("/api/accounts/settings/", payload);

    toast.promise(promise, {
      loading: "Saving keys...",
      success: () => {
        // After successful save clear inputs and refresh SWR cache
        setSettings((prev) => ({
          ...prev,
          llm_openai_api_key: "",
          llm_anthropic_api_key: "",
          llm_google_api_key: "",
        }));
        // Update existing flags (will also be updated on next revalidate)
        setExisting((prev) => ({
          ...prev,
          openai: Boolean(payload.llm_openai_api_key) || prev.openai,
          anthropic: Boolean(payload.llm_anthropic_api_key) || prev.anthropic,
          google: Boolean(payload.llm_google_api_key) || prev.google,
        }));
        return "API keys saved.";
      },
      error: "Failed to save keys.",
    });
  };

  /** Save provider / model selections */
  const handleSaveModels = async () => {
    const payload: Partial<typeof settings> = {};
    if (settings.llm_chat_provider)
      payload.llm_chat_provider = settings.llm_chat_provider;
    if (settings.llm_chat_model)
      payload.llm_chat_model = settings.llm_chat_model;
    if (settings.llm_embeddings_provider)
      payload.llm_embeddings_provider = settings.llm_embeddings_provider;
    if (settings.llm_embeddings_model)
      payload.llm_embeddings_model = settings.llm_embeddings_model;

    const promise = api.put("/api/accounts/settings/", payload);

    toast.promise(promise, {
      loading: "Saving model configuration...",
      success: "Model configuration saved.",
      error: "Failed to save model configuration.",
    });
  };

  /* ------------ RENDER ------------- */

  /* Reusable API-key input row (masked value always shown) */
  const KeyInput = (
    provider: "openai" | "anthropic" | "google",
    label: string,
    field: keyof typeof settings
  ) => (
    <div className="space-y-2">
      <Label htmlFor={`${provider}-api-key`}>{label}</Label>
      <Input
        id={`${provider}-api-key`}
        type={existing[provider] && !editedKeys[provider] ? "password" : "text"}
        placeholder="Enter new key to update..."
        value={settings[field] as string}
        onChange={(e) => handleChange(field as string, e.target.value)}
      />
    </div>
  );

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
            {/* ----------- API KEYS ----------- */}
            {/* Temporarily hidden to simplify onboarding – keep code for future use */}
            {/* <Card>
              <CardHeader>
                <CardTitle>API Keys</CardTitle>
                <CardDescription>
                  Provide API keys for the LLM providers you want to use. Existing keys
                  are not displayed for security.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                 {KeyInput("openai", "OpenAI API Key", "llm_openai_api_key")}
                 {KeyInput("anthropic", "Anthropic API Key", "llm_anthropic_api_key")}
                 {KeyInput("google", "Google API Key", "llm_google_api_key")}
                 <div className="flex justify-end pt-4">
                   <Button onClick={handleSaveKeys} className="px-6" size="sm">
                     Save API Keys
                   </Button>
                 </div>
              </CardContent>
            </Card> */}

            {/* ----------- MODEL SELECTION ----------- */}
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
                <div className="flex justify-end pt-4">
                  <Button onClick={handleSaveModels} className="px-6" size="sm">
                    Save Model Configuration
                  </Button>
                </div>
              </CardContent>
            </Card>
          </CardContent>
        </Card>
      </TabsContent>

      {/* ---------- Workspace Tab Placeholder ---------- */}
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
