"use client";

import React from "react";
import { useParams, useRouter } from "next/navigation";
import useSWR from "swr";
import { useSession } from "next-auth/react";
import { fetcher } from "@/utils/fetcher";
import { toast } from "sonner";
import Heading from "@/components/heading";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ArrowLeft,
  Calendar,
  User,
  MessageSquare,
  Trash2,
  Hash,
  Activity,
} from "lucide-react";
import { Conversation, ConversationPart } from "@/types/conversations";
import { CHAT_MODELS } from "@/lib/llm_models";

export default function ConversationDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { data: session } = useSession();
  const conversationId = params.id as string;

  const apiURL = session?.accessToken
    ? `/api/workflows/conversations/${conversationId}/`
    : null;

  const {
    data: conversation,
    error,
    mutate,
  } = useSWR<Conversation>(apiURL, (url: string) =>
    fetcher(url, session?.accessToken)
  );

  const handleDeleteConversation = async () => {
    if (!session?.accessToken || !conversation) return;

    const confirmed = window.confirm(
      "Are you sure you want to delete this conversation? This action cannot be undone."
    );

    if (!confirmed) return;

    try {
      await fetcher(
        `/api/workflows/conversations/${conversationId}/`,
        session.accessToken,
        { method: "DELETE" }
      );

      toast.success("Conversation deleted successfully");
      router.push("/dashboard/chats");
    } catch (error) {
      console.error("Failed to delete conversation", error);
      toast.error("Failed to delete conversation. Please try again.");
    }
  };

  const formatDateTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getModelDisplayName = (provider: string, model?: string) => {
    if (!model) return provider;

    const providerModels = CHAT_MODELS[provider];
    if (!providerModels) return model;

    const modelInfo = providerModels.find((m) => m.api_name === model);
    return modelInfo?.public_name || model;
  };

  const getMessageAlignment = (actor: ConversationPart["actor"]) => {
    switch (actor) {
      case "user":
        return "flex-row-reverse"; // Right aligned
      case "system":
        return "justify-center"; // Center aligned
      default:
        return "flex-row"; // Left aligned (agent, llm, tool)
    }
  };

  const getMessageBackground = (actor: ConversationPart["actor"]) => {
    switch (actor) {
      case "user":
        return "bg-blue-500 text-white"; // User messages - blue
      case "agent":
      case "llm": // Treat LLM same as agent
        return "bg-white border border-gray-200"; // Assistant messages - white
      case "system":
        return "bg-gray-100 text-gray-700"; // System messages - gray
      case "tool":
        return "bg-orange-50 border border-orange-200 text-orange-800"; // Tool - orange
      default:
        return "bg-gray-50 border border-gray-200";
    }
  };

  const getActorLabel = (actor: ConversationPart["actor"]) => {
    switch (actor) {
      case "user":
        return "You";
      case "agent":
      case "llm":
        return "Assistant";
      case "system":
        return "System";
      case "tool":
        return "Tool";
      default:
        return String(actor).charAt(0).toUpperCase() + String(actor).slice(1);
    }
  };

  const shouldShowInTimeline = (part: ConversationPart) => {
    // Filter out internal system messages that aren't user-facing
    const hiddenTypes = ["thinking", "llm_input", "tool_execution"];
    const isEmptyLlm =
      part.actor === "llm" && (!part.content || part.content.trim() === "");
    return !hiddenTypes.includes(part.message_type) && !isEmptyLlm;
  };

  // Function to format message content with code blocks
  const formatMessageContent = (content: string) => {
    // Split content by code blocks (```...```)
    const parts = content.split(/(```[\s\S]*?```)/g);

    return parts.map((part, index) => {
      if (part.startsWith("```") && part.endsWith("```")) {
        // This is a code block
        const codeContent = part.slice(3, -3); // Remove ``` from start and end
        const lines = codeContent.split("\n");
        const language = lines[0].trim(); // First line might be language
        const code =
          language &&
          ["sql", "python", "javascript", "json", "yaml", "bash"].includes(
            language.toLowerCase()
          )
            ? lines.slice(1).join("\n") // Remove language line
            : codeContent; // Keep all content if no valid language detected

        return (
          <pre
            key={index}
            className="mt-3 mb-3 p-4 bg-gray-900 text-gray-100 rounded-lg overflow-x-auto text-sm font-mono whitespace-pre w-full"
          >
            <code className="block min-w-0">{code}</code>
          </pre>
        );
      } else {
        // Regular text content
        return part ? (
          <span key={index} className="whitespace-pre-wrap">
            {part}
          </span>
        ) : null;
      }
    });
  };

  if (error) {
    return (
      <div className="flex h-16 items-center border-b border-gray-200 px-4">
        <Heading
          title="Conversation Details"
          subtitle="Error loading conversation"
        />
        <div className="p-4">
          <div className="text-center py-12">
            <p className="text-red-600">Failed to load conversation</p>
          </div>
        </div>
      </div>
    );
  }

  if (!conversation) {
    return (
      <div className="flex h-16 items-center border-b border-gray-200 px-4">
        <Heading title="Conversation Details" subtitle="Loading..." />
        <div className="p-4">
          <div className="text-center py-12">
            <p>Loading conversation...</p>
          </div>
        </div>
      </div>
    );
  }

  const channelVariant =
    conversation.channel === "slack"
      ? "default"
      : conversation.channel === "web"
      ? "secondary"
      : conversation.channel === "mcp"
      ? "outline"
      : "destructive";

  const statusVariant =
    conversation.status === "completed"
      ? "default"
      : conversation.status === "active"
      ? "secondary"
      : conversation.status === "error"
      ? "destructive"
      : "outline";

  const visibleParts = conversation.parts?.filter(shouldShowInTimeline) || [];

  return (
    <>
      {/* Header */}
      <div className="flex h-16 items-center justify-between border-b border-gray-200 px-4">
        <Heading
          title="Conversation Details"
          subtitle={`Started ${formatDateTime(conversation.started_at)}`}
        />
        <Button variant="destructive" onClick={handleDeleteConversation}>
          <Trash2 className="mr-2 h-4 w-4" />
          Delete
        </Button>
      </div>

      {/* Main Content - Two Column Layout */}
      <div className="flex h-[calc(100vh-4rem)]">
        {/* Left Sidebar - Conversation Details */}
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
          {/* Back Button */}
          <div className="p-4">
            <Button
              variant="ghost"
              onClick={() => router.push("/dashboard/chats")}
              className="w-full justify-start"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Conversations
            </Button>
          </div>

          {/* Conversation Overview */}
          <div className="px-4 space-y-4 flex-1 overflow-y-auto">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Overview</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4 text-gray-500" />
                  <span className="text-sm font-medium">Channel</span>
                  <Badge variant={channelVariant} className="ml-auto">
                    {conversation.channel.charAt(0).toUpperCase() +
                      conversation.channel.slice(1)}
                  </Badge>
                </div>

                <div className="flex items-center gap-2">
                  <User className="h-4 w-4 text-gray-500" />
                  <span className="text-sm font-medium">Started by</span>
                  <span className="text-sm text-gray-600 ml-auto">
                    {conversation.user_external_id ||
                      conversation.user_id ||
                      "Unknown"}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <Calendar className="h-4 w-4 text-gray-500" />
                  <span className="text-sm font-medium">Started</span>
                  <span className="text-sm text-gray-600 ml-auto">
                    {formatDateTime(conversation.started_at)}
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4 text-gray-500" />
                  <span className="text-sm font-medium">Messages</span>
                  <span className="text-sm text-gray-600 ml-auto">
                    {visibleParts.length}
                  </span>
                </div>

                {/* Chat model used */}
                <div className="flex items-center gap-2">
                  <Hash className="h-4 w-4 text-gray-500" />
                  <span className="text-sm font-medium">Model</span>
                  {/* Value */}
                  <span
                    className="text-sm text-gray-600 ml-auto truncate max-w-[140px]"
                    title={getModelDisplayName(
                      conversation.llm_provider,
                      conversation.llm_chat_model
                    )}
                  >
                    {getModelDisplayName(
                      conversation.llm_provider,
                      conversation.llm_chat_model
                    )}
                  </span>
                </div>

                {/* Token breakdown */}
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-gray-500" />
                  <span className="text-sm font-medium">Input Tokens</span>
                  <span className="text-sm text-gray-600 ml-auto">
                    {conversation.input_tokens?.toLocaleString() || 0}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-gray-500" />
                  <span className="text-sm font-medium">Output Tokens</span>
                  <span className="text-sm text-gray-600 ml-auto">
                    {conversation.output_tokens?.toLocaleString() || 0}
                  </span>
                </div>
              </CardContent>
            </Card>

            {/* Removed Initial Question card as redundant */}
          </div>
        </div>

        {/* Right Main Area - Chat Messages */}
        <div className="flex-1 flex flex-col bg-gray-50">
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-none mx-auto p-6 space-y-4 pr-8">
              {visibleParts.length > 0 ? (
                visibleParts
                  .sort((a, b) => a.sequence_number - b.sequence_number)
                  .map((part) => (
                    <div
                      key={part.id}
                      className={`flex ${getMessageAlignment(part.actor)} ${
                        part.actor === "system" ? "justify-center" : ""
                      }`}
                    >
                      <div
                        className={`max-w-4xl ${
                          part.actor === "system" ? "max-w-2xl" : ""
                        }`}
                      >
                        {/* Message Bubble */}
                        <div
                          className={`rounded-lg p-4 ${getMessageBackground(
                            part.actor
                          )} ${
                            part.actor === "user"
                              ? "ml-16"
                              : part.actor === "system"
                              ? "mx-16"
                              : "mr-16"
                          }`}
                        >
                          {/* Actor Label */}
                          <div className="flex items-center gap-2 mb-2">
                            <span
                              className={`text-xs font-medium ${
                                part.actor === "user"
                                  ? "text-blue-100"
                                  : "text-gray-600"
                              }`}
                            >
                              {getActorLabel(part.actor)}
                            </span>
                            <span
                              className={`text-xs ${
                                part.actor === "user"
                                  ? "text-blue-200"
                                  : "text-gray-500"
                              }`}
                            >
                              {formatTime(part.created_at)}
                            </span>
                          </div>

                          {/* Message Content with Code Block Support */}
                          <div
                            className={`text-sm ${
                              part.actor === "user" ? "text-white" : ""
                            }`}
                          >
                            {formatMessageContent(part.content)}
                          </div>

                          {/* Tool Information */}
                          {part.tool_name && (
                            <div className="mt-3 pt-2 border-t border-gray-300">
                              <p className="text-xs font-medium mb-1">
                                Tool: {part.tool_name}
                              </p>
                              {part.tool_input &&
                                Object.keys(part.tool_input).length > 0 && (
                                  <details className="mt-1">
                                    <summary className="text-xs cursor-pointer text-gray-600 hover:text-gray-800">
                                      Show Input
                                    </summary>
                                    <pre className="text-xs mt-1 p-2 bg-gray-100 rounded overflow-x-auto">
                                      {JSON.stringify(part.tool_input, null, 2)}
                                    </pre>
                                  </details>
                                )}
                              {part.tool_output &&
                                Object.keys(part.tool_output).length > 0 && (
                                  <details className="mt-1">
                                    <summary className="text-xs cursor-pointer text-gray-600 hover:text-gray-800">
                                      Show Output
                                    </summary>
                                    <pre className="text-xs mt-1 p-2 bg-gray-100 rounded overflow-x-auto">
                                      {JSON.stringify(
                                        part.tool_output,
                                        null,
                                        2
                                      )}
                                    </pre>
                                  </details>
                                )}
                            </div>
                          )}

                          {/* Cost and Performance */}
                          {(parseFloat(String(part.cost)) || 0) > 0 && (
                            <div className="mt-3 pt-2 border-t border-gray-300">
                              <div className="flex justify-between text-xs text-gray-600">
                                <span>
                                  Cost: $
                                  {(parseFloat(String(part.cost)) || 0).toFixed(
                                    4
                                  )}
                                </span>
                                <span>Tokens: {part.tokens_used}</span>
                                {part.duration_ms > 0 && (
                                  <span>{part.duration_ms}ms</span>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
              ) : (
                <div className="text-center py-12 text-gray-500">
                  No messages found in this conversation
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
