export interface ConversationPart {
  id: number;
  sequence_number: number;
  actor: "user" | "agent" | "system" | "llm" | "tool";
  message_type:
    | "message"
    | "intent_classification"
    | "llm_input"
    | "llm_output"
    | "tool_execution"
    | "tool_error"
    | "slack_output"
    | "slack_file_output"
    | "workflow_completion"
    | "error"
    | "thinking";
  content: string;
  tool_name?: string;
  tool_input?: any;
  tool_output?: any;
  tokens_used: number;
  cost: number | string;
  duration_ms: number;
  metadata: Record<string, any>;
  created_at: string;
}

export interface Organisation {
  id: number;
  name: string;
  // Add other organisation fields as needed
}

export interface Conversation {
  id: number;
  external_id?: string;
  channel: "slack" | "web" | "mcp" | "api";
  user_id?: string;
  user_external_id?: string;
  status: "active" | "completed" | "error" | "timeout";
  trigger: "slack_mention" | "web_interface" | "mcp_server" | "api_call";
  title?: string;
  summary?: string;
  initial_question: string;
  channel_type: string;
  channel_id?: string;
  llm_provider: string;
  llm_chat_model?: string;
  enabled_integrations: string[];
  total_parts: number;
  total_tokens_used: number;
  total_cost: number | string;
  input_tokens?: number;
  output_tokens?: number;
  thinking_tokens?: number;
  started_at: string;
  completed_at?: string;
  user_rating?: number;
  user_feedback?: string;
  conversation_context: Record<string, any>;
  organisation: Organisation;
  parts?: ConversationPart[];
  // Calculated properties from backend
  calculated_total_parts?: number;
  calculated_total_tokens?: number;
  calculated_total_cost?: number | string;
}

export interface ConversationListItem {
  id: number;
  channel: "slack" | "web" | "mcp" | "api";
  user_id?: string;
  user_external_id?: string;
  status: "active" | "completed" | "error" | "timeout";
  trigger: "slack_mention" | "web_interface" | "mcp_server" | "api_call";
  title?: string;
  initial_question: string;
  total_parts: number;
  total_tokens_used: number;
  total_cost: number | string;
  input_tokens?: number;
  output_tokens?: number;
  thinking_tokens?: number;
  started_at: string;
  completed_at?: string;
  user_rating?: number;
  organisation: Organisation;
}
