export const CHAT_MODELS: Record<
  string,
  { public_name: string; api_name: string }[]
> = {
  openai: [
    // { public_name: "GPT-3.5 Turbo", api_name: "gpt-3.5-turbo" },
    // { public_name: "GPT-3.5 Turbo (Jan 2024)", api_name: "gpt-3.5-turbo-0125" },
    // { public_name: "GPT-3.5 Turbo (Nov 2023)", api_name: "gpt-3.5-turbo-1106" },
    // { public_name: "GPT-3.5 Turbo 16K", api_name: "gpt-3.5-turbo-16k" },

    // { public_name: "GPT-4", api_name: "gpt-4" },

    // { public_name: "GPT-4 Turbo", api_name: "gpt-4-turbo" },
    // { public_name: "GPT-4 Turbo (Nov 2023)", api_name: "gpt-4-1106-preview" },

    // { public_name: "GPT-4o", api_name: "gpt-4o" },
    // { public_name: "GPT-4o (May 2024)", api_name: "gpt-4o-2024-05-13" },
    // { public_name: "GPT-4o (Aug 2024)", api_name: "gpt-4o-2024-08-06" },
    { public_name: "GPT-4o (Nov 2024)", api_name: "gpt-4o-2024-11-20" },

    // { public_name: "GPT-4o Mini", api_name: "gpt-4o-mini" },
    {
      public_name: "GPT-4o Mini (July 2024)",
      api_name: "gpt-4o-mini-2024-07-18",
    },

    // { public_name: "GPT-4o Audio Preview", api_name: "gpt-4o-audio-preview" },
    // {
    //   public_name: "GPT-4o Realtime Audio Preview",
    //   api_name: "gpt-4o-realtime-preview",
    // },

    // { public_name: "o1 Preview", api_name: "o1-preview" },
    { public_name: "o1", api_name: "o1" },
    { public_name: "o1 Mini", api_name: "o1-mini" },

    { public_name: "o3 Mini (Jan 2025)", api_name: "o3-mini-2025-01-31" },
    { public_name: "o3", api_name: "o3" },
    // { public_name: "o3 Pro", api_name: "o3-pro" },

    { public_name: "o4 Mini", api_name: "o4-mini" },
    { public_name: "o4 Mini High", api_name: "o4-mini-high" },

    { public_name: "GPT-4.1", api_name: "gpt-4.1" },
    { public_name: "GPT-4.1 Mini", api_name: "gpt-4.1-mini" },
    { public_name: "GPT-4.1 Nano", api_name: "gpt-4.1-nano" },

    { public_name: "GPT-4.5", api_name: "gpt-4.5" },
  ],
  anthropic: [
    // { public_name: "Claude 3 Haiku", api_name: "claude-3-haiku-20240307" },
    // { public_name: "Claude 3 Sonnet", api_name: "claude-3-sonnet-20240229" },
    // { public_name: "Claude 3 Opus", api_name: "claude-3-opus-20240229" },
    { public_name: "Claude 3.5 Haiku", api_name: "claude-3-5-haiku-20241022" },
    {
      public_name: "Claude 3.5 Sonnet",
      api_name: "claude-3-5-sonnet-20241022",
    },
    {
      public_name: "Claude 3.7 Sonnet",
      api_name: "claude-3-7-sonnet-20250219",
    },
    { public_name: "Claude 4 Sonnet", api_name: "claude-4-sonnet-20250522" },
    { public_name: "Claude 4 Opus", api_name: "claude-4-opus-20250522" },
  ],
  google: [
    {
      public_name: "Gemini 2.5 Flash (Preview)",
      api_name: "models/gemini-2.5-flash-preview-05-20",
    },
    {
      public_name: "Gemini 2.5 Pro (Preview)",
      api_name: "models/gemini-2.5-pro-preview-06-05",
    },
    { public_name: "Gemini 2.0 Flash", api_name: "models/gemini-2.0-flash" },
    {
      public_name: "Gemini 2.0 Flash (stable 001)",
      api_name: "models/gemini-2.0-flash-001",
    },
    {
      public_name: "Gemini 2.0 Flash Experimental",
      api_name: "models/gemini-2.0-flash-exp",
    },
    {
      public_name: "Gemini 2.0 Flash‑Lite",
      api_name: "models/gemini-2.0-flash-lite",
    },
    // { public_name: "Gemini 1.5 Flash", api_name: "models/gemini-1.5-flash" },
    // {
    //   public_name: "Gemini 1.5 Flash 8B",
    //   api_name: "models/gemini-1.5-flash-8b",
    // },
    // { public_name: "Gemini 1.5 Pro", api_name: "models/gemini-1.5-pro" },
  ],
};

export const EMBEDDING_MODELS: Record<
  string,
  { public_name: string; api_name: string }[]
> = {
  openai: [
    {
      public_name: "Text Embedding 3 Small",
      api_name: "text-embedding-3-small",
    },
    {
      public_name: "Text Embedding 3 Large",
      api_name: "text-embedding-3-large",
    },
    {
      public_name: "Text Embedding Ada 002",
      api_name: "text-embedding-ada-002",
    },
  ],
  google: [
    {
      public_name: "Gemini Embedding Experimental",
      api_name: "gemini-embedding-exp-03-07",
    },
    { public_name: "Text Embedding 004", api_name: "text-embedding-004" },
    { public_name: "Embedding 001", api_name: "embedding-001" },
  ],
};
