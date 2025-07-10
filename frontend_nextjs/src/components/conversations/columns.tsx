"use client";

import { ColumnDef } from "@tanstack/react-table";
import { Checkbox } from "@/components/ui/checkbox";
import { DataTableColumnHeader } from "@/components/data-table/data-table-column-header";
import { MoreHorizontal, MessageSquare, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import { ConversationListItem } from "@/types/conversations";
import Link from "next/link";

type ColumnsProps = {
  handleDeleteConversation: (conversationId: number) => void;
};

export const getColumns = ({
  handleDeleteConversation,
}: ColumnsProps): ColumnDef<ConversationListItem>[] => [
  {
    id: "select",
    header: ({ table }) => (
      <Checkbox
        checked={
          table.getIsAllPageRowsSelected() ||
          (table.getIsSomePageRowsSelected() && "indeterminate")
        }
        onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
        aria-label="Select all"
      />
    ),
    cell: ({ row }) => (
      <Checkbox
        checked={row.getIsSelected()}
        onCheckedChange={(value) => row.toggleSelected(!!value)}
        aria-label="Select row"
      />
    ),
    enableSorting: false,
    enableHiding: false,
  },
  {
    accessorKey: "initial_question",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Initial Question" />
    ),
    cell: ({ row }) => {
      const question = row.getValue("initial_question") as string;
      const conversationId = row.original.id;

      // Utility to decode a handful of common HTML entities that sometimes
      // appear in questions (e.g. Slack encodes < and >)
      const decodeHtmlEntities = (str: string) =>
        str
          .replace(/&lt;/g, "<")
          .replace(/&gt;/g, ">")
          .replace(/&amp;/g, "&")
          .replace(/&quot;/g, '"')
          .replace(/&#039;/g, "'");

      const decodedQuestion = decodeHtmlEntities(question);

      // Limit the displayed question length further to keep table compact
      const MAX_CHARS = 100;
      const displayQuestion =
        decodedQuestion.length > MAX_CHARS
          ? `${decodedQuestion.slice(0, MAX_CHARS)}…`
          : decodedQuestion;

      return (
        <Link href={`/dashboard/chats/${conversationId}`}>
          {/*
            The combination of max-w and truncate ensures the cell never
            forces the table to exceed the page width while still showing an
            ellipsis for overflow text. We intentionally remove flex-1 so the
            column stops greedily expanding.
          */}
          <div className="min-w-[200px] max-w-[300px] truncate font-medium hover:text-blue-600 cursor-pointer">
            {displayQuestion}
          </div>
        </Link>
      );
    },
  },
  {
    accessorKey: "channel",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Channel" />
    ),
    cell: ({ row }) => {
      const channel = row.getValue("channel") as string;
      const variant =
        channel === "slack"
          ? "default"
          : channel === "web"
          ? "secondary"
          : channel === "mcp"
          ? "outline"
          : "destructive";

      return (
        <Badge variant={variant}>
          {channel.charAt(0).toUpperCase() + channel.slice(1)}
        </Badge>
      );
    },
    filterFn: (row, id, value) => {
      return value.includes(row.getValue(id));
    },
  },
  {
    accessorKey: "trigger",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Trigger" />
    ),
    cell: ({ row }) => {
      const trigger = row.getValue("trigger") as string;
      const displayTrigger = trigger.replace(/_/g, " ");
      return (
        <span className="text-sm text-gray-600">
          {displayTrigger.charAt(0).toUpperCase() + displayTrigger.slice(1)}
        </span>
      );
    },
  },
  // Status and Cost columns removed per updated requirements
  {
    accessorKey: "user_id",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Started By" />
    ),
    cell: ({ row }) => {
      // Prefer user_external_id (display name) but fall back to user_id
      const userName = (row.original as ConversationListItem).user_external_id;
      const userId = row.getValue("user_id") as string;
      return <span className="text-sm">{userName || userId || "Unknown"}</span>;
    },
  },
  {
    accessorKey: "total_parts",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Messages" />
    ),
    cell: ({ row }) => {
      const totalParts = row.getValue("total_parts") as number;
      return (
        <div className="flex items-center gap-1">
          <MessageSquare className="h-4 w-4 text-gray-500" />
          <span className="text-sm">{totalParts}</span>
        </div>
      );
    },
  },
  {
    accessorKey: "started_at",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Started" />
    ),
    cell: ({ row }) => {
      const date = new Date(row.getValue("started_at"));
      const formattedDate = date.toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
      return (
        <div className="flex items-center gap-1">
          <Calendar className="h-4 w-4 text-gray-500" />
          <span className="text-sm">{formattedDate}</span>
        </div>
      );
    },
  },
  {
    id: "actions",
    cell: ({ row }) => {
      const conversation = row.original;

      return (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-8 w-8 p-0">
              <span className="sr-only">Open menu</span>
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Actions</DropdownMenuLabel>
            <DropdownMenuItem asChild>
              <Link href={`/dashboard/chats/${conversation.id}`}>
                View Details
              </Link>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-red-600"
              onClick={() => handleDeleteConversation(conversation.id)}
            >
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      );
    },
  },
];
