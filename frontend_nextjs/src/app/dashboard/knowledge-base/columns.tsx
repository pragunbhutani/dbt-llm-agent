"use client";

import { ColumnDef } from "@tanstack/react-table";
import { Checkbox } from "@/components/ui/checkbox";
import { DataTableColumnHeader } from "@/components/data-table/data-table-column-header";
import { MoreHorizontal } from "lucide-react";
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
import Link from "next/link";

export type DbtModel = {
  id: string;
  name: string;
  path: string;
  schema_name: string;
  tags: string[];
  updated_at: string;
  answering_status: "Yes" | "No" | "Training";
};

type ColumnsProps = {
  handleToggleAnswering: (
    modelId: string,
    status: DbtModel["answering_status"]
  ) => void;
  handleRefreshModel: (modelId: string) => void;
};

export const getColumns = ({
  handleToggleAnswering,
  handleRefreshModel,
}: ColumnsProps): ColumnDef<DbtModel>[] => [
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
    accessorKey: "name",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Model Name" />
    ),
    cell: ({ row }) => {
      return (
        <Link href={`/dashboard/knowledge-base/models/${row.original.id}`}>
          <div className="min-w-[200px] flex-1 truncate font-medium hover:text-blue-600 cursor-pointer">
            {row.getValue("name")}
          </div>
        </Link>
      );
    },
  },
  {
    accessorKey: "path",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Path" />
    ),
    cell: ({ row }) => {
      return (
        <div className="min-w-[200px] flex-1 truncate">
          {row.getValue("path")}
        </div>
      );
    },
  },
  {
    accessorKey: "schema_name",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Schema" />
    ),
  },
  {
    accessorKey: "tags",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Tags" />
    ),
    cell: ({ row }) => {
      const tags = row.getValue("tags") as string[];
      if (!tags || tags.length === 0) {
        return null;
      }
      return (
        <div className="flex flex-wrap gap-1">
          {tags.map((tag) => (
            <Badge key={tag} variant="outline">
              {tag}
            </Badge>
          ))}
        </div>
      );
    },
  },
  {
    accessorKey: "answering_status",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Answering" />
    ),
    cell: ({ row }) => {
      const status = row.getValue(
        "answering_status"
      ) as DbtModel["answering_status"];
      const variant =
        status === "Yes"
          ? "default"
          : status === "Training"
          ? "secondary"
          : "destructive";
      return <Badge variant={variant}>{status}</Badge>;
    },
    filterFn: (row, id, value) => {
      return value.includes(row.getValue(id));
    },
  },
  {
    accessorKey: "updated_at",
    header: ({ column }) => (
      <DataTableColumnHeader column={column} title="Last Refreshed" />
    ),
    cell: ({ row }) => {
      const date = new Date(row.getValue("updated_at"));
      const formattedDate = date.toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
      return <span>{formattedDate}</span>;
    },
  },
  {
    id: "actions",
    cell: ({ row }) => {
      const model = row.original;
      const status = model.answering_status;
      const isTraining = status === "Training";
      const actionText =
        status === "Yes" ? "Don't use for Answering" : "Use for Answering";

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
              <Link href={`/dashboard/knowledge-base/models/${model.id}`}>
                View Details
              </Link>
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => handleToggleAnswering(model.id, status)}
              disabled={isTraining}
            >
              {actionText}
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => handleRefreshModel(model.id)}
              disabled={isTraining}
            >
              Refresh
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="text-red-600">Delete</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      );
    },
  },
];
