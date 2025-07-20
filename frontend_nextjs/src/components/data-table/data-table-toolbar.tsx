"use client";

import { Table } from "@tanstack/react-table";
import { Input } from "@/components/ui/input";
import { DataTableViewOptions } from "./data-table-view-options";
import * as React from "react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { RefreshCw, DatabaseZap, Trash2, ChevronDown } from "lucide-react";

interface FilterOption {
  value: string;
  label: string;
}

interface BulkAction {
  key: string;
  label: string;
  icon?: React.ReactNode;
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost";
}

interface DataTableToolbarProps<TData> {
  table: Table<TData>;
  filterOptions?: FilterOption[];
  bulkActions?: BulkAction[];
  onBulkAction?: (action: string, selectedRows: TData[]) => void;
  leadingComponents?: React.ReactNode;
}

export function DataTableToolbar<TData>({
  table,
  filterOptions = [
    { value: "name", label: "Name" },
    { value: "path", label: "Path" },
    { value: "schema_name", label: "Schema" },
    { value: "tags", label: "Tags" },
  ],
  bulkActions = [
    {
      key: "refresh",
      label: "Refresh",
      icon: <RefreshCw className="mr-2 h-4 w-4" />,
    },
    { key: "enable", label: "Use for Answering" },
    { key: "disable", label: "Don't use for Answering" },
    {
      key: "delete",
      label: "Delete",
      variant: "destructive" as const,
      icon: <Trash2 className="mr-2 h-4 w-4" />,
    },
  ],
  onBulkAction,
  leadingComponents,
}: DataTableToolbarProps<TData>) {
  const [filterBy, setFilterBy] = React.useState(filterOptions[0]?.value || "");
  const numSelected = table.getFilteredSelectedRowModel().rows.length;

  const handleFilterChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    table.getColumn(filterBy)?.setFilterValue(event.target.value);
  };

  const handleSelectChange = (value: string) => {
    // Clear the filter value of the previous column
    table.getColumn(filterBy)?.setFilterValue("");
    setFilterBy(value);
  };

  const handleBulkActionClick = (actionKey: string) => {
    if (onBulkAction) {
      onBulkAction(
        actionKey,
        table.getFilteredSelectedRowModel().rows.map((row) => row.original)
      );
    }
  };

  // Group bulk actions by type for better UI organization
  const primaryActions = bulkActions.filter(
    (action) =>
      action.key === "refresh" ||
      action.key === "enable" ||
      action.key === "disable"
  );
  const dangerousActions = bulkActions.filter(
    (action) => action.variant === "destructive" || action.key === "delete"
  );

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-2">
        <DataTableViewOptions table={table} />
        {leadingComponents}
        {filterOptions.length > 0 && (
          <>
            {filterOptions.length > 1 && (
              <Select value={filterBy} onValueChange={handleSelectChange}>
                <SelectTrigger className="h-9 w-[120px]">
                  <SelectValue placeholder="Filter by" />
                </SelectTrigger>
                <SelectContent>
                  {filterOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            <Input
              placeholder={`Filter by ${
                filterOptions
                  .find((opt) => opt.value === filterBy)
                  ?.label?.toLowerCase() || filterBy
              }...`}
              value={
                (table.getColumn(filterBy)?.getFilterValue() as string) ?? ""
              }
              onChange={handleFilterChange}
              className="h-9"
            />
          </>
        )}
      </div>

      {bulkActions.length > 0 && (
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium text-muted-foreground">
            Bulk Actions
          </span>

          {/* Primary Actions */}
          {primaryActions.map((action) => {
            if (action.key === "refresh") {
              return (
                <Button
                  key={action.key}
                  variant="outline"
                  size="sm"
                  className="ml-2 h-9"
                  disabled={numSelected === 0}
                  onClick={() => handleBulkActionClick(action.key)}
                >
                  {action.icon}
                  {action.label}
                </Button>
              );
            }
            return null;
          })}

          {/* Enable/Disable Dropdown */}
          {primaryActions.some(
            (action) => action.key === "enable" || action.key === "disable"
          ) && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-9"
                  disabled={numSelected === 0}
                >
                  <DatabaseZap className="mr-2 h-4 w-4" />
                  Answering
                  <ChevronDown className="ml-2 h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {primaryActions
                  .filter(
                    (action) =>
                      action.key === "enable" || action.key === "disable"
                  )
                  .map((action) => (
                    <DropdownMenuItem
                      key={action.key}
                      onClick={() => handleBulkActionClick(action.key)}
                    >
                      {action.label}
                    </DropdownMenuItem>
                  ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}

          {/* Dangerous Actions */}
          {dangerousActions.length > 0 && (
            <>
              <span className="text-gray-300">|</span>
              {dangerousActions.map((action) => (
                <Button
                  key={action.key}
                  variant={action.variant || "destructive"}
                  size="sm"
                  className="h-9"
                  disabled={numSelected === 0}
                  onClick={() => handleBulkActionClick(action.key)}
                >
                  {action.icon}
                  {action.label}
                </Button>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
