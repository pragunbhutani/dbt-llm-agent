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

interface DataTableToolbarProps<TData> {
  table: Table<TData>;
  onBulkAction?: (
    action: "enable" | "disable" | "refresh",
    selectedRows: TData[]
  ) => void;
}

export function DataTableToolbar<TData>({
  table,
  onBulkAction,
}: DataTableToolbarProps<TData>) {
  const [filterBy, setFilterBy] = React.useState("name");
  const numSelected = table.getFilteredSelectedRowModel().rows.length;

  const handleFilterChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    table.getColumn(filterBy)?.setFilterValue(event.target.value);
  };

  const handleSelectChange = (value: string) => {
    // Clear the filter value of the previous column
    table.getColumn(filterBy)?.setFilterValue("");
    setFilterBy(value);
  };

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-2">
        <DataTableViewOptions table={table} />
        <Input
          placeholder={`Filter by ${filterBy}...`}
          value={(table.getColumn(filterBy)?.getFilterValue() as string) ?? ""}
          onChange={handleFilterChange}
          className="h-9 w-[150px] lg:w-[250px]"
        />
        <Select value={filterBy} onValueChange={handleSelectChange}>
          <SelectTrigger className="h-9 w-[120px]">
            <SelectValue placeholder="Filter by" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="name">Name</SelectItem>
            <SelectItem value="path">Path</SelectItem>
            <SelectItem value="schema_name">Schema</SelectItem>
            <SelectItem value="tags">Tags</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="flex items-center space-x-2">
        <span className="text-sm font-medium text-muted-foreground">
          Bulk Actions
        </span>
        <Button
          variant="outline"
          size="sm"
          className="ml-2 h-9"
          disabled={numSelected === 0}
          onClick={() =>
            onBulkAction &&
            onBulkAction(
              "refresh",
              table
                .getFilteredSelectedRowModel()
                .rows.map((row) => row.original)
            )
          }
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>

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
            <DropdownMenuItem
              onClick={() =>
                onBulkAction &&
                onBulkAction(
                  "enable",
                  table
                    .getFilteredSelectedRowModel()
                    .rows.map((row) => row.original)
                )
              }
            >
              Use for Answering
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() =>
                onBulkAction &&
                onBulkAction(
                  "disable",
                  table
                    .getFilteredSelectedRowModel()
                    .rows.map((row) => row.original)
                )
              }
            >
              Don&apos;t use for Answering
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        <span className="text-gray-300">|</span>
        <Button
          variant="destructive"
          size="sm"
          className="h-9"
          disabled={numSelected === 0}
        >
          <Trash2 className="mr-2 h-4 w-4" />
          Delete
        </Button>
      </div>
    </div>
  );
}
