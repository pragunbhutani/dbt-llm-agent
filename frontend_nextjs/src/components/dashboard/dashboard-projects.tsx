"use client";

import { useState } from "react";
import useSWR from "swr";
import { useSession } from "next-auth/react";
import { fetcher } from "@/utils/fetcher";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export function DashboardProjects() {
  const { data: session } = useSession();
  const [actioningProject, setActioningProject] = useState<{
    id: string;
    action: "refresh" | "delete";
  } | null>(null);

  const { data: projectsData, mutate: mutateProjects } = useSWR(
    session?.accessToken ? "/api/data_sources/projects/" : null,
    (url: string) => fetcher(url, session?.accessToken),
    { suspense: true }
  );

  const projects = projectsData || [];

  const handleRefresh = async (projectId: string) => {
    if (!session?.accessToken) return;
    setActioningProject({ id: projectId, action: "refresh" });
    try {
      await fetcher(
        `/api/data_sources/projects/${projectId}/refresh/`,
        session.accessToken,
        { method: "POST" }
      );
      mutateProjects();
    } catch (error) {
      console.error("Failed to refresh project", error);
      // TODO: Add user-facing error feedback
    } finally {
      setActioningProject(null);
    }
  };

  const handleDelete = async (projectId: string) => {
    if (!session?.accessToken) return;
    setActioningProject({ id: projectId, action: "delete" });
    try {
      await fetcher(
        `/api/data_sources/projects/${projectId}/`,
        session.accessToken,
        { method: "DELETE" }
      );
      await mutateProjects();
    } catch (error) {
      console.error("Failed to delete project", error);
      // TODO: Add user-facing error feedback
    } finally {
      setActioningProject(null);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Connected dbt Projects</CardTitle>
        <CardDescription>
          Manage your connected dbt projects and their status.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Project Name</TableHead>
              <TableHead>Connection Type</TableHead>
              <TableHead>Models Synced</TableHead>
              <TableHead>Last Refreshed</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {projects.length > 0 ? (
              projects.map((project: any) => (
                <TableRow key={project.id}>
                  <TableCell className="font-medium">{project.name}</TableCell>
                  <TableCell>{project.connection_type}</TableCell>
                  <TableCell>N/A</TableCell>
                  <TableCell>
                    {new Date(project.updated_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    <div className="flex space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleRefresh(project.id)}
                        disabled={actioningProject?.id === project.id}
                      >
                        {actioningProject?.id === project.id &&
                        actioningProject?.action === "refresh" ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : null}
                        Refresh
                      </Button>
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => handleDelete(project.id)}
                        disabled={actioningProject?.id === project.id}
                      >
                        {actioningProject?.id === project.id &&
                        actioningProject?.action === "delete" ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : null}
                        Delete
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={5} className="text-center">
                  No dbt projects found.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
