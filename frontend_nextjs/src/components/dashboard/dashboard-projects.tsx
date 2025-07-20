"use client";

import useSWR from "swr";
import Link from "next/link";
import { useAuth } from "@/lib/useAuth";
import { fetcher } from "@/utils/fetcher";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowTopRightOnSquareIcon } from "@heroicons/react/24/outline";

interface DbtProject {
  id: number;
  name: string;
  dbt_cloud_account_id: number;
  dbt_cloud_url: string;
  created_at: string;
  updated_at: string;
}

import { toast } from "sonner";

function DeleteProjectButton({
  id,
  mutate,
}: {
  id: number;
  mutate: () => void;
}) {
  const { accessToken } = useAuth();

  const handleDelete = async () => {
    if (!accessToken) return;
    if (
      !confirm(
        "Are you sure you want to delete this project? All associated models will be removed."
      )
    )
      return;
    try {
      await fetcher(`/api/data_sources/projects/${id}/`, accessToken, {
        method: "DELETE",
      });
      toast.success("Project deleted");
      mutate();
    } catch (e) {
      toast.error("Failed to delete project");
      console.error(e);
    }
  };

  return (
    <Button variant="destructive" size="sm" onClick={handleDelete}>
      Delete
    </Button>
  );
}

export function DashboardProjects() {
  const { accessToken, isAuthenticated } = useAuth();

  const { data: projects, mutate } = useSWR<DbtProject[]>(
    isAuthenticated && accessToken ? "/api/data_sources/projects/" : null,
    (url: string) => fetcher(url, accessToken),
    { suspense: true }
  );

  if (!projects || projects.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>dbt Projects</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <p className="text-gray-500 mb-4">No dbt projects connected yet.</p>
            <p className="text-sm text-gray-400">
              Connect your first dbt project to get started with AI-powered data
              analysis.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>dbt Projects</CardTitle>
          <Badge variant="secondary">{projects.length} connected</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {projects.map((project) => (
            <div
              key={project.id}
              className="flex items-center justify-between p-3 border rounded-lg"
            >
              <div>
                <h4 className="font-medium">{project.name}</h4>
                <p className="text-sm text-gray-500">
                  Account ID: {project.dbt_cloud_account_id}
                </p>
                <p className="text-xs text-gray-400">
                  Created: {new Date(project.created_at).toLocaleDateString()}
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant="outline" className="text-green-600">
                  Active
                </Badge>
                <Button variant="ghost" size="sm" asChild>
                  <Link
                    href={`/dashboard/knowledge-base`}
                    className="flex items-center space-x-1"
                  >
                    <span>View Models</span>
                    <ArrowTopRightOnSquareIcon className="h-4 w-4" />
                  </Link>
                </Button>
                <DeleteProjectButton id={project.id} mutate={mutate} />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
