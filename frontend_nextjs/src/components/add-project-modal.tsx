"use client";

import { useState } from "react";
import { useAuth } from "@/lib/useAuth";
import { fetcher } from "@/utils/fetcher";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { mutate } from "swr";
import GitHubConnection from "./integrations/github-connection";

export function AddProjectModal({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    dbt_cloud_url: "https://cloud.getdbt.com",
    dbt_cloud_account_id: "",
    dbt_cloud_api_key: "",
    name: "",
  });
  const { accessToken } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!accessToken) return;

    setLoading(true);
    toast.info("Connecting to dbt Cloud...");

    try {
      await fetcher(
        "/api/data_sources/projects/create_dbt_cloud_project/",
        accessToken,
        {
          method: "POST",
          body: {
            dbt_cloud_url: formData.dbt_cloud_url,
            dbt_cloud_account_id: parseInt(formData.dbt_cloud_account_id, 10),
            dbt_cloud_api_key: formData.dbt_cloud_api_key,
            name: formData.name,
          },
        }
      );

      toast.success("Project added successfully!");

      // Refresh the projects list and stats
      mutate("/api/data_sources/projects/");
      mutate("/api/data_sources/dashboard-stats/");

      setOpen(false);
      setFormData({
        dbt_cloud_url: "https://cloud.getdbt.com",
        dbt_cloud_account_id: "",
        dbt_cloud_api_key: "",
        name: "",
      });
    } catch (error: any) {
      toast.error(
        `Failed to add project: ${error.info?.error || "Unknown error"}`
      );
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Add dbt Project</DialogTitle>
          <DialogDescription>
            Connect a new dbt project to your Ragstar workspace.
          </DialogDescription>
        </DialogHeader>
        <Tabs defaultValue="dbt-cloud" className="mt-4">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="dbt-cloud">dbt Cloud</TabsTrigger>
            <TabsTrigger value="github">Source Code (GitHub)</TabsTrigger>
          </TabsList>
          <TabsContent value="dbt-cloud" className="mt-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="dbt-cloud-url">dbt Cloud URL</Label>
                <Input
                  id="dbt-cloud-url"
                  placeholder="https://cloud.getdbt.com"
                  value={formData.dbt_cloud_url}
                  onChange={(e) =>
                    setFormData({ ...formData, dbt_cloud_url: e.target.value })
                  }
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="dbt-cloud-account-id">
                  dbt Cloud Account ID
                </Label>
                <Input
                  id="dbt-cloud-account-id"
                  placeholder="123456"
                  value={formData.dbt_cloud_account_id}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      dbt_cloud_account_id: e.target.value,
                    })
                  }
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="dbt-cloud-api-key">dbt Cloud API Key</Label>
                <Input
                  id="dbt-cloud-api-key"
                  type="password"
                  placeholder="••••••••••••••••••••"
                  value={formData.dbt_cloud_api_key}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      dbt_cloud_api_key: e.target.value,
                    })
                  }
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="project-name">Project Name (Optional)</Label>
                <Input
                  id="project-name"
                  placeholder="My dbt Project"
                  value={formData.name}
                  onChange={(e) =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                />
              </div>
              <DialogFooter className="mt-6">
                <Button type="submit" disabled={loading}>
                  {loading ? "Connecting..." : "Connect Project"}
                </Button>
              </DialogFooter>
            </form>
          </TabsContent>
          <TabsContent value="github">
            <GitHubConnection
              onSuccess={() => {
                setOpen(false);
                mutate("/api/data_sources/projects/");
                mutate("/api/data_sources/dashboard-stats/");
              }}
            />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
