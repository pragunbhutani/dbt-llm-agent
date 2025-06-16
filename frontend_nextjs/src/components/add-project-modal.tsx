"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
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
import { fetcher } from "@/utils/fetcher";

interface AddProjectModalProps {
  children: React.ReactNode;
}

export function AddProjectModal({ children }: AddProjectModalProps) {
  const { data: session } = useSession();
  const [open, setOpen] = useState(false);
  const [dbtCloudUrl, setDbtCloudUrl] = useState("");
  const [dbtCloudAccountId, setDbtCloudAccountId] = useState("");
  const [dbtCloudApiKey, setDbtCloudApiKey] = useState("");
  const [projectName, setProjectName] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleDbtCloudSubmit = async () => {
    setIsLoading(true);
    toast.info("Connecting to dbt Cloud...");

    try {
      await fetcher(
        "/api/data_sources/projects/create_dbt_cloud_project/",
        session?.accessToken,
        {
          method: "POST",
          body: {
            dbt_cloud_url: dbtCloudUrl,
            dbt_cloud_account_id: parseInt(dbtCloudAccountId, 10),
            dbt_cloud_api_key: dbtCloudApiKey,
            name: projectName,
          },
        }
      );

      toast.success("Project added successfully!");
      setOpen(false); // Close modal on success
    } catch (error: any) {
      toast.error(
        `Failed to add project: ${error.info?.error || "Unknown error"}`
      );
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="sm:max-w-[600px]">
        <Tabs defaultValue="dbt-cloud" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="dbt-cloud">dbt Cloud</TabsTrigger>
            <TabsTrigger value="github" disabled>
              Source Code (Github)
            </TabsTrigger>
          </TabsList>
          <TabsContent value="dbt-cloud">
            <DialogHeader>
              <DialogTitle>Connect to dbt Cloud</DialogTitle>
              <DialogDescription>
                Enter your dbt Cloud credentials to connect your project.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="dbt-cloud-url" className="text-right">
                  URL
                </Label>
                <Input
                  id="dbt-cloud-url"
                  placeholder="https://cloud.getdbt.com"
                  className="col-span-3"
                  value={dbtCloudUrl}
                  onChange={(e) => setDbtCloudUrl(e.target.value)}
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="dbt-cloud-account-id" className="text-right">
                  Account ID
                </Label>
                <Input
                  id="dbt-cloud-account-id"
                  placeholder="123456"
                  className="col-span-3"
                  value={dbtCloudAccountId}
                  onChange={(e) => setDbtCloudAccountId(e.target.value)}
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="dbt-cloud-api-key" className="text-right">
                  API Key
                </Label>
                <Input
                  id="dbt-cloud-api-key"
                  type="password"
                  placeholder="••••••••••••••••••••"
                  className="col-span-3"
                  value={dbtCloudApiKey}
                  onChange={(e) => setDbtCloudApiKey(e.target.value)}
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="project-name" className="text-right">
                  Project Name
                </Label>
                <Input
                  id="project-name"
                  placeholder="My dbt Project (Optional)"
                  className="col-span-3"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                />
              </div>
            </div>
            <DialogFooter>
              <Button onClick={handleDbtCloudSubmit} disabled={isLoading}>
                {isLoading ? "Connecting..." : "Connect Project"}
              </Button>
            </DialogFooter>
          </TabsContent>
          <TabsContent value="github">
            <DialogHeader>
              <DialogTitle>Connect with GitHub</DialogTitle>
              <DialogDescription>
                This feature is coming soon.
              </DialogDescription>
            </DialogHeader>
            <div className="py-4">
              <p>
                You&apos;ll be able to connect your dbt project from a GitHub
                repository here.
              </p>
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
