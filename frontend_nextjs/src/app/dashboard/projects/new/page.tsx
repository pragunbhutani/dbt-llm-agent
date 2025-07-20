"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { useSession } from "next-auth/react";
import { fetcher } from "@/utils/fetcher";
import GitHubConnection from "@/components/integrations/github-connection";

export default function NewProjectPage() {
  const { data: session } = useSession();
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
      // TODO: Redirect to the projects list page
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
    <div className="flex flex-col items-center justify-center min-h-screen">
      <Tabs defaultValue="dbt-cloud" className="w-[600px]">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="dbt-cloud">dbt Cloud</TabsTrigger>
          <TabsTrigger value="github">Source Code (Github)</TabsTrigger>
        </TabsList>
        <TabsContent value="dbt-cloud">
          <Card>
            <CardHeader>
              <CardTitle>Connect to dbt Cloud</CardTitle>
              <CardDescription>
                Enter your dbt Cloud credentials to connect your project.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="dbt-cloud-url">dbt Cloud URL</Label>
                <Input
                  id="dbt-cloud-url"
                  placeholder="https://cloud.getdbt.com"
                  value={dbtCloudUrl}
                  onChange={(e) => setDbtCloudUrl(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="dbt-cloud-account-id">
                  dbt Cloud Account ID
                </Label>
                <Input
                  id="dbt-cloud-account-id"
                  placeholder="123456"
                  value={dbtCloudAccountId}
                  onChange={(e) => setDbtCloudAccountId(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="dbt-cloud-api-key">dbt Cloud API Key</Label>
                <Input
                  id="dbt-cloud-api-key"
                  type="password"
                  placeholder="••••••••••••••••••••"
                  value={dbtCloudApiKey}
                  onChange={(e) => setDbtCloudApiKey(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="project-name">Project Name (Optional)</Label>
                <Input
                  id="project-name"
                  placeholder="My dbt Project"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                />
              </div>
            </CardContent>
            <CardFooter>
              <Button onClick={handleDbtCloudSubmit} disabled={isLoading}>
                {isLoading ? "Connecting..." : "Connect Project"}
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
        <TabsContent value="github">
          <GitHubConnection />
        </TabsContent>
      </Tabs>
    </div>
  );
}
