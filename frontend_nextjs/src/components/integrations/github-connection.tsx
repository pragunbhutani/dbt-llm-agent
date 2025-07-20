"use client";

import { useEffect, useState } from "react";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useAuth } from "@/lib/useAuth";
import useSWR from "swr";
import { fetcher } from "@/utils/fetcher";
import { toast } from "sonner";

interface GitHubRepo {
  name: string;
  full_name: string;
}

interface GitHubOwner {
  login: string;
  type: string;
}

export default function GitHubConnection({
  onSuccess,
}: {
  onSuccess?: () => void;
}) {
  const { accessToken } = useAuth();
  const [selectedRepo, setSelectedRepo] = useState<string>("");
  const [owner, setOwner] = useState<string>("");
  const [branch, setBranch] = useState<string>("main");
  const [projectName, setProjectName] = useState<string>("");
  const [projectFolder, setProjectFolder] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const handleConnect = async () => {
    if (!accessToken) return;

    try {
      const data = await fetcher(
        "/api/integrations/github/install/",
        accessToken
      );
      if (data.authorization_url) {
        window.location.href = data.authorization_url;
      }
    } catch (error) {
      toast.error("Could not connect to GitHub. Please try again.");
      console.error(error);
    }
  };

  // Owners list
  const { data: owners, error: ownersError } = useSWR<GitHubOwner[]>(
    accessToken ? "/api/integrations/github/owners/" : null,
    (url: string) => fetcher(url, accessToken)
  );

  // Set default owner when owners list loads
  useEffect(() => {
    if (owners && owners.length > 0 && !owner) {
      setOwner(owners[0].login);
    }
  }, [owners]);

  // Check for GitHub connection status
  const { data: githubStatus, error: githubStatusError } = useSWR(
    accessToken
      ? "/api/integrations/organisation-integrations/status/?integration_keys=github"
      : null,
    (url: string) => fetcher(url, accessToken)
  );

  useEffect(() => {
    console.log("GitHub Status:", githubStatus);
  }, [githubStatus]);

  const isConnected = githubStatus?.[0]?.is_enabled;

  // Fetch repositories if connected
  const { data: repos, error: reposError } = useSWR<GitHubRepo[]>(
    isConnected && owner
      ? `/api/integrations/github/repositories/?owner=${owner}`
      : null,
    (url: string) => fetcher(url, accessToken)
  );

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("github_status") === "success") {
      toast.success("GitHub connected successfully!");
      // Clean the URL
      window.history.replaceState(null, "", window.location.pathname);
    }
  }, []);

  const handleCreateProject = async () => {
    setIsLoading(true);
    toast.info("Creating project from GitHub repository...");

    try {
      await fetcher(
        "/api/data_sources/projects/create_github_project/",
        accessToken,
        {
          method: "POST",
          body: {
            name: projectName,
            github_repository_url: `https://github.com/${selectedRepo}`,
            github_branch: branch,
            github_project_folder: projectFolder,
          },
        }
      );

      toast.success("Project creation started successfully!");

      // Trigger parent success handler (e.g., close modal & refresh lists)
      if (onSuccess) {
        onSuccess();
      }
    } catch (error: any) {
      toast.error(
        `Failed to create project: ${error.info?.error || "Unknown error"}`
      );
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  if (githubStatusError) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Error</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Could not load GitHub integration status.</p>
        </CardContent>
      </Card>
    );
  }

  if (githubStatus === undefined) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Loading...</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Checking GitHub connection status...</p>
        </CardContent>
      </Card>
    );
  }

  if (!isConnected) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Connect with GitHub</CardTitle>
          <CardDescription>
            Connect your GitHub account to select a dbt project repository.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={handleConnect}>Connect to GitHub</Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Create project from GitHub</CardTitle>
        <CardDescription>
          Select a repository containing your dbt project.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="owner-select">Owner</Label>
          <Select
            onValueChange={(val) => {
              setOwner(val);
              setSelectedRepo("");
            }}
            value={owner}
          >
            <SelectTrigger id="owner-select">
              <SelectValue placeholder="Select an owner" />
            </SelectTrigger>
            <SelectContent>
              {ownersError && (
                <SelectItem value="error" disabled>
                  Could not load owners
                </SelectItem>
              )}
              {owners ? (
                owners.map((o) => (
                  <SelectItem key={o.login} value={o.login}>
                    {o.login} ({o.type})
                  </SelectItem>
                ))
              ) : (
                <SelectItem value="loading" disabled>
                  Loading owners...
                </SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="repo-select">Repository</Label>
          <Select onValueChange={setSelectedRepo} value={selectedRepo}>
            <SelectTrigger id="repo-select">
              <SelectValue placeholder="Select a repository" />
            </SelectTrigger>
            <SelectContent>
              {reposError && (
                <SelectItem value="error" disabled>
                  Could not load repositories
                </SelectItem>
              )}
              {repos ? (
                repos.map((repo) => (
                  <SelectItem key={repo.full_name} value={repo.full_name}>
                    {repo.full_name}
                  </SelectItem>
                ))
              ) : (
                <SelectItem value="loading" disabled>
                  Loading repositories...
                </SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label htmlFor="branch-name">Branch</Label>
          <Input
            id="branch-name"
            placeholder="main"
            value={branch}
            onChange={(e) => setBranch(e.target.value)}
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="project-folder">Project Folder (Optional)</Label>
          <Input
            id="project-folder"
            placeholder="e.g., dbt_project"
            value={projectFolder}
            onChange={(e) => setProjectFolder(e.target.value)}
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="project-name">Project Name</Label>
          <Input
            id="project-name"
            placeholder="My GitHub dbt Project"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
          />
        </div>
      </CardContent>
      <CardFooter>
        <Button
          onClick={handleCreateProject}
          disabled={isLoading || !selectedRepo || !projectName}
        >
          {isLoading ? "Creating..." : "Create Project"}
        </Button>
      </CardFooter>
    </Card>
  );
}
