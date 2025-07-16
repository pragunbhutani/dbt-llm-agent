---
description:
globs:
alwaysApply: true
---

# Project Description

We are building an LLM powered AI data analyst for Data Engineering teams that work with dbt to manage their analytics code bases. To use this project, users should be able to connect with their dbt cloud projects or their dbt core github repos via an interface which is then used to build a knowlege base. We then use various AI based workflows to allow users to ask questions about their data and then surface the queries, charts or insights needed to answer their data questions.

We will enable interaction through a variety of interfaces including Slack, MCP connectivity with LLM apps, a streaming chat interface and maybe even more in future.

We will also allow users to connect our app with their data warehouses so we can try and answer their questions directly, and support our answers with charts and visualisations etc. If not, we should surface the queries users can use to answer their questions.

We may add more functionality in future or get rid of some of the functionality I've mentioned. I've shared this information for context but we may not be building all of these things at once.

Ask clarifying questions and give your feedback on design decisions objectively - there is no need to agree with every design decision I propose, feel free to provide constructive feedback.

# Monorepo Structure

The project is a monorepo that looks something like this:

/ragstar-project-root/
├── backend_django/ # Django application (moved from root)
│ ├── manage.py
│ ├── ragstar/ # Django settings directory
│ ├── apps/ # Django apps
│ │ ├── accounts # user accounts, organisations, org settings etc.
│ │ ├── data_sources # connection to knowledge sources like dbt
│ │ ├── embeddings # storing and retrieving embeddings
│ │ ├── integrations # integrations to external tools like slack, metabase etc.
│ │ ├── knowledge_base # information about our dbt projects, models, questions etc.
│ │ ├── llm_providers # interface for interacting with LLM provider APIs
│ │ ├── workflows # all workflows, agentic or not
│ │ │ ├── workflow_name # each workflow has a workflow.py, prompts.py etc.
│ │ └── ... # More apps may be found here
│ ├── static/
│ ├── pyproject.toml # Python dependencies
│ ├── uv.lock # Lock file
│ ├── .python-version # Python version specification
│ └── Dockerfile # Backend-specific Docker config
├── frontend_nextjs/ # Renamed from client/ - ready for Next.js
│ ├── public/ # Public static assets
│ ├── src/
│ │ ├── app/ # Next JS App router project structure
│ │ │ ├── (auth)/ # Signin and Signup pages
│ │ │ ├── dashboard/ # Dashboard pages
│ │ │ ├── ...
│ │ ├── components/ # Creates reusable react components
│ │ └── ... # Other reusable utilities should be placed here
│ └── ... # NextJS, Typescript, package.json, eslint etc.
├── mcp_server # FastMCP server (FastMCP is starlette not FastAPI)
├── config_examples/ # Config examples like .slack_manifest.example.json, .ragstarrules.example.yml
├── docs/ # GitHub Pages docs (unchanged)
├── docker-compose.yml # Orchestrates all services
├── .env.example # Example environment file
└── .env # Shared environment variables

## Monorepo Projects

- backend_django
  - The API for our NextJS frontend, MCP server, Authentication, Third party integrations etc.
- frontend_nextjs
  - Admin panel used to configure our app, build and edit the knowledge base, study conversations etc.

### Monorepo Project Rules

#### frontend_nextjs

- We're using the app router with NextJS.
- Use auth.js for authentication.
- Use TailwindCSS classes for style only. Do not define custom or inline css.
- Examine the structure of the src/ folder before creating new directories to avoid duplication.
- The `shadcn-ui` package has been deprecated in favour of `shadcn`

#### backend_django

- All of our workflows, whether agentic or not are stored under apps/workflows.
- Each workflow has a workflow.py, prompts.py and other files that store this infomation consistently.

#### mcp_server

- Basic MCP server built using FastMCP.
  - FastMCP is not FastAPI and FastAPI middleware does not work with FastMCP

## Other Services Used

- postgres db with pgvector
- redis

## Authentication

This project uses next-auth/react on the frontend and Django REST Framework with Simple JWT on the backend. Here's the correct way to handle authenticated API requests from the Next.js client to the Django API:

1. Session Management: Authentication is handled by next-auth. The JWT access token required by the Django backend is made available on the client-side session object.
2. Fetching the Token: In any component that needs to make an authenticated API call, we must use the useSession hook from next-auth/react to get the current session. The access token is located at session.accessToken.
3. Making Authenticated Requests: All API requests to the Django backend must include an Authorization header with the format: Bearer <accessToken>.
   A custom fetcher function should be created for data-fetching libraries like SWR. This function should accept the access token and attach the header to the request.
4. Conditional Fetching: It's crucial to only trigger API requests after the session token is available. Sending requests before the session is established will result in 401 Unauthorized errors. With SWR, this is achieved by making the key conditional, like this:
   useSWR(session?.accessToken ? '/api/endpoint' : null, fetcher)

# Environment Variable Strategy

- Root .env: Shared configuration for all services, used by docker-compose.yml
- Local Development: backend_django/manage.py loads the root .env file
- Docker: docker-compose.yml injects root .env variables into containers

# General Rules

- Use UV for python package management. We're using a pyproject.toml and not requirements.txt.
- You need to put `uv run` before Python commands to make sure they run in the right environment. e.g. instead of `cd backend_django && python manage.py migrate`, run `cd backend_django && uv run python manage.py migrate`.
- Use pnpm for javascript / typescript package management. Make sure you're within the correct monorepo project before running pnpm or uv commands.
- When adding or removing packages, do so via the command line in the appropriate directory instead of directly adding or removing package names to the respective configuration in order to ensure you're always installing the latest package because your training knowledge base may have become outdated.
- Use comments when necessary but don't leave unnecessary comments in your wake like leaving breadcrumbs for all the changes you've done. When you want to remove lines, remove them and don't just comment them out.
- You may not be able to access .env or .env.example so you should ask me to make any changes to those files as required.

# Layout and Page Design

Dashboard layout guidelines (Cursor rule snippet)

1. High-level structure  
   • All dashboard pages render inside `AppShell`, which already provides:  
    – Left-hand fixed nav bar with “active” state.  
    – Right-hand “page container” where each routed page is mounted.  
   • Inside the page container every page should use the shared `PageLayout` component located at `src/components/layout/page-layout.tsx`.

2. `PageLayout` anatomy

   ```
   <PageLayout title="…" subtitle="…" actions={…}>
     …children…
   </PageLayout>
   ```

   • `title` and `subtitle` feed the reusable `Heading` component.  
   • `actions` (optional) renders on the RIGHT side of the header for buttons, status badges, etc.  
   • Children are automatically wrapped in a scrollable `PageBody` section.

3. Header (`PageHeader`) rules  
   • Fixed (`sticky top-0 z-10`), white background, bottom border (`border-b border-gray-200`).  
   • Padding: `px-2 py-2` on mobile → `lg:px-4` on large screens.  
   • NOTHING (buttons, icons, breadcrumbs) is placed to the LEFT of the title/subtitle.  
   • Optional right-hand `actions` only.

4. Body (`PageBody`) rules  
   • Occupies remaining height with `flex-1 overflow-auto`.  
   • Background: `bg-gray-50`.  
   • Default inner padding `px-2 py-4` → `lg:px-4`.  
   • All scrollable content lives here so the header never scrolls away.

5. Breadcrumbs  
   • Optional. When used, place at the very top of the body, before main content.  
   • Breadcrumbs should not be placed on top level pages like /integrations, but on second level onwards e.g. /integrations/slack
   • Use shared `<Breadcrumb>` component.  
   • Do NOT put breadcrumbs in the header.

6. Sizing & spacing conventions  
   • Default horizontal padding already provided by `PageBody`; avoid extra wrapping `p-4` unless you need additional top/bottom spacing.  
   • If you need a full-width section use negative margins or a dedicated component, not raw HTML outside `PageBody`.

7. Right-hand header actions  
   • Keep them compact: use `size="sm"` buttons, badges, etc.  
   • Multiple elements should be grouped with `flex items-center gap-3`.

8. No “Back” buttons in header  
   • Navigation is via breadcrumbs inside the body or sidebar links—do not add Arrow-Left buttons beside the title.

9. Location of layout code  
   • All layout primitives live in `src/components/layout/`.  
   • Individual pages must not redefine their own header/body; instead import and compose.

10. Example skeleton for a new page

```tsx
import PageLayout from "@/components/layout/page-layout";
import { Breadcrumb } from "@/components/ui/breadcrumb";

export default function MyPage() {
  return (
    <PageLayout
      title="My Feature"
      subtitle="What this feature does"
      actions={<MyHeaderButtons />}
    >
      <Breadcrumb items={[{ label: "My Feature" }]} className="mb-4" />

      {/* page content */}
    </PageLayout>
  );
}
```

Documenting the above in your Cursor rules file will reinforce consistent, polished dashboards across the project.
