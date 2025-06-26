# Ragstar — AI Data Analyst for dbt Projects

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue.svg)](https://www.typescriptlang.org/)
[![Docker Compose](https://img.shields.io/badge/built%20with-Docker%20Compose-blue.svg)](https://docs.docker.com/compose/)

> **Ragstar is in public βeta.** Expect rapid changes and occasional rough edges.

---

## 1 What is Ragstar?

Ragstar connects to your **dbt** project, builds a knowledge base from models & documentation, and lets everyone ask data-related questions in plain English via a beautiful web dashboard or Slack. Under the hood Ragstar combines:

- PostgreSQL + `pgvector` for fast similarity search
- Embeddings + LLMs (OpenAI, Anthropic, etc.) for reasoning
- A modern **Next.js** frontend & **Django** backend

---

## 2 Quick start (🚀 2 commands)

```bash
# ① clone & prepare env file
$ git clone https://github.com/pragunbhutani/ragstar.git && cd ragstar
$ cp .env.example .env && ${EDITOR:-vi} .env  # ⇒ edit just the vars shown below

# ② build & run everything
$ docker compose up --build -d
```

When the containers are healthy:

- Frontend: http://localhost:3000 (Next.js)
- API: http://localhost:8000 (Django/DRF)
- Flower: http://localhost:5555 (background tasks)

Run first-time Django tasks:

```bash
# inside the running backend container
$ docker compose exec backend-django \
    uv run python manage.py migrate && \
    uv run python manage.py createsuperuser
```

🎉 That's it — open http://localhost:3000, sign in with the super-user you just created, and follow the onboarding wizard.

---

## 3 Environment variables

Only a handful of variables are truly **required** for a local/dev install. The rest are advanced overrides.

### 3.1 Required

| Var                   | Example                   | Purpose                               |
| --------------------- | ------------------------- | ------------------------------------- |
| `AUTH_SECRET`         | `openssl rand -base64 32` | Required by **next-auth** JWT cookies |
| `NEXTAUTH_URL`        | `http://localhost:3000`   | Public URL of the frontend            |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000`   | Public URL of the Django API          |

### 3.2 Optional but common

| Var        | Example                            | When you need it                                                                                                                 |
| ---------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `APP_HOST` | `localhost` or `myapp.example.com` | Backend uses this to extend `ALLOWED_HOSTS` & CORS rules. If omitted, Ragstar will automatically extract it from `NEXTAUTH_URL`. |

> Legacy vars like `APP_PORT`, `FRONTEND_PORT`, etc. are now only used by **docker-compose.yml** for host-port mapping. The application itself does not read them.

### 3.3 Postgres (defaults shown)

<<<<<<< fix/better-query-debugging
If you're happy with the Docker-Compose postgres container you can ignore these — they already default to sensible values.
=======
10. **Initialize your project (for `dbt` core):**
    If you use `dbt` core you might need to set up adapters first.  Sample below for PostgreSQL.

    ```bash
    uv pip install dbt-core dbt-postgres
    uv run python manage.py init_project --method core
    ```

11. **Run the Development Server:**
>>>>>>> main

```
POSTGRES_DB=ragstar
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_PORT=5432
```

**Bring-your-own Postgres?** Set a single `EXTERNAL_POSTGRES_URL` and the compose stack will connect to that instead (make sure `pgvector` is installed).

```
EXTERNAL_POSTGRES_URL=postgresql://user:pw@host:5432/dbname
```

### 3.4 Other useful vars

```
INTERNAL_API_URL=http://backend-django:8000   # only used by server-side Next.js requests
```

---

## 4 First-run onboarding

After logging into the dashboard you'll be guided through these steps:

1. **Add a dbt project** → _Projects › New_ (dbt **Cloud** recommended — just paste the service token). GitHub or local zip upload also supported.
2. **Add LLM provider keys** → _Settings › LLM Providers_ and paste your OpenAI / Anthropic keys.
3. **Pick default models** → choose which model to use for ∙ questions ∙ embeddings ∙ SQL verification.
4. **Configure Slack** (optional)
   - Go to _Integrations › Slack_.
   - Follow the inline manifest to create a Slack app.
   - Paste **Bot Token**, **Signing Secret**, **App Token**.
5. **Ask questions!** Use the chat on the dashboard or `/ask` in Slack.

> Other integrations (Metabase, Snowflake, MCP) are available under _Integrations_ but currently **βeta / experimental**. MCP server is temporarily disabled while we stabilise streaming support.

---

## 5 Managing the stack

Common operations are wrapped in one-liners:

```bash
# shell into backend or frontend
$ docker compose exec backend-django bash
$ docker compose exec frontend-nextjs sh

# tail logs
$ docker compose logs -f backend-django

# stop / remove containers
docker compose down          # keep volumes
docker compose down -v       # destroy DB
```

---

## 6 Local dev without Docker (advanced)

1. Install **Python 3.10+**, **Node 18+**, **uv**, **pnpm**, and Postgres16+ with `pgvector`.
2. `uv venv && source .venv/bin/activate && uv pip install -e backend_django/`
3. `pnpm install --filter frontend_nextjs`
4. Start services in two terminals:
   - **Backend** — `cd backend_django && uv run python manage.py runserver 0.0.0.0:8000`
   - **Frontend** — `cd frontend_nextjs && pnpm dev`
5. Export the same env vars listed above.

Docker is strongly recommended unless you're hacking on the codebase itself.

---

## 7 Contributing

We 💛 community PRs. Please file an issue first for major changes. Make sure `ruff`, `black`, `mypy`, and `eslint` pass before opening a pull request.

---

## 8 License

Ragstar is released under the MIT License — see [LICENSE](./LICENSE).
