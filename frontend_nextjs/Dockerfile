# ---- Base Image ----
# Use a specific Node.js version on Alpine for a small and secure base.
FROM node:20-alpine AS base
WORKDIR /app
# Install pnpm globally for package management.
RUN npm install -g pnpm

# ---- Dependencies Stage ----
# This stage is dedicated to installing all dependencies.
# It's a separate stage to leverage Docker's layer caching.
# If package.json or pnpm-lock.yaml don't change, this layer won't be rebuilt.
FROM base AS dependencies
COPY package.json pnpm-lock.yaml ./
# Install all dependencies, including devDependencies, which are needed for the dev server.
RUN pnpm install --frozen-lockfile

# ---- Development Stage ----
# This stage is for local development. It uses the installed dependencies
# and starts the Next.js development server.
FROM dependencies AS development
# Copy the TypeScript configuration and other config files first
COPY tsconfig.json next.config.ts components.json ./
COPY postcss.config.mjs eslint.config.mjs ./
# Copy the rest of the application source code.
# While docker-compose will mount the local directory, this makes the image runnable on its own.
COPY . .
# Ensure the .next directory exists (permissions inherited from root)
RUN mkdir -p .next

# NOTE: We intentionally run the dev server as root inside the container
# to avoid permission issues when the project directory is bind-mounted
# from the host (UID/GID mismatches can cause EACCES errors when writing
# to .next or node_modules). This only affects the development stage. The
# production stage below still runs as an unprivileged user for security.
# The command to start the dev server with hot-reloading.
CMD ["pnpm", "dev"]

# ---- Builder Stage ----
# This stage builds the production-ready application.
# It uses the dependencies from the 'dependencies' stage and runs the build script.
FROM dependencies AS builder
COPY . .
RUN pnpm build

# ---- Production Stage ----
# This is the final, optimized image for production.
# It copies only the necessary build artifacts from the 'builder' stage.
FROM base AS production
ENV NODE_ENV production
WORKDIR /app
# Copy only the necessary files for a production server.
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/pnpm-lock.yaml ./pnpm-lock.yaml
# Copy the standalone Next.js server output.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
# Install only production dependencies to keep the image small.
RUN pnpm install --prod --frozen-lockfile
# Create a non-root user for security.
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001
USER nextjs
EXPOSE 3000
# The command to start the production server.
CMD ["node", "server.js"] 