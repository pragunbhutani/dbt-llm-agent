# Stage 1: Builder
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

# Copy package.json and pnpm-lock.yaml
COPY package.json pnpm-lock.yaml ./

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy the rest of the application code
COPY . .

# Build the Next.js application
RUN pnpm build

# Stage 2: Runner
FROM node:20-alpine AS runner

WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

# Copy package.json and pnpm-lock.yaml for pnpm start
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/pnpm-lock.yaml ./pnpm-lock.yaml

# Create a non-root user and group
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Copy built assets from the builder stage
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder /app/next.config.ts ./next.config.ts

# Ensure node_modules are available for 'pnpm start' if it needs runtime dependencies not bundled by Next.js build.
# Typically, 'next start' relies on the .next standalone output or its own server capabilities,
# but having node_modules from a pruned production install can be safer.
# We will install only production dependencies here.
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --prod --frozen-lockfile

# Change ownership of the app directory
RUN chown -R nextjs:nodejs /app

# Switch to the non-root user
USER nextjs

# Expose the port Next.js runs on
EXPOSE 3000

# Set the command to start the Next.js application
CMD ["pnpm", "start"] 