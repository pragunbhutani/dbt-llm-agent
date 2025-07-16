"use client";

export default function BentoFeatures() {
  return (
    <div id="features" className="bg-white py-16 sm:py-24">
      <div className="mx-auto max-w-2xl px-6 lg:max-w-7xl lg:px-8">
        <h2 className="text-base/7 font-semibold text-indigo-600">
          Production-Ready
        </h2>
        <p className="mt-2 max-w-lg text-4xl font-semibold tracking-tight text-pretty text-gray-950 sm:text-5xl">
          Built for real data teams
        </p>
        <div className="mt-10 grid grid-cols-1 gap-4 sm:mt-16 lg:grid-cols-6 lg:grid-rows-2">
          {/* dbt Cloud & Core */}
          <div className="relative lg:col-span-3">
            <div className="absolute inset-0 rounded-lg bg-white max-lg:rounded-t-4xl lg:rounded-tl-4xl" />
            <div className="relative flex h-full flex-col overflow-hidden rounded-[calc(var(--radius-lg)+1px)] max-lg:rounded-t-[calc(2rem+1px)] lg:rounded-tl-[calc(2rem+1px)]">
              <div className="bg-gradient-to-r from-indigo-500 to-purple-600 p-8">
                <div className="text-white">
                  <div className="text-xs font-mono bg-white/20 rounded px-2 py-1 inline-block mb-3">
                    dbt_project.yml
                  </div>
                  <div className="text-sm font-mono space-y-1">
                    <div>models/</div>
                    <div>├── marts/</div>
                    <div>│ ├── core/</div>
                    <div>│ │ ├── dim_customers.sql</div>
                    <div>│ │ └── fct_orders.sql</div>
                    <div>└── staging/</div>
                  </div>
                </div>
              </div>
              <div className="p-10 pt-4">
                <h3 className="text-sm/4 font-semibold text-indigo-600">
                  dbt Cloud & Core
                </h3>
                <p className="mt-2 text-lg font-medium tracking-tight text-gray-950">
                  Works with both dbt Cloud and dbt Core
                </p>
                <p className="mt-2 max-w-lg text-sm/6 text-gray-600">
                  Seamlessly integrates with your existing dbt setup, regardless
                  of hosting.
                </p>
              </div>
            </div>
            <div className="pointer-events-none absolute inset-0 rounded-lg shadow-sm outline outline-black/5 max-lg:rounded-t-4xl lg:rounded-tl-4xl" />
          </div>

          {/* Smart Documentation */}
          <div className="relative lg:col-span-3">
            <div className="absolute inset-0 rounded-lg bg-white lg:rounded-tr-4xl" />
            <div className="relative flex h-full flex-col overflow-hidden rounded-[calc(var(--radius-lg)+1px)] lg:rounded-tr-[calc(2rem+1px)]">
              <div className="bg-gradient-to-r from-green-500 to-teal-600 p-8">
                <div className="text-white space-y-3">
                  <div className="text-sm">
                    🧠 Schema inference{"\n"}
                    📊 Column analysis{"\n"}
                    🔍 Pattern recognition{"\n"}⚡ Auto-documentation
                  </div>
                </div>
              </div>
              <div className="p-10 pt-4">
                <h3 className="text-sm/4 font-semibold text-indigo-600">
                  Smart Documentation
                </h3>
                <p className="mt-2 text-lg font-medium tracking-tight text-gray-950">
                  Understands your project without docs
                </p>
                <p className="mt-2 max-w-lg text-sm/6 text-gray-600">
                  Infers meaning from schema, column names, and data patterns
                  when documentation is missing.
                </p>
              </div>
            </div>
            <div className="pointer-events-none absolute inset-0 rounded-lg shadow-sm outline outline-black/5 lg:rounded-tr-4xl" />
          </div>

          {/* Slack Integration */}
          <div className="relative lg:col-span-2">
            <div className="absolute inset-0 rounded-lg bg-white lg:rounded-bl-4xl" />
            <div className="relative flex h-full flex-col overflow-hidden rounded-[calc(var(--radius-lg)+1px)] lg:rounded-bl-[calc(2rem+1px)]">
              <div className="bg-gradient-to-r from-orange-500 to-red-600 p-8">
                <div className="text-white">
                  <div className="text-xs font-mono bg-white/20 rounded px-2 py-1 inline-block mb-3">
                    #data-team
                  </div>
                  <div className="text-sm">
                    💬 Native Slack threads{"\n"}⚡ Instant responses{"\n"}
                    📊 Rich formatting
                  </div>
                </div>
              </div>
              <div className="p-10 pt-4">
                <h3 className="text-sm/4 font-semibold text-indigo-600">
                  Slack Native
                </h3>
                <p className="mt-2 text-lg font-medium tracking-tight text-gray-950">
                  Stay where your team already is
                </p>
                <p className="mt-2 max-w-lg text-sm/6 text-gray-600">
                  Ask questions directly in Slack threads and get formatted
                  responses instantly.
                </p>
              </div>
            </div>
            <div className="pointer-events-none absolute inset-0 rounded-lg shadow-sm outline outline-black/5 lg:rounded-bl-4xl" />
          </div>

          {/* Snowflake Integration */}
          <div className="relative lg:col-span-2">
            <div className="absolute inset-0 rounded-lg bg-white" />
            <div className="relative flex h-full flex-col overflow-hidden rounded-[calc(var(--radius-lg)+1px)]">
              <div className="bg-gradient-to-r from-blue-500 to-cyan-600 p-8">
                <div className="text-white text-sm">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-3 h-3 bg-white rounded-full"></div>
                    <span>Snowflake connected</span>
                  </div>
                  <div className="text-xs font-mono bg-white/20 rounded px-2 py-1 inline-block">
                    /query SELECT * FROM dim_customers
                  </div>
                </div>
              </div>
              <div className="p-10 pt-4">
                <h3 className="text-sm/4 font-semibold text-indigo-600">
                  Snowflake Shortcuts
                </h3>
                <p className="mt-2 text-lg font-medium tracking-tight text-gray-950">
                  Execute queries via Slack
                </p>
                <p className="mt-2 max-w-lg text-sm/6 text-gray-600">
                  Run queries directly from Slack using shortcuts, no context
                  switching needed.
                </p>
              </div>
            </div>
            <div className="pointer-events-none absolute inset-0 rounded-lg shadow-sm outline outline-black/5" />
          </div>

          {/* Metabase Charts */}
          <div className="relative lg:col-span-2">
            <div className="absolute inset-0 rounded-lg bg-white max-lg:rounded-b-4xl lg:rounded-br-4xl" />
            <div className="relative flex h-full flex-col overflow-hidden rounded-[calc(var(--radius-lg)+1px)] max-lg:rounded-b-[calc(2rem+1px)] lg:rounded-br-[calc(2rem+1px)]">
              <div className="bg-gradient-to-r from-purple-500 to-pink-600 p-8">
                <div className="text-white">
                  <div className="text-xs font-mono bg-white/20 rounded px-2 py-1 inline-block mb-3">
                    /chart revenue by month
                  </div>
                  <div className="text-sm">
                    📊 Auto-generated charts{"\n"}
                    🎨 Beautiful visualizations{"\n"}
                    🔗 Metabase integration
                  </div>
                </div>
              </div>
              <div className="p-10 pt-4">
                <h3 className="text-sm/4 font-semibold text-indigo-600">
                  Metabase Charts
                </h3>
                <p className="mt-2 text-lg font-medium tracking-tight text-gray-950">
                  Create charts via Slack shortcuts
                </p>
                <p className="mt-2 max-w-lg text-sm/6 text-gray-600">
                  Generate beautiful charts in Metabase directly from Slack
                  commands.
                </p>
              </div>
            </div>
            <div className="pointer-events-none absolute inset-0 rounded-lg shadow-sm outline outline-black/5 max-lg:rounded-b-4xl lg:rounded-br-4xl" />
          </div>
        </div>

        {/* MCP Server - Alpha Feature */}
        <div className="mt-20 mx-auto max-w-2xl">
          <div className="relative">
            <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-amber-50 to-yellow-50 border border-amber-200" />
            <div className="relative flex h-full flex-col overflow-hidden rounded-lg p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="px-2 py-1 bg-amber-100 text-amber-800 text-xs font-medium rounded">
                  ALPHA
                </div>
                <h3 className="text-lg font-semibold text-gray-900">
                  MCP Server for Claude.ai (self-hosted only)
                </h3>
              </div>
              <p className="text-gray-600 text-sm">
                Remote MCP server integration allows Claude.ai to directly query
                your dbt models and analyze your data through a secure API
                connection.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
