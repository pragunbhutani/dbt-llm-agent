"""
ASGI config for ragstar_django project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
import django
from django.core.asgi import get_asgi_application
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.routing import Lifespan, Mount
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware  # If you need CORS
from typing import Optional

# Load environment variables from .env file
load_dotenv()

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ragstar.settings")
# Initialize Django settings and applications
# THIS MUST BE CALLED BEFORE ANY DJANGO-SPECIFIC IMPORTS (LIKE MODELS)
django.setup()

# Attempt to import the agent or its checkpointer
# This path is based on your logs. Adjust if necessary.
try:
    from apps.workflows.question_answerer.workflow import QuestionAnswererAgent
    from psycopg_pool import AsyncConnectionPool  # Import for type checking
except ImportError:
    QuestionAnswererAgent = None
    AsyncConnectionPool = None  # Define if import failed
    print(
        "WARNING: Could not import QuestionAnswererAgent or AsyncConnectionPool for checkpointer setup in asgi.py"
    )


django_asgi_app = get_asgi_application()

# Store the pool globally or in a way that shutdown_event can access it
_global_checkpointer_pool: Optional[AsyncConnectionPool] = None


async def run_checkpointer_setup():
    """Initializes the LangGraph AsyncPostgresSaver checkpointer."""
    global _global_checkpointer_pool
    if QuestionAnswererAgent and AsyncConnectionPool:
        print(
            "INFO: Attempting to setup LangGraph checkpointer via QuestionAnswererAgent..."
        )
        agent = None
        try:
            # Instantiate the agent. If it's a singleton or needs specific config,
            # this might need to be handled differently (e.g., accessing a global instance).
            # For this example, we assume it can be instantiated directly.
            agent = QuestionAnswererAgent()
            checkpointer = getattr(agent, "memory", None)

            if checkpointer:
                pool_to_manage = getattr(checkpointer, "conn", None)

                if isinstance(pool_to_manage, AsyncConnectionPool):
                    print(
                        f"INFO: Found AsyncConnectionPool: {pool_to_manage.name}. Attempting to open..."
                    )
                    await pool_to_manage.open()
                    print(
                        f"INFO: AsyncConnectionPool {pool_to_manage.name} opened successfully."
                    )
                    _global_checkpointer_pool = pool_to_manage  # Store for shutdown
                else:
                    print(
                        "INFO: Checkpointer does not have an AsyncConnectionPool instance at .conn, or AsyncConnectionPool not imported."
                    )

                if hasattr(checkpointer, "setup") and callable(checkpointer.setup):
                    await checkpointer.setup()
                    print("INFO: LangGraph checkpointer.setup() complete.")
                # The langgraph library itself typically uses setup().
                # langchain_postgres (a separate package) uses acreate_tables().
                # Checking for acreate_tables as a fallback or alternative.
                elif hasattr(checkpointer, "acreate_tables") and callable(
                    checkpointer.acreate_tables
                ):
                    await checkpointer.acreate_tables()
                    print("INFO: LangGraph checkpointer.acreate_tables() complete.")
                else:
                    print(
                        "ERROR: QuestionAnswererAgent.memory does not have a recognized setup method (setup or acreate_tables)."
                    )
            else:
                print(
                    "ERROR: QuestionAnswererAgent instance does not have a 'memory' attribute."
                )
        except Exception as e:
            print(f"ERROR: Failed to setup LangGraph checkpointer: {e}")
            import traceback

            traceback.print_exc()
            # If setup failed after pool was opened, try to close it
            if _global_checkpointer_pool and not _global_checkpointer_pool.closed:
                print(
                    f"INFO: Closing pool {_global_checkpointer_pool.name} due to setup error."
                )
                await _global_checkpointer_pool.close()
                _global_checkpointer_pool = None

    else:
        print(
            "INFO: QuestionAnswererAgent or AsyncConnectionPool not available, skipping checkpointer setup."
        )


async def startup_event():
    print("INFO: ASGI application startup...")
    await run_checkpointer_setup()
    print("INFO: ASGI startup tasks complete.")


async def shutdown_event():
    print("INFO: ASGI application shutdown...")
    global _global_checkpointer_pool
    if _global_checkpointer_pool and not _global_checkpointer_pool.closed:
        print(
            f"INFO: Closing checkpointer pool {_global_checkpointer_pool.name} on application shutdown."
        )
        try:
            await _global_checkpointer_pool.close()
            print(f"INFO: Pool {_global_checkpointer_pool.name} closed successfully.")
        except Exception as e:
            print(f"ERROR: Failed to close pool {_global_checkpointer_pool.name}: {e}")
            import traceback

            traceback.print_exc()
    elif _global_checkpointer_pool and _global_checkpointer_pool.closed:
        print(
            f"INFO: Checkpointer pool {_global_checkpointer_pool.name} was already closed."
        )
    else:
        print("INFO: No global checkpointer pool to close or pool not initialized.")

    # Add any cleanup tasks here if needed
    print("INFO: ASGI shutdown tasks complete.")


# Define middleware (optional, example for CORS)
middleware = [
    Middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
]

# Create the Starlette application
application = Starlette(
    on_startup=[startup_event],
    on_shutdown=[shutdown_event],
    middleware=middleware,  # Add middleware if you have any
    routes=[Mount("/", app=django_asgi_app)],  # Mount the Django app
)
