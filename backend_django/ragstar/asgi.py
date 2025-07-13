"""
ASGI config for ragstar_django project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
import django
from django.core.asgi import get_asgi_application
from starlette.applications import Starlette
from starlette.routing import Lifespan, Mount
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware  # If you need CORS
from typing import Optional
from contextlib import asynccontextmanager
from django.conf import settings

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ragstar.settings")
# Initialize Django settings and applications
# THIS MUST BE CALLED BEFORE ANY DJANGO-SPECIFIC IMPORTS (LIKE MODELS)
django.setup()

# Attempt to import the agent or its checkpointer
# This path is based on your logs. Adjust if necessary.
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool  # Import for type checking
except ImportError:
    AsyncPostgresSaver = None
    AsyncConnectionPool = None  # Define if import failed
    print(
        "WARNING: Could not import LangGraph components for checkpointer setup in asgi.py"
    )


django_asgi_app = get_asgi_application()

# Store the pool globally or in a way that shutdown_event can access it
_global_checkpointer_pool: Optional[AsyncConnectionPool] = None


@asynccontextmanager
async def lifespan(app: Starlette):
    """
    Application lifespan context manager.
    Handles startup and shutdown events for the application.
    """
    # Startup logic
    print("INFO: ASGI application startup...")
    await run_checkpointer_setup()
    print("INFO: ASGI startup tasks complete.")

    yield  # Application is now running

    # Shutdown logic
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


async def run_checkpointer_setup():
    """Initializes the LangGraph AsyncPostgresSaver checkpointer."""
    global _global_checkpointer_pool
    if AsyncPostgresSaver and AsyncConnectionPool:
        print("INFO: Attempting to setup LangGraph checkpointer...")
        try:
            db_settings = settings.DATABASES["default"]
            pg_conn_string = f"postgresql://{db_settings['USER']}:{db_settings['PASSWORD']}@{db_settings['HOST']}:{db_settings['PORT']}/{db_settings['NAME']}"

            # Create and open the connection pool
            pool = AsyncConnectionPool(
                conninfo=pg_conn_string,
                max_size=20,
                min_size=5,
                open=False,
            )
            await pool.open()
            print(f"INFO: AsyncConnectionPool {pool.name} opened successfully.")
            _global_checkpointer_pool = pool  # Store for shutdown

            # Instantiate the checkpointer and run setup with autocommit connection
            checkpointer = AsyncPostgresSaver(conn=pool)

            # Get a connection and set it to autocommit mode for the setup
            async with pool.connection() as conn:
                await conn.set_autocommit(True)
                # Create a temporary checkpointer with this autocommit connection
                temp_checkpointer = AsyncPostgresSaver(conn=conn)

                if hasattr(temp_checkpointer, "setup") and callable(
                    temp_checkpointer.setup
                ):
                    await temp_checkpointer.setup()
                    print("INFO: LangGraph checkpointer.setup() complete.")
                else:
                    print(
                        "ERROR: AsyncPostgresSaver does not have a recognized setup method."
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
        print("INFO: LangGraph components not available, skipping checkpointer setup.")


# Define middleware (optional, example for CORS)
middleware = [
    Middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
]


# Import the Django-integrated MCP FastAPI app
# (Removed import and mount of MCP app after extraction to standalone service.)

# Define routes with explicit ordering
routes = [
    # Mount Django app as the sole application now that MCP is standalone
    Mount("/", app=django_asgi_app),
]

# Create the main application with proper middleware and lifespan management
application = Starlette(
    debug=settings.DEBUG,
    routes=routes,
    middleware=middleware,
    lifespan=lifespan,
)
