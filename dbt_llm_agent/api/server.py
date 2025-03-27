"""FastAPI server for the dbt-llm-agent."""

import os
import tempfile
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path

import uvicorn
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    File,
    UploadFile,
    BackgroundTasks,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Import agent
from dbt_llm_agent.core.agent import DBTAgent
from dbt_llm_agent.core.dbt_parser import DBTProjectParser
from dbt_llm_agent.storage.model_storage import ModelStorage
from dbt_llm_agent.storage.model_embedding_storage import ModelEmbeddingStorage
from dbt_llm_agent.storage.question_storage import QuestionStorage
from dbt_llm_agent.utils.model_selector import ModelSelector
from dbt_llm_agent.utils.config import load_config
from dbt_llm_agent.utils.cli_utils import load_dotenv_once
from dbt_llm_agent.core.models import DBTModel, Question, ModelEmbedding
from dbt_llm_agent.llm.client import LLMClient, is_openai_api_key_valid
from dbt_llm_agent.core.agent import Agent
from dbt_llm_agent.parsers.dbt_manifest_parser import DBTManifestParser
from dbt_llm_agent.parser.dbt_query_parser import parse_sql
from dbt_llm_agent.utils.logging import get_logger

# Backward compatibility imports
from dbt_llm_agent.storage import (
    PostgresStorage,
    PostgresVectorStore,
    QuestionTrackingService,
)

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv_once()

# Initialize FastAPI app
app = FastAPI(
    title="dbt-llm-agent API",
    description="API for interacting with the dbt-llm-agent",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API requests and responses
class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    id: int
    question: str
    answer: str
    relevant_models: List[Dict[str, Any]]


class DocumentationRequest(BaseModel):
    model_name: str


class DocumentationResponse(BaseModel):
    model_name: str
    model_description: Optional[str] = None
    column_descriptions: Dict[str, str] = Field(default_factory=dict)
    full_documentation: str
    error: Optional[str] = None


class InterpretModelRequest(BaseModel):
    model_name: str


class InterpretModelResponse(BaseModel):
    model_name: str
    yaml_documentation: str
    success: bool
    error: Optional[str] = None


class ParseProjectRequest(BaseModel):
    project_path: str
    force: bool = False


class ParseProjectResponse(BaseModel):
    project_path: str
    models_count: int
    models: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class ConfigResponse(BaseModel):
    success: bool
    message: str
    config: Dict[str, Any] = Field(default_factory=dict)


class ModelListResponse(BaseModel):
    models: List[str] = Field(default_factory=list)


class ModelDetailResponse(BaseModel):
    name: str
    description: str
    columns: Dict[str, Any] = Field(default_factory=dict)
    schema: str
    database: str
    materialization: str
    tags: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    tests: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    was_useful: bool
    feedback: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str


class SearchModelsResponse(BaseModel):
    models: List[Dict[str, Any]] = Field(default_factory=list)


# Models for new embedding endpoint
class EmbedModelsRequest(BaseModel):
    select: str
    force: Optional[bool] = False


class EmbedModelsResponse(BaseModel):
    selected_models: List[str]
    message: str


# Cache for the agent
agent_cache = {}


def get_agent():
    """Get the agent instance, initializing it if necessary."""
    global agent_cache

    # Check if agent is already initialized
    if "agent" in agent_cache:
        return agent_cache["agent"]

    # Load configuration
    config = load_config()

    # Check for required configuration
    if not config.get("openai_api_key"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    # Initialize PostgreSQL storage
    postgres = PostgresStorage(connection_string=config["postgres_uri"], echo=False)

    # Initialize vector store
    vector_store = PostgresVectorStore(
        persist_directory=config["vector_db_path"], collection_name="dbt_models"
    )

    # Initialize agent
    agent = DBTAgent(
        postgres_storage=postgres,
        vector_store=vector_store,
        openai_api_key=config["openai_api_key"],
        model_name=config["openai_model"],
        temperature=float(config["temperature"]),
    )

    # Cache the agent
    agent_cache["agent"] = agent

    return agent


def reset_agent_cache():
    """Reset the agent cache."""
    global agent_cache
    agent_cache = {}


# API routes
@app.get("/")
async def root():
    """Root endpoint that returns basic API information."""
    return {"message": "dbt-llm-agent API is running"}


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    agent: DBTAgent = Depends(get_agent),
    question_tracking: QuestionStorage = Depends(get_question_tracking),
):
    """Ask a question to the agent."""
    try:
        # Ask the question
        result = agent.answer_question(request.question)

        # Record the question and its answer
        model_names = [model["name"] for model in result["relevant_models"]]
        question_id = question_tracking.record_question(
            question_text=request.question,
            answer_text=result["answer"],
            model_names=model_names,
        )

        # Format response
        return {
            "id": question_id,
            "question": request.question,
            "answer": result["answer"],
            "relevant_models": result["relevant_models"],
        }
    except Exception as e:
        logger.error(f"Error asking question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documentation", response_model=DocumentationResponse)
async def generate_documentation(
    request: DocumentationRequest, agent: DBTAgent = Depends(get_agent)
):
    """Generate documentation for a model."""
    try:
        # Call the agent to generate documentation
        result = agent.generate_documentation(request.model_name)

        # Write the documentation to a temporary file for later use
        with open(f"temp_docs_{request.model_name}.yml", "w") as f:
            f.write(result.get("full_documentation", ""))

        return result
    except Exception as e:
        logger.error(f"Error generating documentation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating documentation: {str(e)}"
        )


@app.post("/documentation/{model_name}/save")
async def save_documentation(model_name: str, agent: DBTAgent = Depends(get_agent)):
    """Save generated documentation for a model."""
    try:
        # First generate the documentation
        result = agent.generate_documentation(model_name)

        if "error" in result:
            return JSONResponse(
                status_code=400, content={"success": False, "message": result["error"]}
            )

        # Then update the model documentation
        success = agent.update_model_documentation(model_name, result)

        if success:
            return {
                "success": True,
                "message": f"Documentation for {model_name} saved successfully",
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Failed to save documentation for {model_name}",
                },
            )
    except Exception as e:
        logger.error(f"Error saving documentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documentation/{model_name}/save-yaml")
async def save_yaml_documentation(
    model_name: str, agent: DBTAgent = Depends(get_agent)
):
    """Save YAML documentation for a model from a temporary file."""
    try:
        # Get the current documentation in memory
        temp_path = f"temp_docs_{model_name}.yml"
        if not os.path.exists(temp_path):
            raise HTTPException(
                status_code=404,
                detail=f"No temporary documentation found for model {model_name}",
            )

        with open(temp_path, "r") as f:
            documentation_content = f.read()

        # Parse the documentation
        import yaml

        try:
            yaml_content = yaml.safe_load(documentation_content)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid YAML format: {str(e)}"
            )

        # Find the model definition
        model_def = None
        for model in yaml_content.get("models", []):
            if model.get("name") == model_name:
                model_def = model
                break

        if not model_def:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} not found in the YAML documentation",
            )

        # Extract model documentation
        model_description = model_def.get("description", "")
        column_descriptions = {}
        for column in model_def.get("columns", []):
            column_name = column.get("name")
            column_description = column.get("description", "")
            column_descriptions[column_name] = column_description

        # Update model in database
        success = agent.update_model_documentation(
            model_name,
            {
                "model_description": model_description,
                "column_descriptions": column_descriptions,
            },
        )

        # Remove temporary file
        os.remove(temp_path)

        if success:
            return {"success": True, "message": "Documentation saved successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save documentation")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving documentation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error saving documentation: {str(e)}"
        )


@app.post("/interpret", response_model=InterpretModelResponse)
async def interpret_model(
    request: InterpretModelRequest, agent: DBTAgent = Depends(get_agent)
):
    """Interpret a model and generate documentation for it."""
    try:
        # Validate model existence
        model = agent.postgres.get_model(request.model_name)
        if not model:
            raise HTTPException(
                status_code=404, detail=f"Model {request.model_name} not found"
            )

        # Call the agent to interpret the model
        result = agent.interpret_model(request.model_name)

        # Handle errors
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error interpreting model: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error interpreting model: {str(e)}"
        )


@app.post("/interpret/{model_name}/save")
async def save_interpreted_documentation(
    model_name: str, request: Dict[str, str], agent: DBTAgent = Depends(get_agent)
):
    """Save interpreted documentation for a model."""
    try:
        if "yaml_documentation" not in request:
            raise HTTPException(
                status_code=400, detail="Missing 'yaml_documentation' field in request"
            )

        yaml_documentation = request["yaml_documentation"]

        # Call the agent to save the interpreted documentation
        result = agent.save_interpreted_documentation(model_name, yaml_documentation)

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to save interpreted documentation"),
            )

        return {
            "success": True,
            "message": "Interpreted documentation saved successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving interpreted documentation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error saving interpreted documentation: {str(e)}"
        )


@app.post("/parse", response_model=ParseProjectResponse)
async def parse_project(
    request: ParseProjectRequest,
    background_tasks: BackgroundTasks,
    agent: DBTAgent = Depends(get_agent),
):
    """Parse a dbt project."""
    try:
        # Validate project path
        if not os.path.exists(request.project_path):
            raise HTTPException(
                status_code=400,
                detail=f"Project path {request.project_path} does not exist",
            )

        # Parse the project
        parser = DBTProjectParser(request.project_path)
        project = parser.parse_project()

        # Store models in PostgreSQL and vector store
        for model_name, model in project.models.items():
            agent.postgres.store_model(model)

        model_texts = {
            model_name: model.get_readable_representation()
            for model_name, model in project.models.items()
        }
        agent.vector_store.store_models(model_texts)

        # Get model names
        model_names = list(project.models.keys())

        return {
            "project_path": request.project_path,
            "models_count": len(model_names),
            "models": model_names,
        }
    except Exception as e:
        logger.error(f"Error parsing project: {e}")
        return {
            "project_path": request.project_path,
            "models_count": 0,
            "models": [],
            "error": str(e),
        }


@app.post("/parse/upload")
async def parse_project_upload(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    """Parse a dbt project from an uploaded file (zip archive)."""
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # If it's a zip file, extract it
            if file.filename.endswith(".zip"):
                import zipfile

                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

            # Get agent
            agent = get_agent()

            # Parse the project
            parser = DBTProjectParser(temp_dir)
            project = parser.parse_project()

            # Store models in PostgreSQL and vector store
            for model_name, model in project.models.items():
                agent.postgres.store_model(model)

            model_texts = {
                model_name: model.get_readable_representation()
                for model_name, model in project.models.items()
            }
            agent.vector_store.store_models(model_texts)

            # Get model names
            model_names = list(project.models.keys())

            return {
                "project_path": temp_dir,
                "models_count": len(model_names),
                "models": model_names,
            }
    except Exception as e:
        logger.error(f"Error parsing uploaded project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get the current configuration."""
    try:
        config = load_config()

        # Mask sensitive values
        safe_config = config.copy()
        if "openai_api_key" in safe_config:
            safe_config["openai_api_key"] = (
                "********" if safe_config["openai_api_key"] else ""
            )
        if "slack_bot_token" in safe_config:
            safe_config["slack_bot_token"] = (
                "********" if safe_config["slack_bot_token"] else ""
            )
        if "slack_app_token" in safe_config:
            safe_config["slack_app_token"] = (
                "********" if safe_config["slack_app_token"] else ""
            )
        if "slack_signing_secret" in safe_config:
            safe_config["slack_signing_secret"] = (
                "********" if safe_config["slack_signing_secret"] else ""
            )

        return {
            "success": True,
            "message": "Configuration retrieved successfully from environment variables",
            "config": safe_config,
        }
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        return {"success": False, "message": str(e), "config": {}}


@app.get("/models", response_model=ModelListResponse)
async def list_models(agent: DBTAgent = Depends(get_agent)):
    """List all models."""
    try:
        model_names = agent.postgres.get_all_models()
        return {"models": model_names}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}", response_model=ModelDetailResponse)
async def get_model_details(model_name: str, agent: DBTAgent = Depends(get_agent)):
    """Get details for a specific model."""
    try:
        model = agent.postgres.get_model(model_name)

        if not model:
            return JSONResponse(
                status_code=404, content={"error": f"Model {model_name} not found"}
            )

        # Convert model to dict
        model_dict = model.to_dict()

        # Extract the necessary fields for the response
        return {
            "name": model_dict["name"],
            "description": model_dict["description"],
            "columns": model_dict["columns"],
            "schema": model_dict["schema"],
            "database": model_dict["database"],
            "materialization": model_dict["materialization"],
            "tags": model_dict["tags"],
            "depends_on": model_dict["depends_on"],
            "tests": model_dict["tests"],
        }
    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        return {"name": model_name, "error": str(e)}


@app.post("/questions/{question_id}/feedback")
def provide_feedback(
    question_id: int,
    request: FeedbackRequest,
    question_tracking: QuestionStorage = Depends(get_question_tracking),
):
    try:
        # Get the question to make sure it exists
        question = question_tracking.get_question(question_id)
        if not question:
            raise HTTPException(
                status_code=404, detail=f"Question {question_id} not found"
            )

        # Update the feedback
        success = question_tracking.update_feedback(
            question_id=question_id,
            was_useful=request.was_useful,
            feedback=request.feedback,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update feedback")

        return {"message": f"Feedback recorded for question {question_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error providing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/questions")
def list_questions(
    limit: int = 10,
    offset: int = 0,
    useful: Optional[bool] = None,
    question_tracking: QuestionStorage = Depends(get_question_tracking),
):
    try:
        questions = question_tracking.get_all_questions(
            limit=limit, offset=offset, was_useful=useful
        )

        return {"questions": questions, "count": len(questions)}
    except Exception as e:
        logger.error(f"Error listing questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", response_model=EmbedModelsResponse)
def embed_models(
    request: EmbedModelsRequest,
    postgres_storage: ModelStorage = Depends(get_postgres_storage),
    vector_store: ModelEmbeddingStorage = Depends(get_vector_store),
):
    try:
        # Get all models from the database
        all_models = postgres_storage.get_all_models()
        logger.info(f"Found {len(all_models)} models in the database")

        # Create model selector
        models_dict = {model.name: model for model in all_models}
        selector = ModelSelector(models_dict)
        selected_model_names = selector.select(request.select)

        if not selected_model_names:
            return {
                "selected_models": [],
                "message": f"No models matched the selector: {request.select}",
            }

        # Filter to only selected models
        selected_models = [
            model for model in all_models if model.name in selected_model_names
        ]

        # Embed each model
        models_dict = {}
        metadata_dict = {}

        for model in selected_models:
            model_text = model.get_readable_representation()
            models_dict[model.name] = model_text

            # Create metadata
            metadata = {
                "schema": model.schema,
                "materialization": model.materialization,
            }
            if hasattr(model, "tags") and model.tags:
                metadata["tags"] = model.tags

            metadata_dict[model.name] = metadata

        # Store models in vector database
        vector_store.store_models(models_dict, metadata_dict)

        return {
            "selected_models": selected_model_names,
            "message": f"Successfully embedded {len(selected_model_names)} models",
        }
    except Exception as e:
        logger.error(f"Error embedding models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_postgres_storage():
    """Get a PostgreSQL storage instance."""
    global postgres
    if postgres is None:
        postgres = ModelStorage(connection_string=config["postgres_uri"], echo=False)
    return postgres


def get_vector_store():
    """Get a vector store instance."""
    global vector_store
    if vector_store is None:
        vector_store = ModelEmbeddingStorage(
            connection_string=config["postgres_uri"],
            collection_name="dbt_models",
        )
    return vector_store


def get_question_tracking():
    """Get a question tracking service instance."""
    global question_tracking
    if question_tracking is None:
        question_tracking = QuestionStorage(config["postgres_uri"])
    return question_tracking


async def get_related_models(
    query: str,
    postgres_storage: ModelStorage = Depends(get_postgres_storage),
    vector_store: ModelEmbeddingStorage = Depends(get_vector_store),
):
    """Get models related to a query using vector search."""
    try:
        # Use vector store to find related models
        search_results = vector_store.search_models(query, n_results=5)

        # Get full model details for each result
        related_models = []
        for result in search_results:
            model_name = result["model_name"]
            model = postgres_storage.get_model(model_name)
            if model:
                related_models.append(
                    {
                        "name": model_name,
                        "description": model.description,
                        "similarity": result["similarity_score"],
                    }
                )

        return related_models
    except Exception as e:
        logger.error(f"Error getting related models: {e}")
        return []


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run("dbt_llm_agent.api.server:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    start_server()
