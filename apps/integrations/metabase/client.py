import os
import requests
from typing import Optional, Dict, Any, List
import json

# Placeholder for your Metabase database ID.
# You need to find this in your Metabase instance (e.g., by looking at the URL when viewing a database).
DEFAULT_INTEGRATIONS_METABASE_DATABASE_ID = 1


class MetabaseClient:
    def __init__(
        self,
        metabase_url: Optional[str] = None,
        api_key: Optional[str] = None,
        database_id: Optional[int] = None,
    ):
        self.metabase_url = metabase_url or os.environ.get("INTEGRATIONS_METABASE_URL")
        self.api_key = api_key or os.environ.get("INTEGRATIONS_METABASE_API_KEY")
        self.database_id = database_id or int(
            os.environ.get(
                "INTEGRATIONS_METABASE_DATABASE_ID",
                DEFAULT_INTEGRATIONS_METABASE_DATABASE_ID,
            )
        )

        if not self.metabase_url:
            raise ValueError(
                "Metabase URL not provided or found in INTEGRATIONS_METABASE_URL environment variable."
            )
        if not self.api_key:
            raise ValueError(
                "Metabase API key not provided or found in INTEGRATIONS_METABASE_API_KEY environment variable."
            )

        self.headers = {
            "Content-Type": "application/json",
            "X-Metabase-Session": self.api_key,  # According to some docs, API key can be used as a session token
            # For newer versions, 'x-api-key': self.api_key might be needed.
            # We should verify this. The initial search showed x-api-key header.
        }
        # Let's adjust to use x-api-key as primary, as per the latest docs.
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

    def _request(
        self, method: str, endpoint: str, json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.metabase_url.rstrip('/')}/api/{endpoint.lstrip('/')}"
        try:
            response = requests.request(
                method, url, headers=self.headers, json=json_data
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Log error details, including response text if available
            error_message = f"Metabase API request failed: {e}"
            if e.response is not None:
                error_message += f" | Response: {e.response.text}"
            # Consider raising a custom exception here for better error handling upstream
            raise Exception(error_message) from e
        except requests.exceptions.RequestException as e:
            # Handle other request exceptions (e.g., connection errors)
            raise Exception(f"Metabase API request failed: {e}") from e

    def list_collections(self, parent_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Lists collections, filtered by parent_id (None for root collections)."""

        response_data = self._request("GET", "collection")

        # The /api/collection endpoint returns all collections.
        # Metabase's own UI fetches all and then filters/organizes client-side in many views.
        # Some API versions might have a flat list, others a dict with a 'data' key.
        all_collections: List[Dict[str, Any]] = []
        if isinstance(response_data, list):
            all_collections = response_data
        elif isinstance(response_data, dict) and isinstance(
            response_data.get("data"), list
        ):
            all_collections = response_data["data"]
        else:
            # Log a warning or handle unexpected structure
            # For now, returning empty list if structure is not recognized.
            if not all_collections:
                print(
                    f"DEBUG MetabaseClient.list_collections: Could not parse collections from response: {response_data}"
                )

        # Filter by location instead of parent_id for more reliable results
        # Root collections have location "/"
        # Child collections have location "/parent_id/"
        if parent_id is None:
            # Root collections have location "/"
            expected_location = "/"
        else:
            # Child collections have location "/parent_id/"
            expected_location = f"/{parent_id}/"

        filtered_collections = [
            c
            for c in all_collections
            if c.get("location") == expected_location and not c.get("archived", False)
        ]

        return filtered_collections

    def get_collection_by_name(
        self, name: str, parent_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Gets a non-archived collection by name, under a specific parent collection (None for root)."""
        # list_collections now correctly filters by location and archived status
        collections_at_level = self.list_collections(parent_id=parent_id)

        for collection in collections_at_level:
            if collection.get("name") == name:
                return collection

        return None

    def create_collection(
        self,
        name: str,
        parent_id: Optional[int] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates a new collection."""
        payload: Dict[str, Any] = {"name": name}
        if parent_id is not None:
            payload["parent_id"] = parent_id
        if description is not None:
            payload["description"] = description

        created_collection_data = self._request("POST", "collection", json_data=payload)

        if not isinstance(created_collection_data, dict):
            raise Exception(
                f"Metabase API returned unexpected data type for create_collection: {type(created_collection_data)}. Expected dict. Response: {created_collection_data}"
            )
        return created_collection_data

    def get_or_create_collection_by_path(self, path: List[str]) -> Optional[int]:
        """
        Ensures a nested collection path exists, creating parts of the path if necessary.
        Returns the ID of the final collection in the path.
        Example: path = ["Ragstar", "User Name"]
        """
        current_parent_id: Optional[int] = None
        collection_id: Optional[int] = None

        for collection_name in path:
            collection = self.get_collection_by_name(
                name=collection_name, parent_id=current_parent_id
            )
            if collection:
                collection_id = collection.get("id")
            else:
                new_collection = self.create_collection(
                    name=collection_name, parent_id=current_parent_id
                )
                collection_id = new_collection.get("id")

            if not collection_id:  # Should not happen if creation was successful
                # Log or raise an error indicating failure to create/find collection part
                raise Exception(
                    f"Failed to get or create collection part: {collection_name} under parent_id {current_parent_id}"
                )

            current_parent_id = collection_id  # The current collection becomes the parent for the next part

        return collection_id

    def create_native_query_question(
        self,
        name: str,
        sql_query: str,
        collection_id: Optional[int],
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates a new question (card) with a native SQL query."""
        payload = {
            "name": name,
            "dataset_query": {
                "type": "native",
                "native": {
                    "query": sql_query,
                    # "template_tags": {} # If we need to parameterize the query later
                },
                "database": self.database_id,
            },
            "display": "table",  # Default display type
            "collection_id": collection_id,
            "visualization_settings": {},
        }
        if description:
            payload["description"] = description

        return self._request("POST", "card", json_data=payload)


if __name__ == "__main__":
    # Example Usage (requires Metabase running and env vars set)
    # Ensure INTEGRATIONS_METABASE_URL, INTEGRATIONS_METABASE_API_KEY, and INTEGRATIONS_METABASE_DATABASE_ID are set in your .env or environment

    print("Attempting to connect to Metabase...")
    try:
        client = MetabaseClient()
        print(
            f"Successfully initialized MetabaseClient for URL: {client.metabase_url} and DB ID: {client.database_id}"
        )

        # 1. Define the collection path
        user_name = "Test User (API)"  # Replace with dynamic user name in Slack handler
        collection_path = ["Ragstar", user_name]
        print(f"Ensuring collection path exists: {collection_path}")

        target_collection_id = client.get_or_create_collection_by_path(collection_path)

        if target_collection_id:
            print(
                f"Successfully found/created collection path. Target Collection ID: {target_collection_id}"
            )

            # 2. Create a question
            question_name = "Sample API Query"
            # Make sure this SQL is valid for your INTEGRATIONS_METABASE_DATABASE_ID
            # For example, if using the Sample Database (often ID 1 or 2):
            sql = "SELECT COUNT(*) AS num_users FROM core_user;"
            description = f"This question was created via API for {user_name}."

            print(
                f"Creating question: '{question_name}' in collection ID: {target_collection_id}"
            )
            new_question = client.create_native_query_question(
                name=question_name,
                sql_query=sql,
                collection_id=target_collection_id,
                description=description,
            )
            print(
                f"Successfully created question: {new_question.get('name')}, ID: {new_question.get('id')}"
            )
            print(
                f"You can view it at: {client.metabase_url.rstrip('/')}/question/{new_question.get('id')}"
            )

        else:
            print("Failed to get or create the target collection.")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An error occurred during Metabase client operations: {e}")
        # For more detailed debugging, you might want to print the full traceback
        # import traceback
        # traceback.print_exc()
