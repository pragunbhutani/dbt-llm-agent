import os

# Default data warehouse type
DEFAULT_DATA_WAREHOUSE_TYPE = "snowflake"

# Environment variable names
DATA_WAREHOUSE_TYPE_ENV_VAR = "SETTINGS_DATA_WAREHOUSE_TYPE"
SNOWFLAKE_ACCOUNT_IDENTIFIER_ENV_VAR = "SETTINGS_SNOWFLAKE_ACCOUNT_IDENTIFIER"
SNOWFLAKE_USERNAME_ENV_VAR = "SETTINGS_SNOWFLAKE_USERNAME"
SNOWFLAKE_PASSWORD_ENV_VAR = "SETTINGS_SNOWFLAKE_PASSWORD"
SNOWFLAKE_WAREHOUSE_ENV_VAR = "SETTINGS_SNOWFLAKE_WAREHOUSE"
SNOWFLAKE_DATABASE_ENV_VAR = "SETTINGS_SNOWFLAKE_DATABASE"
SNOWFLAKE_SCHEMA_ENV_VAR = "SETTINGS_SNOWFLAKE_SCHEMA"

# Load settings from environment variables
DATA_WAREHOUSE_TYPE = os.getenv(
    DATA_WAREHOUSE_TYPE_ENV_VAR, DEFAULT_DATA_WAREHOUSE_TYPE
)
SNOWFLAKE_ACCOUNT_IDENTIFIER = os.getenv(SNOWFLAKE_ACCOUNT_IDENTIFIER_ENV_VAR)
SNOWFLAKE_USERNAME = os.getenv(SNOWFLAKE_USERNAME_ENV_VAR)
SNOWFLAKE_PASSWORD = os.getenv(SNOWFLAKE_PASSWORD_ENV_VAR)
# Load new optional settings
SNOWFLAKE_WAREHOUSE = os.getenv(SNOWFLAKE_WAREHOUSE_ENV_VAR)
SNOWFLAKE_DATABASE = os.getenv(SNOWFLAKE_DATABASE_ENV_VAR)
SNOWFLAKE_SCHEMA = os.getenv(SNOWFLAKE_SCHEMA_ENV_VAR)


def get_snowflake_credentials():
    """
    Returns a dictionary containing Snowflake credentials.
    Raises ValueError if essential credentials are not set.
    Includes optional warehouse, database, and schema if set.
    """
    if not SNOWFLAKE_ACCOUNT_IDENTIFIER:
        raise ValueError(f"{SNOWFLAKE_ACCOUNT_IDENTIFIER_ENV_VAR} is not set.")
    if not SNOWFLAKE_USERNAME:
        raise ValueError(f"{SNOWFLAKE_USERNAME_ENV_VAR} is not set.")
    if not SNOWFLAKE_PASSWORD:
        raise ValueError(f"{SNOWFLAKE_PASSWORD_ENV_VAR} is not set.")

    creds = {
        "account": SNOWFLAKE_ACCOUNT_IDENTIFIER,
        "user": SNOWFLAKE_USERNAME,
        "password": SNOWFLAKE_PASSWORD,
    }

    if SNOWFLAKE_WAREHOUSE:
        creds["warehouse"] = SNOWFLAKE_WAREHOUSE
    if SNOWFLAKE_DATABASE:
        creds["database"] = SNOWFLAKE_DATABASE
    if SNOWFLAKE_SCHEMA:
        creds["schema"] = SNOWFLAKE_SCHEMA

    return creds


def get_data_warehouse_type():
    """Returns the configured data warehouse type."""
    return DATA_WAREHOUSE_TYPE
