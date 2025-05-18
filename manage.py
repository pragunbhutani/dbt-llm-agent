#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# Add python-dotenv integration
from pathlib import Path


def load_dotenv():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv as dotenv_load

        # Determine the base directory (where manage.py is)
        BASE_DIR = Path(__file__).resolve().parent
        # Specify the path to the .env file
        dotenv_path = BASE_DIR / ".env"
        if dotenv_path.exists():
            print(f"Loading environment variables from: {dotenv_path}")
            dotenv_load(dotenv_path=dotenv_path)
        else:
            print("No .env file found to load.")
    except ImportError:
        print(
            "Warning: python-dotenv not installed. Cannot load .env file."
            " Please install it: pip install python-dotenv or add to pyproject.toml"
        )
    except Exception as e:
        print(f"Error loading .env file: {e}")


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ragstar.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # Load .env file before executing command line
    load_dotenv()

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
