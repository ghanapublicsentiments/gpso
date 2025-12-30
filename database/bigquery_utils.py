"""BigQuery utility functions for GPSO.

This module provides simple connection and helper utilities for BigQuery operations.
Supports both file-based credentials (for local/pipeline) and in-memory credentials (for Streamlit).
"""

import json
from pathlib import Path
from typing import Optional

from google.cloud import bigquery
from google.oauth2 import service_account

from config import BIGQUERY_CREDENTIALS_PATH, BIGQUERY_DATASET


def get_project_id(creds_dict: Optional[dict] = None) -> str:
    """Get project ID from credentials file or dict.

    Args:
        creds_dict: Optional credentials dictionary (for in-memory credentials).

    Returns:
        str: Project ID string.

    Raises:
        ValueError: If credentials not found or project_id not in credentials.
    """
    # If credentials dict provided, use it
    if creds_dict:
        project_id = creds_dict.get("project_id")
        if not project_id:
            raise ValueError("project_id not found in credentials dictionary")
        return project_id
    
    # Otherwise, read from file
    credentials_path = Path(BIGQUERY_CREDENTIALS_PATH)
    if not credentials_path.exists():
        raise ValueError(f"Credentials file not found: {credentials_path}")

    with open(credentials_path) as f:
        creds_data = json.load(f)
        project_id = creds_data.get("project_id")
        if not project_id:
            raise ValueError("project_id not found in credentials file")
        return project_id


def get_bigquery_client(creds_dict: Optional[dict] = None) -> bigquery.Client:
    """Get BigQuery client instance with credentials.

    Supports both file-based and in-memory credentials for flexibility across
    deployment scenarios (local development, pipeline, Streamlit Cloud).

    Args:
        creds_dict: Optional credentials dictionary from Streamlit secrets.
                   If None, will attempt to read from file.

    Returns:
        bigquery.Client: Configured BigQuery client.

    Raises:
        ValueError: If credentials not found or invalid.
    """
    scopes = ["https://www.googleapis.com/auth/bigquery"]
    
    # Use in-memory credentials if provided
    if creds_dict:
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=scopes
        )
        project_id = get_project_id(creds_dict)
    else:
        # Use file-based credentials
        credentials_path = Path(BIGQUERY_CREDENTIALS_PATH)

        if not credentials_path.exists():
            raise ValueError(
                f"""
                BigQuery credentials file not found: {credentials_path}
                Please download your service account JSON key and save it as:
                  bigquery-credentials.json
                in the root directory of this project.
                """.strip()
            )

        # Load credentials from service account JSON
        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path),
            scopes=scopes
        )
        project_id = get_project_id()

    return bigquery.Client(project=project_id, credentials=credentials)

def get_dataset_id() -> str:
    """Get fully qualified dataset ID.

    Returns:
        str: Fully qualified dataset ID in format 'project_id.dataset_name'.
    """
    project_id = get_project_id()
    return f"{project_id}.{BIGQUERY_DATASET}"