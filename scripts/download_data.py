#!/usr/bin/env python
"""
Script to download the Home Credit Default Risk dataset from Kaggle.
"""

import os
import sys
import zipfile
from pathlib import Path

# Add the project root to the path so we can import app
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings  # noqa: E402


def main() -> None:
    """Download and extract the Kaggle dataset."""
    print("Starting dataset download process...")

    # Inject Kaggle credentials into env variables for the kaggle module
    if settings.KAGGLE_USERNAME and settings.KAGGLE_KEY:
        os.environ["KAGGLE_USERNAME"] = settings.KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = settings.KAGGLE_KEY
    else:
        print("Warning: KAGGLE_USERNAME or KAGGLE_KEY not found in settings.")
        print(
            "The script will proceed, but if ~/.kaggle/kaggle.json does "
            "not exist, it will fail."
        )

    try:
        # Import kaggle here after setting environment variables
        import kaggle
    except OSError as e:
        print(f"\nError initializing Kaggle API: {e}")
        print(
            "Please ensure your KAGGLE_USERNAME and KAGGLE_KEY are set "
            "in the .env file."
        )
        sys.exit(1)

    dataset_name = "home-credit-default-risk"
    raw_data_dir = project_root / "data" / "raw"

    # Ensure the directory exists
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name} to {raw_data_dir}...")

    try:
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            dataset_name, path=raw_data_dir, quiet=False
        )
        print("Download complete.")

        # The downloaded file is a zip file
        zip_file_path = raw_data_dir / f"{dataset_name}.zip"

        if zip_file_path.exists():
            print(f"Extracting {zip_file_path}...")
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(raw_data_dir)
            print("Extraction complete.")

            # Optionally remove the zip file to save space
            zip_file_path.unlink()
            print("Removed zip file.")
        else:
            print(
                "Zip file not found. Files might be downloaded directly."
            )

    except Exception as e:
        print(f"An error occurred during download/extraction: {e}")
        sys.exit(1)

    print("Data download process finished successfully.")


if __name__ == "__main__":
    main()
