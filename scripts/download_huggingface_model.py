#!/usr/bin/env python3
"""
download_huggingface_model.py

Download the trained DJMGNN model from HuggingFace repository.

Usage:
    python scripts/download_huggingface_model.py
"""

import os
import sys
import logging
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        logger.info(f"Downloading {url.split('/')[-1]}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create parent directory if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"✅ Downloaded to {destination}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        return False


def main():
    # HuggingFace repository
    repo_id = "saketh11/MoML-CA"
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"
    
    # Target directory
    target_dir = Path("/tmp/djmgnn_model/finetuned_model")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to download
    files_to_download = [
        ("finetuned_model/pytorch_model.pt", target_dir / "pytorch_model.pt"),
        ("finetuned_model/config.json", target_dir / "config.json"),
    ]
    
    logger.info("=" * 70)
    logger.info(f"📥 Downloading DJMGNN model from HuggingFace: {repo_id}")
    logger.info("=" * 70)
    
    success_count = 0
    
    for file_path, destination in files_to_download:
        url = f"{base_url}/{file_path}"
        if download_file(url, destination):
            success_count += 1
    
    logger.info("\n" + "=" * 70)
    if success_count == len(files_to_download):
        logger.info("✅ All files downloaded successfully!")
        logger.info(f"📁 Model location: {target_dir}")
        logger.info("\nYou can now run:")
        logger.info("  python scripts/test_djmgnn_force_field_pipeline.py")
        logger.info("  python scripts/test_huggingface_djmgnn.py")
    else:
        logger.error(f"❌ Only {success_count}/{len(files_to_download)} files downloaded")
        logger.error("Please check your internet connection and try again")
        sys.exit(1)
    
    logger.info("=" * 70)


if __name__ == '__main__':
    main()