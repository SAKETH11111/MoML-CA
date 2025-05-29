#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 MoML-CA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---
#
# Test script for ORCA output parsing and conversion to QM9 NPZ format.
import subprocess
import argparse
import logging
import json # Added for writing JSON output
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from moml.simulation.qm.parser.orca_parser import parse_orca_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test ORCA parsing and conversion pipeline")
    parser.add_argument("--output_file", required=True, help="Path to ORCA output file (.out)")
    parser.add_argument("--json_file", required=True, help="Path to intermediate JSON file")
    parser.add_argument("--npz_file", required=True, help="Path to QM9-format output NPZ file")
    args = parser.parse_args()
    output_file = args.output_file
    json_file = args.json_file
    npz_file = args.npz_file

    # 1. Parse ORCA output to JSON
    logger.info(f"Parsing ORCA output: {output_file}")
    parsed_data = parse_orca_output(output_file)

    if parsed_data.get("status") != "completed":
        logger.error(f"Failed to parse ORCA output or calculation was not successful: {output_file}. Status: {parsed_data.get('status')}")
        if parsed_data.get("error_message"):
            logger.error(f"Error details: {parsed_data.get('error_message')}")
        return

    logger.info(f"Writing parsed data to JSON: {json_file}")
    try:
        with open(json_file, "w") as f:
            json.dump(parsed_data, f, indent=2)
        logger.info(f"Successfully wrote parsed data to {json_file}")
    except IOError as e:
        logger.error(f"Failed to write JSON file {json_file}: {e}")
        return

    # 2. Convert JSON to QM9-style NPZ
    logger.info(f"Converting JSON to QM9-style NPZ: {json_file} -> {npz_file}")
    cmd = [
        "python",
        "scripts/orca_json_to_qm9_npz.py",
        json_file,
        "-o",
        npz_file,
    ]
    try:
        result_process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully converted JSON to NPZ: {npz_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting JSON to NPZ: {e}")
        return

    logger.info("Successfully completed parsing and conversion.")


if __name__ == "__main__":
    main()
