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
import logging
from moml.simulation.quantum_mechanics.orca_pfas_wrapper import parse_orca_output_to_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    output_file = "PFAS003.out"
    json_file = "PFAS003.json"
    npz_file = "PFAS003_qm9.npz"

    # 1. Parse ORCA output to JSON
    logger.info(f"Parsing ORCA output to JSON: {output_file} -> {json_file}")
    if not parse_orca_output_to_json(output_file, json_file):
        logger.error(f"Failed to convert ORCA output to JSON: {output_file}")
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
