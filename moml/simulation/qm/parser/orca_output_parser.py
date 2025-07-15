"""
ORCA Output Parser

This module provides functionality for parsing ORCA output files to extract quantum mechanical data.
"""

import os
import re
import logging
from typing import Dict, List, Union

import numpy as np

# Define FLOAT constant for robust matching of numeric data
FLOAT = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"

# Set up logging
logger = logging.getLogger("orca_output_parser")

def parse_orca_output(orca_output_path: str) -> Dict[str, Union[List[float], np.ndarray, Dict]]:
    """
    Parse ORCA output file to extract quantum mechanical data in a single pass.

    Args:
        orca_output_path: Path to the ORCA output file

    Returns:
        Dictionary containing extracted data with keys:
            - 'mulliken_charges': List of Mulliken atomic charges
            - 'loewdin_charges': List of Loewdin atomic charges (if available)
            - 'dipole_moment': Dipole moment vector [dx, dy, dz, total] (if available)
            - 'homo_lumo_gap': HOMO-LUMO gap in eV (if available)
            - 'homo_lumo_contributions': Dict with 'homo' and 'lumo' orbital contributions
            - 'electrostatic_potential': List of ESP values at atom positions
            - 'optimized_geometry': List of dictionaries with atom coordinates
            - 'status': Calculation status ('completed', 'error', 'incomplete')
    """
    if not os.path.exists(orca_output_path):
        raise FileNotFoundError(f"ORCA output file not found: {orca_output_path}")

    # Initialize result dictionary - only status is guaranteed.
    result = {
        "status": "incomplete",  # Default status
        "mulliken_charges": [],  # Initialize as empty list
        "loewdin_charges": [],  # Initialize as empty list
        "dipole_moment": None,  # Remains None if not found
        "homo_lumo_gap": None,  # Remains None if not found
        "optimized_geometry": [],  # Initialize as empty list
        "error_message": None,
        "homo_lumo_contributions": {"homo": [], "lumo": []},  # Initialize for consistency
        "electrostatic_potential": [],  # Initialize for consistency
    }

    # Regex patterns for all data types we need to extract
    patterns = {
        "calculation_completed": r".*ORCA TERMINATED NORMALLY.*",
        "error": r".*(ERROR|Error):.*",
        "mulliken_charges": f"MULLIKEN ATOMIC CHARGES.*?\\n(?:\\s*\\d+\\s+\\w+\\s+({FLOAT}).*?\\n)+",
        "loewdin_charges": f"LOEWDIN ATOMIC CHARGES.*?\\n(?:\\s*\\d+\\s+\\w+\\s+({FLOAT}).*?\\n)+",
        "dipole_moment": f"DIPOLE MOMENT(?:.|\\n)*?Total\\s+({FLOAT})\\s+({FLOAT})\\s+({FLOAT})\\s+({FLOAT})",
        "homo_lumo_gap_direct": f"HOMO-LUMO gap:\\s*({FLOAT})\\s*Eh\\s*=\\s*({FLOAT})\\s*eV",
        # More flexible HOMO/LUMO patterns to match lines like "   0  -10.0... 2.0... (HOMO)"
        "homo_energy": f"^\\s*\\d+\\s+({FLOAT})\\s+({FLOAT})+\\s+\\(HOMO\\)",  # Group 1 is energy
        "lumo_energy": f"^\\s*\\d+\\s+({FLOAT})\\s+({FLOAT})+\\s+\\(LUMO\\)",  # Group 1 is energy
        "geometry": r"CARTESIAN COORDINATES \(ANGSTROEM\).*?\n(.*?)\n\n",
    }

    try:
        # Read the file in chunks to reduce memory usage
        with open(orca_output_path, "r") as f:
            content = f.read()

        # Check calculation status
        if re.search(patterns["calculation_completed"], content, re.DOTALL):
            result["status"] = "completed"
        # Check for explicit error messages or abnormal termination
        elif re.search(patterns["error"], content, re.DOTALL) or "ORCA TERMINATED ABNORMALLY" in content:
            result["status"] = "error"
        # If neither completed nor explicitly error/abnormal, it remains 'incomplete'
        # unless a parsing exception occurs later, which will set it to 'error'.

        # Extract Mulliken charges
        mulliken_header_match = re.search(r"MULLIKEN ATOMIC CHARGES", content)
        if mulliken_header_match:
            start_index = mulliken_header_match.end()
            # Define a pattern for a single charge line: number, symbol, optional colon, charge
            # This pattern is reused for both Mulliken and Loewdin charges
            CHARGE_LINE_PATTERN = re.compile(f"^\\s*\\d+\\s+[A-Za-z]{{1,3}}\\s*:?\\s*({FLOAT})")

            temp_mulliken_charges = []

            # Heuristic to find end of block: two newlines, or start of another common section
            end_pattern_search_str = content[start_index:]
            end_match = re.search(r"\n\s*\n|[A-Z\s]{10,}\n-{5,}", end_pattern_search_str)

            block_limit = len(end_pattern_search_str)
            if end_match:
                block_limit = end_match.start()

            relevant_block = end_pattern_search_str[:block_limit]

            for line in relevant_block.splitlines():
                line_strip = line.strip()
                if not line_strip:  # Skip empty lines that might be before the actual end
                    continue

                match = CHARGE_LINE_PATTERN.match(line_strip)
                if match:
                    try:
                        temp_mulliken_charges.append(float(match.group(1)))
                    except ValueError:
                        logger.warning(f"Could not parse float from Mulliken charge line: '{line_strip}'")
                # Stop if a line doesn't match and isn't empty, likely end of charge data or junk
                elif (
                    temp_mulliken_charges and line_strip and not line_strip.startswith("-")
                ):  # Allow for "---" separator lines
                    break

            if temp_mulliken_charges:  # Only update if we found some
                result["mulliken_charges"] = temp_mulliken_charges

        # Extract Loewdin charges
        loewdin_header_match = re.search(r"LOEWDIN ATOMIC CHARGES", content)
        if loewdin_header_match:
            start_index = loewdin_header_match.end()
            # Reuse the defined pattern
            charge_line_pattern = CHARGE_LINE_PATTERN

            temp_loewdin_charges = []

            end_pattern_search_str = content[start_index:]
            end_match = re.search(r"\n\s*\n|[A-Z\s]{10,}\n-{5,}", end_pattern_search_str)

            block_limit = len(end_pattern_search_str)
            if end_match:
                block_limit = end_match.start()

            relevant_block = end_pattern_search_str[:block_limit]

            for line in relevant_block.splitlines():
                line_strip = line.strip()
                if not line_strip:
                    continue

                match = charge_line_pattern.match(line_strip)
                if match:
                    try:
                        temp_loewdin_charges.append(float(match.group(1)))
                    except ValueError:
                        logger.warning(f"Could not parse float from Loewdin charge line: '{line_strip}'")
                elif temp_loewdin_charges and line_strip and not line_strip.startswith("-"):
                    break

            if temp_loewdin_charges:
                result["loewdin_charges"] = temp_loewdin_charges

        # Extract dipole moment
        dipole_match = re.search(patterns["dipole_moment"], content, re.DOTALL)
        if dipole_match:
            dx = float(dipole_match.group(1))
            dy = float(dipole_match.group(2))
            dz = float(dipole_match.group(3))
            total = float(dipole_match.group(4))
            result["dipole_moment"] = [dx, dy, dz, total]

        # Extract HOMO-LUMO gap
        homo_lumo_gap_match = re.search(patterns["homo_lumo_gap_direct"], content)
        if homo_lumo_gap_match:
            result["homo_lumo_gap"] = float(homo_lumo_gap_match.group(2))  # Use the gap in eV
        else:
            # If not found directly, try to calculate from HOMO and LUMO energies
            homo_match = re.search(patterns["homo_energy"], content, re.MULTILINE)
            lumo_match = re.search(patterns["lumo_energy"], content, re.MULTILINE)

            if homo_match and lumo_match:
                homo_energy = float(homo_match.group(1))
                lumo_energy = float(lumo_match.group(1))
                gap_hartree = lumo_energy - homo_energy
                gap_ev = gap_hartree * 27.211  # Convert Hartree to eV
                result["homo_lumo_gap"] = gap_ev

        # Extract optimized geometry
        geometry_match = re.search(patterns["geometry"], content, re.DOTALL)
        if geometry_match:
            result["optimized_geometry"] = []  # Initialize as list
            geometry_text = geometry_match.group(1)
            atom_pattern = f"(\\w+)\\s+({FLOAT})\\s+({FLOAT})\\s+({FLOAT})"

            for line in geometry_text.split("\n"):
                if not line.strip():
                    continue

                atom_match = re.search(atom_pattern, line)
                if atom_match:
                    symbol = atom_match.group(1)
                    x = float(atom_match.group(2))
                    y = float(atom_match.group(3))
                    z = float(atom_match.group(4))

                    result["optimized_geometry"].append({"symbol": symbol, "coordinates": [x, y, z]})

    except Exception as e:
        logger.error(f"Error parsing ORCA output: {str(e)}")
        result["status"] = "error"
        result["error_message"] = str(e)

    return result 