import os
import json
import re
import logging
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define FLOAT constant for robust matching of numeric data
FLOAT = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"

# List of successfully computed PFAS molecule identifiers (excluding DTXSID0047583 and water_test)
SUCCESSFUL_PFAS_IDS = [
    "DTXSID0059794",
    "DTXSID0059958",
    "DTXSID0067848",
    "DTXSID0073168",
    "DTXSID00234929",
    "DTXSID00239570",
    "DTXSID00305289",
    "DTXSID00379829",
    "DTXSID00455632",
]

ORCA_RESULTS_DIR = "genx_orca_results"
OUTPUT_JSON_FILE = os.path.join(ORCA_RESULTS_DIR, "qm_labels_subset.json")

def parse_single_orca_output(orca_output_path: str) -> Optional[Dict[str, Any]]:
    """
    Parses an ORCA output file to extract quantum mechanical properties.
    
    Extracts the final single point energy (in Hartrees), optimized atomic coordinates (in Angstroms), CHELPG atomic charges, and dipole moment components (X, Y, Z, and Total in Debye) from the specified ORCA output file. Handles multiple output formats and attempts to retrieve CHELPG charges from a corresponding `.property.txt` file if not present in the main output. Returns a dictionary with the extracted properties, or `None` if the file does not exist or a critical error occurs. If some properties cannot be parsed, returns a dictionary with available data and missing fields set to `None`.
    
    Args:
        orca_output_path: Path to the ORCA output file.
    
    Returns:
        A dictionary with keys:
            - "energy_hartree": Final single point energy (float).
            - "coordinates_angstrom": List of [atom_symbol, x, y, z] (float).
            - "chelpg_charges": List of CHELPG atomic charges (float).
            - "dipole_moment_debye": Dict with "X", "Y", "Z", "Total" (float).
        Returns None if the file does not exist or a critical error occurs.
    """
    if not os.path.exists(orca_output_path):
        logger.warning(f"ORCA output file not found: {orca_output_path}")
        return None

    properties: Dict[str, Any] = {
        "energy_hartree": None,
        "coordinates_angstrom": None,
        "chelpg_charges": None,
        "dipole_moment_debye": None,
    }

    try:
        with open(orca_output_path, "r") as f:
            content = f.read()

        # 1. Final Single Point Energy
        # Look for "FINAL SINGLE POINT ENERGY" or "Total Energy" in the optimization summary
        energy_match = re.search(r"FINAL SINGLE POINT ENERGY\s+(" + FLOAT + r")", content)
        if not energy_match: # Fallback for optimizations
            energy_match = re.search(r"Total Energy\s+:\s+(" + FLOAT + r")\s+Eh", content) # More general
        if energy_match:
            properties["energy_hartree"] = float(energy_match.group(1))
        else:
            logger.warning(f"Final single point energy not found in {orca_output_path}")

        # 2. Optimized Atomic Coordinates
        # Look for "CARTESIAN COORDINATES (ANGSTROEM)"
        # This section usually appears multiple times, we want the last one for optimized geometry
        coord_sections = list(re.finditer(r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n(-+\s*\n)?(.*?)\n\n", content, re.DOTALL))
        if coord_sections:
            last_coord_section = coord_sections[-1].group(2)
            coords = []
            atom_pattern = re.compile(rf"^\s*([A-Za-z]+)\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})")
            for line in last_coord_section.splitlines():
                line = line.strip()
                if not line:
                    continue
                match = atom_pattern.match(line)
                if match:
                    coords.append([match.group(1), float(match.group(2)), float(match.group(3)), float(match.group(4))])
            if coords:
                properties["coordinates_angstrom"] = coords
            else:
                logger.warning(f"Optimized atomic coordinates not found or failed to parse in {orca_output_path}")
        else:
            logger.warning(f"Cartesian coordinates section not found in {orca_output_path}")


        # 3. CHELPG Atomic Charges
        # Look for "CHELPG Charges" section
        chelpg_match = re.search(r"CHELPG Charges\s*\n(?:.*\n)*?\n\s*Atom\s*Charge\s*Atomic charge\s*\n(?:-+\s*\n)?((?:.*\n)+?)(?:\n\s*\n|Sum of CHELPG charges)", content, re.MULTILINE)
        if not chelpg_match: # Alternative common formatting
             chelpg_match = re.search(r"CHELPG Charges\s*\n\s*-+\s*\n((?:\s*\d+\s+[A-Za-z]+\s*:\s*"+FLOAT+r"\s*\n)+)", content)

        if chelpg_match:
            charges_block = chelpg_match.group(1)
            charges = []
            # Pattern for lines like: "0 C : 0.123456" or "0 C 0.123456"
            charge_pattern = re.compile(rf"^\s*\d+\s+[A-Za-z]+\s*:?\s*({FLOAT})")
            for line in charges_block.splitlines():
                line = line.strip()
                if not line:
                    continue
                match = charge_pattern.match(line)
                if match:
                    charges.append(float(match.group(1)))
            if charges:
                properties["chelpg_charges"] = charges
            else:
                logger.warning(f"CHELPG charges found but failed to parse lines in {orca_output_path}")
        else:
            # Try parsing from .property.txt if it exists
            property_file_path = orca_output_path.replace(".out", ".property.txt")
            if os.path.exists(property_file_path):
                logger.info(f"CHELPG charges not in .out, trying {property_file_path}")
                with open(property_file_path, "r") as pf:
                    prop_content = pf.read()
                # CHELPG charges in .property.txt are usually just a list of numbers
                # Example:
                # $chelpg_charges
                #  num_atoms
                #  charge1
                #  charge2
                #  ...
                # $end
                chelpg_prop_match = re.search(r"\$chelpg_charges\s*\n\s*\d+\s*\n((?:\s*" + FLOAT + r"\s*\n)+)\$end", prop_content, re.MULTILINE)
                if chelpg_prop_match:
                    charges_block_prop = chelpg_prop_match.group(1)
                    charges_prop = [float(c) for c in charges_block_prop.strip().split()]
                    if charges_prop:
                        properties["chelpg_charges"] = charges_prop
                    else:
                        logger.warning(f"CHELPG charges found in {property_file_path} but failed to parse.")
                else:
                    logger.warning(f"CHELPG charges not found in {property_file_path} either.")
            else:
                logger.warning(f"CHELPG charges section not found in {orca_output_path} and no .property.txt found.")


        # 4. Dipole Moment
        # Look for "DIPOLE MOMENT" section, usually the last occurrence is relevant
        dipole_sections = list(re.finditer(r"DIPOLE MOMENT\s*\n(?:.*\n)*?Total Dipole Moment\s*:\s*(" + FLOAT + r")\s*Debye\s*\n(?:.*\n)*?Components\s*\(Debye\)\s*:\s*\n\s*X\s*:\s*(" + FLOAT + r")\s*\n\s*Y\s*:\s*(" + FLOAT + r")\s*\n\s*Z\s*:\s*(" + FLOAT + r")", content, re.DOTALL))
        if not dipole_sections: # Simpler alternative often found
            dipole_sections = list(re.finditer(r"Total Dipole Moment.*?Tot\s+DX\s+DY\s+DZ\s+TX\s+TY\s+TZ\s*\n\s*(\w+)\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")", content))
            if dipole_sections: # Adapt to this format
                last_dipole_match = dipole_sections[-1]
                properties["dipole_moment_debye"] = {
                    "X": float(last_dipole_match.group(3)),
                    "Y": float(last_dipole_match.group(4)),
                    "Z": float(last_dipole_match.group(5)),
                    "Total": float(last_dipole_match.group(2)),
                }
            else: # Try the original more detailed pattern again, but be less strict about "Total Dipole Moment" line
                dipole_sections = list(re.finditer(r"DIPOLE MOMENT[\s\S]*?Magnitude \(Debye\)\s*:\s*(" + FLOAT + r")[\s\S]*?X\s*:\s*(" + FLOAT + r")[\s\S]*?Y\s*:\s*(" + FLOAT + r")[\s\S]*?Z\s*:\s*(" + FLOAT + r")", content))


        if dipole_sections and not properties["dipole_moment_debye"]: # If not already populated by alternative
            last_dipole_match = dipole_sections[-1]
            # Grouping depends on which regex matched
            if len(last_dipole_match.groups()) == 4: # Original detailed pattern
                 properties["dipole_moment_debye"] = {
                    "X": float(last_dipole_match.group(2)),
                    "Y": float(last_dipole_match.group(3)),
                    "Z": float(last_dipole_match.group(4)),
                    "Total": float(last_dipole_match.group(1)),
                }
            elif len(last_dipole_match.groups()) == 5: # Simpler alternative (already handled but for safety)
                 properties["dipole_moment_debye"] = {
                    "X": float(last_dipole_match.group(3)),
                    "Y": float(last_dipole_match.group(4)),
                    "Z": float(last_dipole_match.group(5)),
                    "Total": float(last_dipole_match.group(2)),
                }
        elif not properties["dipole_moment_debye"]:
            logger.warning(f"Dipole moment not found in {orca_output_path}")


    except Exception as e:
        logger.error(f"Error parsing ORCA output file {orca_output_path}: {e}", exc_info=True)
        return None

    # Check if all required fields were populated
    if all(properties[key] is not None for key in ["energy_hartree", "coordinates_angstrom", "chelpg_charges", "dipole_moment_debye"]):
        return properties
    
    logger.warning(f"One or more key properties were not found for {orca_output_path}. Returning partial data: {properties}")
    return properties


def main():
    """
    Parses ORCA output files for a predefined set of PFAS molecules and consolidates extracted quantum mechanical properties into a JSON file.
    
    Iterates over each molecule ID, attempts to extract energy, coordinates, CHELPG charges, and dipole moments from corresponding ORCA output files, and logs any parsing issues. The consolidated results are saved to a JSON file, with a summary report of any missing or failed-to-parse properties.
    """
    logger.info(f"Starting ORCA output parsing for {len(SUCCESSFUL_PFAS_IDS)} molecules.")
    logger.info(f"ORCA results directory: {ORCA_RESULTS_DIR}")
    logger.info(f"Output JSON file: {OUTPUT_JSON_FILE}")

    all_qm_data = {}

    for pfas_id in SUCCESSFUL_PFAS_IDS:
        logger.info(f"Processing {pfas_id}...")
        orca_out_file = os.path.join(ORCA_RESULTS_DIR, f"{pfas_id}.out")

        parsed_data = parse_single_orca_output(orca_out_file)

        if parsed_data:
            all_qm_data[pfas_id] = parsed_data
            logger.info(f"Successfully parsed data for {pfas_id}.")
        else:
            logger.warning(f"Failed to parse or extract all required data for {pfas_id}. It will be excluded or have null values.")
            # Store null for this molecule if parsing failed critically or to represent missing data
            all_qm_data[pfas_id] = {
                "energy_hartree": None,
                "coordinates_angstrom": None,
                "chelpg_charges": None,
                "dipole_moment_debye": None,
                "error": f"Parsing failed or critical data missing from {orca_out_file}"
            }


    # Save the consolidated data to JSON
    try:
        with open(OUTPUT_JSON_FILE, "w") as f:
            json.dump(all_qm_data, f, indent=2)
        logger.info(f"Successfully wrote consolidated QM data to {OUTPUT_JSON_FILE}")
    except IOError as e:
        logger.error(f"Failed to write JSON output to {OUTPUT_JSON_FILE}: {e}")

    # Report on any challenges
    missing_data_report = []
    for pfas_id, data in all_qm_data.items():
        if data is None: # Should not happen with current logic but good check
            missing_data_report.append(f"{pfas_id}: Completely failed to parse.")
            continue
        if "error" in data and data["error"]:
             missing_data_report.append(f"{pfas_id}: {data['error']}")
        else:
            for key, value in data.items():
                if value is None:
                    missing_data_report.append(f"{pfas_id}: Property '{key}' not found or failed to parse.")

    if missing_data_report:
        logger.warning("--- Parsing Challenges Report ---")
        for report_line in missing_data_report:
            logger.warning(report_line)
        logger.warning("--- End of Report ---")
    else:
        logger.info("All specified properties found and parsed for all successful molecules.")


if __name__ == "__main__":
    main()