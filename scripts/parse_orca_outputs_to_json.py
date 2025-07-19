"""
scripts/parse_orca_outputs_to_json.py

This script parses quantum mechanical (QM) properties from a series of ORCA
output files (`.out`). It is designed to extract key information such as energy,
atomic coordinates, CHELPG partial charges, and dipole moments for a predefined
list of molecules.

The script iterates through specified molecule identifiers, reads the
corresponding ORCA output files, and uses regular expressions to find and
extract the relevant data. The collected data is then aggregated and saved into
a single structured JSON file. This serves as a data preprocessing step to
convert raw QM simulation outputs into a machine-readable format for downstream
tasks like model training or analysis.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Robust regex for floating-point numbers
FLOAT_REGEX = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"

# List of successfully computed PFAS molecule identifiers
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

ORCA_RESULTS_DIR = Path("genx_orca_results")
OUTPUT_JSON_FILE = ORCA_RESULTS_DIR / "qm_labels_subset.json"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _parse_energy(content: str) -> Optional[float]:
    """Parse the final single point energy from ORCA output."""
    # Look for "FINAL SINGLE POINT ENERGY"
    match = re.search(r"FINAL SINGLE POINT ENERGY\s+(" + FLOAT_REGEX + r")", content)
    if match:
        return float(match.group(1))
    # Fallback for optimization jobs reporting "Total Energy"
    match = re.search(r"Total Energy\s+:\s+(" + FLOAT_REGEX + r")\s+Eh", content)
    if match:
        return float(match.group(1))
    return None


def _parse_coordinates(content: str) -> Optional[List[Tuple[str, float, float, float]]]:
    """Parse the final optimized atomic coordinates from the last block."""
    coord_sections = list(
        re.finditer(r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n-+\s*\n(.*?)\n\n", content, re.DOTALL)
    )
    if not coord_sections:
        return None
    
    last_section = coord_sections[-1].group(1)
    atom_pattern = re.compile(rf"^\s*([A-Za-z]+)\s+({FLOAT_REGEX})\s+({FLOAT_REGEX})\s+({FLOAT_REGEX})")
    coords = []
    for line in last_section.splitlines():
        match = atom_pattern.match(line.strip())
        if match:
            coords.append(
                (
                    match.group(1),
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4)),
                )
            )
    return coords or None


def _parse_chelpg_charges_from_out(content: str) -> Optional[List[float]]:
    """Parse CHELPG charges from the main .out file, trying multiple patterns."""
    charge_line_pattern = re.compile(r"^\s*\d+\s+[A-Za-z]+\s*:?\s*(" + FLOAT_REGEX + ")")
    
    # Pattern 1: More structured block
    match = re.search(r"CHELPG Charges\s*\n\s*-+\s*\n((?:\s*\d+\s+[A-Za-z]+\s*:\s*" + FLOAT_REGEX + r"\s*\n)+)", content)
    if not match:  # Pattern 2: Less structured block
        match = re.search(
            r"CHELPG Charges\s*\n(?:.*\n)*?\n\s*Atom\s*Charge\s*Atomic charge\s*\n(?:-+\s*\n)?((?:.*\n)+?)(?:\n\s*\n|Sum of CHELPG charges)",
            content,
            re.MULTILINE,
        )

    if match:
        charges_block = match.group(1)
        charges = charge_line_pattern.findall(charges_block)
        if charges:
            return [float(c) for c in charges]
    return None


def _parse_chelpg_charges_from_prop(prop_file_path: Path) -> Optional[List[float]]:
    """Parse CHELPG charges from a companion .property.txt file."""
    if not prop_file_path.exists():
        return None
    
    logger.info(f"CHELPG charges not in .out, trying {prop_file_path}")
    prop_content = prop_file_path.read_text()
    match = re.search(r"\$chelpg_charges\s*\n\s*\d+\s*\n((?:\s*" + FLOAT_REGEX + r"\s*\n)+)\$end", prop_content, re.MULTILINE)
    if match:
        charges_block = match.group(1).strip()
        charges = [float(c) for c in charges_block.split()]
        if charges:
            return charges
        logger.warning(f"CHELPG charges found in {prop_file_path} but failed to parse.")
    else:
        logger.warning(f"CHELPG charges section not found in {prop_file_path}.")
    return None


def _parse_dipole_moment(content: str) -> Optional[Dict[str, float]]:
    """Parse the final dipole moment, trying multiple patterns."""
    # Pattern 1: Detailed block
    pattern1 = re.compile(
        r"DIPOLE MOMENT\s*\n(?:.*\n)*?"
        r"Total Dipole Moment\s*:\s*(" + FLOAT_REGEX + r")\s*Debye\s*\n(?:.*\n)*?"
        r"Components\s*\(Debye\)\s*:\s*\n"
        r"\s*X\s*:\s*(" + FLOAT_REGEX + r")\s*\n"
        r"\s*Y\s*:\s*(" + FLOAT_REGEX + r")\s*\n"
        r"\s*Z\s*:\s*(" + FLOAT_REGEX + r")",
        re.DOTALL,
    )
    matches = list(pattern1.finditer(content))
    if matches:
        m = matches[-1]
        return {"Total": float(m.group(1)), "X": float(m.group(2)), "Y": float(m.group(3)), "Z": float(m.group(4))}
    
    # Pattern 2: Simpler table format
    pattern2 = re.compile(r"Total Dipole Moment.*?Tot\s+DX\s+DY\s+DZ\s+TX\s+TY\s+TZ\s*\n\s*(\w+)\s+(" + FLOAT_REGEX + r")\s+(" + FLOAT_REGEX + r")\s+(" + FLOAT_REGEX + r")\s+(" + FLOAT_REGEX + r")")
    matches = list(pattern2.finditer(content))
    if matches:
        m = matches[-1]
        return {"X": float(m.group(3)), "Y": float(m.group(4)), "Z": float(m.group(5)), "Total": float(m.group(2))}

    return None


def parse_single_orca_output(orca_output_path: Path) -> Optional[Dict[str, Any]]:
    """Parses an ORCA output file to extract key quantum mechanical properties."""
    if not orca_output_path.exists():
        logger.warning(f"ORCA output file not found: {orca_output_path}")
        return None

    try:
        content = orca_output_path.read_text()
        
        chelpg_charges = _parse_chelpg_charges_from_out(content)
        if not chelpg_charges:
            prop_file = orca_output_path.with_suffix(".property.txt")
            chelpg_charges = _parse_chelpg_charges_from_prop(prop_file)

        properties = {
            "energy_hartree": _parse_energy(content),
            "coordinates_angstrom": _parse_coordinates(content),
            "chelpg_charges": chelpg_charges,
            "dipole_moment_debye": _parse_dipole_moment(content),
        }
        
        if not all(properties.values()):
            missing = [k for k, v in properties.items() if v is None]
            logger.warning(f"Missing properties for {orca_output_path.name}: {missing}")

        return properties

    except Exception as e:
        logger.error(f"Critical error parsing {orca_output_path}: {e}", exc_info=True)
        return None


def main():
    """
    Parses ORCA output files for a predefined set of PFAS molecules and
    consolidates extracted QM properties into a JSON file.
    """
    logger.info(f"Starting ORCA output parsing for {len(SUCCESSFUL_PFAS_IDS)} molecules.")
    logger.info(f"ORCA results directory: {ORCA_RESULTS_DIR.resolve()}")
    logger.info(f"Output JSON file: {OUTPUT_JSON_FILE.resolve()}")

    all_qm_data = {}
    failed_mols = []

    for pfas_id in SUCCESSFUL_PFAS_IDS:
        logger.info(f"Processing {pfas_id}...")
        orca_out_file = ORCA_RESULTS_DIR / f"{pfas_id}.out"

        parsed_data = parse_single_orca_output(orca_out_file)

        if parsed_data and all(parsed_data.values()):
            all_qm_data[pfas_id] = parsed_data
            logger.info(f"Successfully parsed all data for {pfas_id}.")
        else:
            failed_mols.append(pfas_id)
            all_qm_data[pfas_id] = parsed_data or {}
            all_qm_data[pfas_id]["error"] = f"Parsing failed or critical data missing from {orca_out_file}"
            logger.warning(f"Failed to parse all required data for {pfas_id}.")
    
    try:
        OUTPUT_JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSON_FILE, "w") as f:
            json.dump(all_qm_data, f, indent=2)
        logger.info(f"Successfully wrote consolidated QM data to {OUTPUT_JSON_FILE}")
    except IOError as e:
        logger.error(f"Failed to write JSON output to {OUTPUT_JSON_FILE}", exc_info=True)

    if failed_mols:
        logger.warning("\n--- Parsing Challenges Report ---")
        logger.warning(f"Failed to parse or had missing data for {len(failed_mols)} molecules:")
        for pfas_id in failed_mols:
            logger.warning(f"  - {pfas_id}")
        logger.warning("--- End of Report ---")
    else:
        logger.info("All specified molecules were parsed successfully.")


if __name__ == "__main__":
    main()