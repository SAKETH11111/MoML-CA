"""
charge_analysis.py

This module provides functionality to analyze partial charges in GNN-generated
force field XML files. It extracts charge information from OpenMM XML force
field files and performs statistical analysis including charge validation,
distribution analysis, and detailed reporting.

Key Features:
- Parse OpenMM XML force field files
- Extract partial charges from NonbondedForce sections
- Validate charge neutrality
- Generate comprehensive charge statistics
- Support for command-line interface
"""

import argparse
import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np

# Constants
CHARGE_TOLERANCE = 1e-6
CHARGE_UNIT = 'e'


def analyze_charges(xml_file: str, molecule_name: str) -> Tuple[float, np.ndarray]:
    """Extract and analyze partial charges from OpenMM XML force field.

    This function parses an OpenMM XML force field file, extracts partial
    charges from the NonbondedForce section, and performs comprehensive
    analysis including charge validation and statistical reporting.

    Args:
        xml_file (str): Path to XML force field file.
        molecule_name (str): Name of molecule for reporting purposes.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing:
            - total_charge (float): Sum of all partial charges
            - charges (np.ndarray): Array of individual partial charges

    Raises:
        ValueError: If charge attribute 'q' is not found for any particle.
        ET.ParseError: If XML file cannot be parsed.
        FileNotFoundError: If XML file does not exist.
    """
    print(f'\nAnalyzing charges in {molecule_name} force field: {xml_file}')

    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract charges from NonbondedForce section
    charges = []
    for force in root.findall('.//Force'):
        if force.get('type') == 'NonbondedForce':
            for particle in force.findall('Particles/Particle'):
                charge_str = particle.get('q')
                if charge_str is not None:
                    charge = float(charge_str)
                    charges.append(charge)
                else:
                    raise ValueError(
                        f'Charge attribute "q" not found for particle in '
                        f'{xml_file}'
                    )

    if not charges:
        raise ValueError(f'No charges found in {xml_file}')

    charges = np.array(charges)

    # Perform statistical analysis
    total_charge = charges.sum()
    abs_charges = np.abs(charges)

    # Display analysis results
    print(f'Number of atoms: {len(charges)}')
    print(f'Total charge: {total_charge:.6f} {CHARGE_UNIT}')
    print(f'Absolute total charge: {abs_charges.sum():.6f} {CHARGE_UNIT}')
    print(f'Charge range: {charges.min():.4f} to {charges.max():.4f} '
          f'{CHARGE_UNIT}')
    print(f'Mean absolute charge: {abs_charges.mean():.4f} {CHARGE_UNIT}')

    # Validate charge neutrality
    if abs(total_charge) < CHARGE_TOLERANCE:
        print(f'✓ PASS: Net charge ≈ 0 (|{total_charge:.6f}| < '
              f'{CHARGE_TOLERANCE})')
    else:
        print(f'✗ FAIL: Net charge = {total_charge:.6f} (not ≈ 0)')

    # Display individual charges
    print('\nIndividual charges:')
    for i, charge in enumerate(charges):
        print(f'  Atom {i+1:2d}: {charge:8.5f} {CHARGE_UNIT}')

    return total_charge, charges


def main() -> int:
    """Main entry point for command-line interface.

    Parses command-line arguments and executes charge analysis on the
    specified XML force field file.
    """
    parser = argparse.ArgumentParser(
        description='Analyze partial charges in OpenMM XML force field files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --xml pfoa.xml --name PFOA
  %(prog)s --xml pfos.xml --name PFOS
        """
    )
    parser.add_argument(
        '--xml',
        required=True,
        help='Path to XML force field file'
    )
    parser.add_argument(
        '--name',
        required=True,
        help='Name of molecule for reporting'
    )

    args = parser.parse_args()

    try:
        analyze_charges(args.xml, args.name)
    except (ValueError, ET.ParseError, FileNotFoundError) as e:
        print(f'Error: {e}')
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
