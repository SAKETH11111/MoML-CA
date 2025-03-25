"""
ORCA Output Parser for Quantum Mechanical Data

This module provides functionality to parse ORCA output files and extract
relevant quantum mechanical data for use in molecular graph representations.
"""

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def parse_orca_output(orca_output_path: str) -> Dict[str, Union[List[float], np.ndarray, Dict]]:
    """
    Parse ORCA output file to extract quantum mechanical data.
    
    Args:
        orca_output_path: Path to the ORCA output file
        
    Returns:
        Dictionary containing extracted data with keys:
            - 'mulliken_charges': List of Mulliken atomic charges
            - 'loewdin_charges': List of Loewdin atomic charges (if available)
            - 'dipole_moment': Dipole moment vector [dx, dy, dz, total] (if available)
            - 'homo_lumo_gap': HOMO-LUMO gap in eV (if available)
            - 'homo_lumo_contributions': Dict with 'homo' and 'lumo' orbital contributions (if available)
            - 'electrostatic_potential': List of ESP values at atom positions (if available)
    """
    if not os.path.exists(orca_output_path):
        raise FileNotFoundError(f"ORCA output file not found: {orca_output_path}")
    
    # Initialize result dictionary
    result = {}
    
    # Read the ORCA output file
    with open(orca_output_path, 'r') as f:
        content = f.read()
    
    # Extract Mulliken charges
    mulliken_charges = extract_mulliken_charges(content)
    if mulliken_charges:
        result['mulliken_charges'] = mulliken_charges
    
    # Extract Loewdin charges if available
    loewdin_charges = extract_loewdin_charges(content)
    if loewdin_charges:
        result['loewdin_charges'] = loewdin_charges
    
    # Extract dipole moment if available
    dipole_moment = extract_dipole_moment(content)
    if dipole_moment is not None:
        result['dipole_moment'] = dipole_moment
    
    # Extract HOMO-LUMO gap if available
    homo_lumo_gap = extract_homo_lumo_gap(content)
    if homo_lumo_gap is not None:
        result['homo_lumo_gap'] = homo_lumo_gap
    
    # Extract HOMO/LUMO contributions if available
    homo_lumo_contributions = extract_homo_lumo_contributions(content)
    if homo_lumo_contributions is not None:
        result['homo_lumo_contributions'] = homo_lumo_contributions
    
    # Extract electrostatic potential values if available
    electrostatic_potential = extract_electrostatic_potential(content)
    if electrostatic_potential is not None:
        result['electrostatic_potential'] = electrostatic_potential
    
    return result


def extract_mulliken_charges(content: str) -> List[float]:
    """
    Extract Mulliken atomic charges from ORCA output.
    
    Args:
        content: Content of the ORCA output file
        
    Returns:
        List of Mulliken charges for each atom
    """
    # Pattern to match the Mulliken charges section
    pattern = r"MULLIKEN ATOMIC CHARGES.*?-{5,}(.*?)(?:-{5,}|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    
    charges = []
    if match:
        charge_section = match.group(1).strip()
        lines = charge_section.split('\n')
        
        # Process each line to extract the charge
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # Format is usually: <atom index> <atom symbol> <charge>
                        charge = float(parts[2])
                        charges.append(charge)
                    except (ValueError, IndexError):
                        continue
    
    return charges


def extract_loewdin_charges(content: str) -> List[float]:
    """
    Extract Loewdin atomic charges from ORCA output.
    
    Args:
        content: Content of the ORCA output file
        
    Returns:
        List of Loewdin charges for each atom, or empty list if not found
    """
    # Pattern to match the Loewdin charges section
    pattern = r"LOEWDIN ATOMIC CHARGES.*?-{5,}(.*?)(?:-{5,}|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    
    charges = []
    if match:
        charge_section = match.group(1).strip()
        lines = charge_section.split('\n')
        
        # Process each line to extract the charge
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # Format is usually: <atom index> <atom symbol> <charge>
                        charge = float(parts[2])
                        charges.append(charge)
                    except (ValueError, IndexError):
                        continue
    
    return charges


def extract_dipole_moment(content: str) -> Optional[np.ndarray]:
    """
    Extract dipole moment from ORCA output.
    
    Args:
        content: Content of the ORCA output file
        
    Returns:
        Numpy array [dx, dy, dz, total] containing dipole moment components and total,
        or None if not found
    """
    # Pattern to match the dipole moment section
    pattern = r"DIPOLE MOMENT\s*\n.*X\s+([-\d.]+).*\n.*Y\s+([-\d.]+).*\n.*Z\s+([-\d.]+).*\n.*Total\s+([-\d.]+)"
    match = re.search(pattern, content)
    
    if match:
        dx = float(match.group(1))
        dy = float(match.group(2))
        dz = float(match.group(3))
        total = float(match.group(4))
        return np.array([dx, dy, dz, total])
    
    return None


def extract_homo_lumo_gap(content: str) -> Optional[float]:
    """
    Extract HOMO-LUMO gap from ORCA output.
    
    Args:
        content: Content of the ORCA output file
        
    Returns:
        HOMO-LUMO gap in eV, or None if not found
    """
    # First try to find the explicit HOMO-LUMO gap
    pattern = r"HOMO-LUMO GAP.*?(\d+\.\d+)\s+eV"
    match = re.search(pattern, content)
    
    if match:
        return float(match.group(1))
    
    # If not found, try to calculate from the orbital energies
    # Find the orbital energies section
    pattern = r"ORBITAL ENERGIES.*?\n.*?((?:\n.*?)+?)(?:\n\n|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        orbital_section = match.group(1)
        homo_energy = None
        lumo_energy = None
        
        # Process each line to find HOMO and LUMO
        lines = orbital_section.split('\n')
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                try:
                    occ = float(parts[1])  # Occupation number
                    energy = float(parts[2])  # Energy in Eh
                    
                    if occ > 0 and (homo_energy is None or energy > homo_energy):
                        homo_energy = energy
                    elif occ == 0 and (lumo_energy is None or energy < lumo_energy):
                        lumo_energy = energy
                except (ValueError, IndexError):
                    continue
        
        # Calculate gap in eV (1 Hartree = 27.2114 eV)
        if homo_energy is not None and lumo_energy is not None:
            gap_hartree = lumo_energy - homo_energy
            gap_ev = gap_hartree * 27.2114
            return gap_ev
    
    return None


def extract_partial_charges_from_orca(orca_output_path: str, 
                                     charge_type: str = 'mulliken') -> List[float]:
    """
    Extract partial charges from ORCA output file.
    
    Args:
        orca_output_path: Path to the ORCA output file
        charge_type: Type of charges to extract ('mulliken' or 'loewdin')
        
    Returns:
        List of partial charges
    """
    data = parse_orca_output(orca_output_path)
    
    if charge_type.lower() == 'mulliken' and 'mulliken_charges' in data:
        return data['mulliken_charges']
    elif charge_type.lower() == 'loewdin' and 'loewdin_charges' in data:
        return data['loewdin_charges']
    else:
        available_types = []
        if 'mulliken_charges' in data:
            available_types.append('mulliken')
        if 'loewdin_charges' in data:
            available_types.append('loewdin')
            
        if available_types:
            # Return the first available charge type
            charge_key = f"{available_types[0]}_charges"
            print(f"Warning: {charge_type} charges not found. Using {available_types[0]} charges instead.")
            return data[charge_key]
        else:
            print(f"Warning: No charges found in {orca_output_path}")
            return []


def save_charges_to_file(charges: List[float], output_path: str) -> None:
    """
    Save a list of partial charges to a file.
    
    Args:
        charges: List of partial charges
        output_path: Path to save the charges
    """
    with open(output_path, 'w') as f:
        for charge in charges:
            f.write(f"{charge}\n")
    
    print(f"Charges saved to {output_path}")


def batch_extract_charges(orca_dir: str, output_dir: str, 
                         charge_type: str = 'mulliken') -> None:
    """
    Extract partial charges from all ORCA output files in a directory.
    
    Args:
        orca_dir: Directory containing ORCA output files
        output_dir: Directory to save extracted charges
        charge_type: Type of charges to extract ('mulliken' or 'loewdin')
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(orca_dir):
        if filename.endswith('.out') or filename.endswith('.log'):
            orca_path = os.path.join(orca_dir, filename)
            
            try:
                # Extract charges
                charges = extract_partial_charges_from_orca(orca_path, charge_type)
                
                if charges:
                    # Save charges to file
                    base_name = os.path.splitext(filename)[0]
                    output_path = os.path.join(output_dir, f"{base_name}_charges.txt")
                    save_charges_to_file(charges, output_path)
                    print(f"Processed {filename} -> {output_path}")
                else:
                    print(f"No charges found in {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def extract_homo_lumo_contributions(content: str) -> Optional[Dict[str, List[float]]]:
    """
    Extract HOMO/LUMO orbital contributions per atom from ORCA output.
    
    Args:
        content: Content of the ORCA output file
        
    Returns:
        Dictionary with keys 'homo' and 'lumo', each containing a list of contributions per atom,
        or None if not found
    """
    # This is a simplified implementation - actual extraction depends on
    # the format of the ORCA output for orbital analysis
    homo_pattern = r"ORBITAL\s+\d+\s+HOMO\s*\n(.*?)(?:\n\s*ORBITAL|\Z)"
    lumo_pattern = r"ORBITAL\s+\d+\s+LUMO\s*\n(.*?)(?:\n\s*ORBITAL|\Z)"
    
    homo_match = re.search(homo_pattern, content, re.DOTALL)
    lumo_match = re.search(lumo_pattern, content, re.DOTALL)
    
    if not homo_match or not lumo_match:
        return None
    
    # Process HOMO contributions
    homo_section = homo_match.group(1)
    homo_contributions = []
    for line in homo_section.strip().split('\n'):
        # Example line format: " 1 C       0.5%"
        if re.match(r'\s*\d+\s+[A-Za-z]+\s+[\d.]+%', line):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    contribution = float(parts[2].strip('%')) / 100.0
                    homo_contributions.append(contribution)
                except (ValueError, IndexError):
                    continue
    
    # Process LUMO contributions
    lumo_section = lumo_match.group(1)
    lumo_contributions = []
    for line in lumo_section.strip().split('\n'):
        if re.match(r'\s*\d+\s+[A-Za-z]+\s+[\d.]+%', line):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    contribution = float(parts[2].strip('%')) / 100.0
                    lumo_contributions.append(contribution)
                except (ValueError, IndexError):
                    continue
    
    if not homo_contributions or not lumo_contributions:
        return None
    
    return {
        'homo': homo_contributions,
        'lumo': lumo_contributions
    }


def extract_electrostatic_potential(content: str) -> Optional[List[float]]:
    """
    Extract electrostatic potential values at atom positions from ORCA output.
    
    Args:
        content: Content of the ORCA output file
        
    Returns:
        List of electrostatic potential values at atom positions, or None if not found
    """
    # This is a simplified implementation - actual extraction depends on
    # the format of the ORCA output for ESP calculation
    esp_pattern = r"ELECTROSTATIC POTENTIAL AT ATOM POSITIONS\s*\n-+\s*\n(.*?)(?:\n\s*-+|\Z)"
    
    esp_match = re.search(esp_pattern, content, re.DOTALL)
    if not esp_match:
        return None
    
    esp_section = esp_match.group(1)
    esp_values = []
    
    for line in esp_section.strip().split('\n'):
        # Example line format: " 1 C       1.234"
        if re.match(r'\s*\d+\s+[A-Za-z]+\s+[-\d.]+', line):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    esp_value = float(parts[2])
                    esp_values.append(esp_value)
                except (ValueError, IndexError):
                    continue
    
    return esp_values if esp_values else None


def extract_orbital_contributions_from_orca(orca_output_path: str) -> Optional[List[List[float]]]:
    """
    Extract HOMO/LUMO orbital contributions from ORCA output file.
    
    Args:
        orca_output_path: Path to the ORCA output file
        
    Returns:
        List of [homo_contribution, lumo_contribution] pairs for each atom,
        or None if not available
    """
    data = parse_orca_output(orca_output_path)
    
    if 'homo_lumo_contributions' not in data:
        print(f"Warning: No HOMO/LUMO contributions found in {orca_output_path}")
        return None
    
    contributions = data['homo_lumo_contributions']
    homo = contributions['homo']
    lumo = contributions['lumo']
    
    # Make sure both lists have the same length
    if len(homo) != len(lumo):
        print(f"Warning: HOMO contributions ({len(homo)}) and LUMO contributions ({len(lumo)}) "
              f"have different lengths in {orca_output_path}")
        # Use the shorter length
        min_length = min(len(homo), len(lumo))
        homo = homo[:min_length]
        lumo = lumo[:min_length]
    
    # Combine HOMO and LUMO contributions for each atom
    return [[homo[i], lumo[i]] for i in range(len(homo))]


def extract_electrostatic_potential_from_orca(orca_output_path: str) -> Optional[List[float]]:
    """
    Extract electrostatic potential values at atom positions from ORCA output file.
    
    Args:
        orca_output_path: Path to the ORCA output file
        
    Returns:
        List of electrostatic potential values for each atom, or None if not available
    """
    data = parse_orca_output(orca_output_path)
    
    if 'electrostatic_potential' in data:
        return data['electrostatic_potential']
    else:
        print(f"Warning: No electrostatic potential values found in {orca_output_path}")
        return None 