#!/usr/bin/env python3
"""
Analyze partial charges in GNN-generated force field XML files.
"""

import xml.etree.ElementTree as ET
import argparse
import numpy as np

def analyze_charges(xml_file, molecule_name):
    """
    Extract and analyze partial charges from OpenMM XML force field.
    
    Args:
        xml_file: Path to XML force field file
        molecule_name: Name of molecule for reporting
    """
    print(f"\nAnalyzing charges in {molecule_name} force field: {xml_file}")
    
    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find NonbondedForce section
    charges = []
    for force in root.findall('.//Force'):
        if force.get('type') == 'NonbondedForce':
            for particle in force.findall('Particles/Particle'):
                charge = float(particle.get('q'))
                charges.append(charge)
    
    charges = np.array(charges)
    
    # Analysis
    total_charge = charges.sum()
    abs_charges = np.abs(charges)
    
    print(f"Number of atoms: {len(charges)}")
    print(f"Total charge: {total_charge:.6f} e")
    print(f"Absolute total charge: {abs_charges.sum():.6f} e")
    print(f"Charge range: {charges.min():.4f} to {charges.max():.4f} e")
    print(f"Mean absolute charge: {abs_charges.mean():.4f} e")
    
    # Check if net charge is approximately zero
    charge_tolerance = 1e-6
    if abs(total_charge) < charge_tolerance:
        print(f"✓ PASS: Net charge ≈ 0 (|{total_charge:.6f}| < {charge_tolerance})")
    else:
        print(f"✗ FAIL: Net charge = {total_charge:.6f} (not ≈ 0)")
    
    # Print individual charges
    print(f"\nIndividual charges:")
    for i, charge in enumerate(charges):
        print(f"  Atom {i+1:2d}: {charge:8.5f} e")
    
    return total_charge, charges

def main():
    parser = argparse.ArgumentParser(description="Analyze charges in force field XML")
    parser.add_argument("--xml", required=True, help="XML force field file")
    parser.add_argument("--name", required=True, help="Molecule name")
    
    args = parser.parse_args()
    analyze_charges(args.xml, args.name)

if __name__ == "__main__":
    main() 
"""
Analyze partial charges in GNN-generated force field XML files.
"""

import xml.etree.ElementTree as ET
import argparse
import numpy as np

def analyze_charges(xml_file, molecule_name):
    """
    Extract and analyze partial charges from OpenMM XML force field.
    
    Args:
        xml_file: Path to XML force field file
        molecule_name: Name of molecule for reporting
    """
    print(f"\nAnalyzing charges in {molecule_name} force field: {xml_file}")
    
    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find NonbondedForce section
    charges = []
    for force in root.findall('.//Force'):
        if force.get('type') == 'NonbondedForce':
            for particle in force.findall('Particles/Particle'):
                charge = float(particle.get('q'))
                charges.append(charge)
    
    charges = np.array(charges)
    
    # Analysis
    total_charge = charges.sum()
    abs_charges = np.abs(charges)
    
    print(f"Number of atoms: {len(charges)}")
    print(f"Total charge: {total_charge:.6f} e")
    print(f"Absolute total charge: {abs_charges.sum():.6f} e")
    print(f"Charge range: {charges.min():.4f} to {charges.max():.4f} e")
    print(f"Mean absolute charge: {abs_charges.mean():.4f} e")
    
    # Check if net charge is approximately zero
    charge_tolerance = 1e-6
    if abs(total_charge) < charge_tolerance:
        print(f"✓ PASS: Net charge ≈ 0 (|{total_charge:.6f}| < {charge_tolerance})")
    else:
        print(f"✗ FAIL: Net charge = {total_charge:.6f} (not ≈ 0)")
    
    # Print individual charges
    print(f"\nIndividual charges:")
    for i, charge in enumerate(charges):
        print(f"  Atom {i+1:2d}: {charge:8.5f} e")
    
    return total_charge, charges

def main():
    parser = argparse.ArgumentParser(description="Analyze charges in force field XML")
    parser.add_argument("--xml", required=True, help="XML force field file")
    parser.add_argument("--name", required=True, help="Molecule name")
    
    args = parser.parse_args()
    analyze_charges(args.xml, args.name)

if __name__ == "__main__":
    main() 