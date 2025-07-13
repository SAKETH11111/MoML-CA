#!/usr/bin/env python3
"""
Simple analysis script for PFAS MD simulations without pandas dependency.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def read_csv_simple(csv_file):
    """Read CSV file without pandas."""
    data = {}
    with open(csv_file, 'r') as f:
        # Read header
        header = f.readline().strip().split(',')
        header = [h.strip('"') for h in header]  # Remove quotes
        
        # Initialize data arrays
        for col in header:
            data[col] = []
        
        # Read data
        for line in f:
            values = line.strip().split(',')
            for i, val in enumerate(values):
                try:
                    data[header[i]].append(float(val))
                except ValueError:
                    pass  # Skip non-numeric values
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data

def analyze_simulation(csv_file, prefix):
    """Analyze simulation data and generate plots."""
    print(f"Reading data from {csv_file}...")
    data = read_csv_simple(csv_file)
    
    # Extract relevant columns
    steps = data['Step']
    energy = data['Potential Energy (kJ/mole)']
    temperature = data['Temperature (K)']
    
    # Convert time from steps to ns (2 fs timestep)
    time_ns = steps * 2e-6
    
    # Create output directory
    os.makedirs('analysis_outputs', exist_ok=True)
    os.chdir('analysis_outputs')
    
    # Energy analysis
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_ns, energy, linewidth=0.8)
    plt.xlabel('Time (ns)')
    plt.ylabel('Potential Energy (kJ/mol)')
    plt.title(f'{prefix.upper()} - Potential Energy vs Time')
    plt.grid(True, alpha=0.3)
    
    # Calculate energy drift
    coeffs = np.polyfit(time_ns, energy, 1)
    energy_drift = coeffs[0]
    trend_line = np.polyval(coeffs, time_ns)
    plt.plot(time_ns, trend_line, '--r', alpha=0.7, 
             label=f'Drift: {energy_drift:.3f} kJ/mol/ns')
    plt.legend()
    
    # Temperature analysis
    plt.subplot(2, 1, 2)
    plt.plot(time_ns, temperature, linewidth=0.8)
    plt.axhline(y=300, color='r', linestyle='--', alpha=0.7, label='Target: 300 K')
    plt.xlabel('Time (ns)')
    plt.ylabel('Temperature (K)')
    plt.title(f'{prefix.upper()} - Temperature vs Time')
    plt.grid(True, alpha=0.3)
    
    avg_temp = temperature.mean()
    plt.axhline(y=avg_temp, color='g', linestyle=':', alpha=0.7, 
                label=f'Average: {avg_temp:.1f} K')
    plt.legend()
    
    plt.tight_layout()
    plot_path = f'{prefix}_analysis.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SIMULATION ANALYSIS SUMMARY - {prefix.upper()}")
    print(f"{'='*50}")
    print(f"Simulation time: {time_ns[-1]:.2f} ns")
    print(f"Energy drift: {energy_drift:.3f} kJ/mol/ns")
    print(f"  ✓ PASS" if abs(energy_drift) < 5.0 else f"  ✗ FAIL (>{5.0} kJ/mol/ns)")
    print(f"Average temperature: {avg_temp:.1f} K")
    # Check if temperature is within 5 K of target
    target_temp = 300.0
    if abs(avg_temp - target_temp) <= 5.0:
        print(f"Temperature check PASSED: {avg_temp:.1f} K is within 5 K of target {target_temp} K")
    else:
        print(f"Temperature check FAILED: {avg_temp:.1f} K is more than 5 K from target {target_temp} K")
    print(f"Final energy: {energy[-1]:.2f} kJ/mol")
    print(f"Energy range: {energy.min():.2f} to {energy.max():.2f} kJ/mol")
    print(f"Temperature range: {temperature.min():.1f} to {temperature.max():.1f} K")
    print(f"\nGenerated plot: {plot_path}")
    
    return energy_drift, avg_temp, plot_path

def main():
    parser = argparse.ArgumentParser(description="Analyze MD simulation")
    parser.add_argument("--csv", required=True, help="CSV file with simulation data")
    parser.add_argument("--prefix", required=True, help="Output file prefix")
    
    args = parser.parse_args()
    analyze_simulation(args.csv, args.prefix)

if __name__ == "__main__":
    main() 