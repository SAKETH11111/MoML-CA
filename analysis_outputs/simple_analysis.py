"""
analysis_outputs/simple_analysis.py

A lightweight molecular dynamics simulation analysis tool for PFAS compounds that provides energy drift analysis and temperature validation without pandas dependency.
"""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Constants
TIMESTEP_FS = 2e-6  # Timestep in femtoseconds converted to nanoseconds
TARGET_TEMPERATURE_K = 300.0  # Target simulation temperature in Kelvin
ENERGY_DRIFT_THRESHOLD = 5.0  # Maximum acceptable energy drift (kJ/mol/ns)
TEMPERATURE_TOLERANCE_K = 5.0  # Temperature tolerance around target (K)
PLOT_DPI = 150  # Plot resolution for saved figures
PLOT_LINEWIDTH = 0.8  # Line width for time series plots
GRID_ALPHA = 0.3  # Transparency for plot grids
LINE_ALPHA = 0.7  # Transparency for trend lines


def read_csv_simple(csv_file: str) -> Dict[str, np.ndarray]:
    """
    Read CSV file without pandas dependency.

    Parses a CSV file containing simulation data and converts numeric
    columns to numpy arrays. Handles quoted headers and skips non-numeric
    values gracefully.

    Args:
        csv_file (str): Path to the CSV file containing simulation data.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping column names to numpy
            arrays of numeric data.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        IOError: If there are issues reading the file.
    """
    data = {}
    
    with open(csv_file, 'r') as f:
        # Read and parse header
        header = f.readline().strip().split(',')
        header = [h.strip('"') for h in header]  # Remove quotes
        
        # Initialize data arrays
        for col in header:
            data[col] = []
        
        # Read data rows
        for line in f:
            values = line.strip().split(',')
            for i, val in enumerate(values):
                try:
                    data[header[i]].append(float(val))
                except (ValueError, IndexError):
                    # Skip non-numeric values or malformed rows
                    pass
    
    # Convert lists to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def analyze_simulation(csv_file: str, prefix: str) -> Tuple[float, float, str]:
    """
    Analyze simulation data and generate comprehensive plots.

    Performs energy drift analysis, temperature stability assessment, and
    generates visualization plots. Creates analysis summary with pass/fail
    criteria for simulation validation.

    Args:
        csv_file (str): Path to CSV file containing simulation data.
        prefix (str): Prefix for output files and plot titles.

    Returns:
        Tuple[float, float, str]: A tuple containing:
            - energy_drift (float): Energy drift rate in kJ/mol/ns
            - avg_temp (float): Average temperature in Kelvin
            - plot_path (str): Path to generated plot file

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If required columns are missing from CSV data.
        ValueError: If data cannot be processed or contains invalid values.
    """
    print(f"Reading data from {csv_file}...")
    data = read_csv_simple(csv_file)
    
    # Extract required columns
    try:
        steps = data['Step']
        energy = data['Potential Energy (kJ/mole)']
        temperature = data['Temperature (K)']
    except KeyError as e:
        raise KeyError(f"Required column not found in CSV: {e}")
    
    # Convert simulation steps to time in nanoseconds
    time_ns = steps * TIMESTEP_FS
    
    # Ensure output directory exists
    os.makedirs('analysis_outputs', exist_ok=True)
    original_dir = os.getcwd()
    os.chdir('analysis_outputs')
    
    try:
        # Create comprehensive analysis plot
        plt.figure(figsize=(10, 6))
        
        # Energy analysis subplot
        plt.subplot(2, 1, 1)
        plt.plot(time_ns, energy, linewidth=PLOT_LINEWIDTH)
        plt.xlabel('Time (ns)')
        plt.ylabel('Potential Energy (kJ/mol)')
        plt.title(f'{prefix.upper()} - Potential Energy vs Time')
        plt.grid(True, alpha=GRID_ALPHA)
        
        # Calculate and display energy drift
        coeffs = np.polyfit(time_ns, energy, 1)
        energy_drift = coeffs[0]
        trend_line = np.polyval(coeffs, time_ns)
        plt.plot(time_ns, trend_line, '--r', alpha=LINE_ALPHA,
                 label=f'Drift: {energy_drift:.3f} kJ/mol/ns')
        plt.legend()
        
        # Temperature analysis subplot
        plt.subplot(2, 1, 2)
        plt.plot(time_ns, temperature, linewidth=PLOT_LINEWIDTH)
        plt.axhline(y=TARGET_TEMPERATURE_K, color='r', linestyle='--',
                    alpha=LINE_ALPHA, label=f'Target: {TARGET_TEMPERATURE_K} K')
        plt.xlabel('Time (ns)')
        plt.ylabel('Temperature (K)')
        plt.title(f'{prefix.upper()} - Temperature vs Time')
        plt.grid(True, alpha=GRID_ALPHA)
        
        # Calculate and display average temperature
        avg_temp = temperature.mean()
        plt.axhline(y=avg_temp, color='g', linestyle=':', alpha=LINE_ALPHA,
                    label=f'Average: {avg_temp:.1f} K')
        plt.legend()
        
        plt.tight_layout()
        plot_path = f'{prefix}_analysis.png'
        plt.savefig(plot_path, dpi=PLOT_DPI)
        plt.close()
        
        # Generate comprehensive analysis summary
        _print_analysis_summary(prefix, time_ns, energy_drift, avg_temp,
                               energy, temperature, plot_path)
        
        return energy_drift, avg_temp, plot_path
        
    finally:
        # Return to original directory
        os.chdir(original_dir)


def _print_analysis_summary(prefix: str, time_ns: np.ndarray,
                           energy_drift: float, avg_temp: float,
                           energy: np.ndarray, temperature: np.ndarray,
                           plot_path: str) -> None:
    """
    Print comprehensive analysis summary with validation results.

    Args:
        prefix (str): Simulation identifier prefix.
        time_ns (np.ndarray): Time array in nanoseconds.
        energy_drift (float): Calculated energy drift rate.
        avg_temp (float): Average temperature.
        energy (np.ndarray): Energy time series data.
        temperature (np.ndarray): Temperature time series data.
        plot_path (str): Path to generated plot file.
    """
    # Validation checks
    energy_check = "✓ PASS" if abs(energy_drift) < ENERGY_DRIFT_THRESHOLD else "✗ FAIL"
    temp_deviation = abs(avg_temp - TARGET_TEMPERATURE_K)
    temp_check = "PASSED" if temp_deviation <= TEMPERATURE_TOLERANCE_K else "FAILED"
    
    print(f"\n{'='*50}")
    print(f"SIMULATION ANALYSIS SUMMARY - {prefix.upper()}")
    print(f"{'='*50}")
    print(f"Simulation time: {time_ns[-1]:.2f} ns")
    print(f"Energy drift: {energy_drift:.3f} kJ/mol/ns")
    print(f"  {energy_check} (threshold: ±{ENERGY_DRIFT_THRESHOLD} kJ/mol/ns)")
    print(f"Average temperature: {avg_temp:.1f} K")
    print(f"Temperature check {temp_check}: {avg_temp:.1f} K is "
          f"{'within' if temp_check == 'PASSED' else 'more than'} "
          f"{TEMPERATURE_TOLERANCE_K} K {'of' if temp_check == 'PASSED' else 'from'} "
          f"target {TARGET_TEMPERATURE_K} K")
    print(f"Final energy: {energy[-1]:.2f} kJ/mol")
    print(f"Energy range: {energy.min():.2f} to {energy.max():.2f} kJ/mol")
    print(f"Temperature range: {temperature.min():.1f} to "
          f"{temperature.max():.1f} K")
    print(f"\nGenerated plot: {plot_path}")


def main() -> int:
    """
    Main entry point for command-line interface.

    Parses command-line arguments and executes simulation analysis with
    comprehensive error handling and validation.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Analyze MD simulation data for PFAS compounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --csv simulation_data.csv --prefix pfoa
  %(prog)s --csv results.csv --prefix pfos
        """
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV file with simulation data (must contain Step, "
             "Potential Energy, and Temperature columns)"
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Output file prefix for plots and analysis results"
    )
    
    args = parser.parse_args()
    
    try:
        analyze_simulation(args.csv, args.prefix)
        return 0
    except (FileNotFoundError, KeyError, ValueError, IOError) as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 