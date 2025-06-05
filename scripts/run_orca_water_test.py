import os
import subprocess
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_water_xyz() -> str:
    """Returns the XYZ coordinates for a water molecule."""
    return """O   0.000000    0.000000    0.117300
H   0.000000    0.757200   -0.469200
H   0.000000   -0.757200   -0.469200"""

def create_orca_input_string(charge: int = 0, multiplicity: int = 1, keywords: str = "!HF DEF2-SVP Opt FREQ CHELPG") -> str:
    """Creates the ORCA input file content string."""
    xyz_coords = get_water_xyz()
    return f"""{keywords}
%pal nprocs 1 end # Explicitly use 1 core for this simple test
* xyz {charge} {multiplicity}
{xyz_coords}
*
"""

def run_orca_water_test(orca_executable_path: str, output_dir_name: str = "orca_water_test_script_output") -> None:
    """
    Runs a single ORCA calculation for a water molecule.

    Args:
        orca_executable_path (str): The full path to the ORCA executable.
        output_dir_name (str): Name of the directory to save ORCA output files.
    """
    base_dir = Path(".") # Assumes script is run from project root
    output_dir = base_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"ORCA output will be saved in: {output_dir.resolve()}")

    input_filename = "water_test.inp"
    output_filename = "water_test.out"
    error_filename = "water_test.err"
    base_name = "water_test" # For ORCA to name its various files

    input_file_path = output_dir / input_filename
    output_file_path = output_dir / output_filename
    error_file_path = output_dir / error_filename

    orca_input_content = create_orca_input_string()

    try:
        with open(input_file_path, "w") as f:
            f.write(orca_input_content)
        logging.info(f"Generated ORCA input file: {input_file_path}")
    except IOError as e:
        logging.error(f"Error writing ORCA input file {input_file_path}: {e}")
        return

    logging.info(f"Starting ORCA calculation for {input_filename} in {output_dir}")
    
    command = [orca_executable_path, str(input_file_path.name)] # ORCA expects just the filename if cwd is set

    try:
        with open(output_file_path, "w") as f_stdout, open(error_file_path, "w") as f_stderr:
            process = subprocess.run(
                command,
                cwd=output_dir, # Run ORCA from the output directory
                stdout=f_stdout,
                stderr=f_stderr,
                text=True,
                check=False  # Don't raise exception for non-zero exit codes immediately
            )

        # Files are written directly by subprocess.
        logging.info(f"ORCA stdout saved to {output_file_path}")
        logging.info(f"ORCA stderr saved to {error_file_path}")

        if process.returncode != 0:
            logging.error(f"ORCA run failed for {input_filename} with exit code {process.returncode}.")
            # Optionally, read and log parts of the files here if needed for immediate display
            try:
                with open(output_file_path, "r") as f_out_read:
                    logging.error(f"Captured STDOUT (see {output_file_path}):\n{f_out_read.read(1000)}...")
                with open(error_file_path, "r") as f_err_read:
                    logging.error(f"Captured STDERR (see {error_file_path}):\n{f_err_read.read(1000)}...")
            except Exception as e:
                logging.warning(f"Could not read output/error files for logging: {e}")
            logging.info(f"ORCA calculation for water failed.")
        else:
            logging.info(f"ORCA run completed successfully for {input_filename}.")
            logging.info(f"Output file: {output_file_path}")
            # Check if error file is empty before unlinking
            if error_file_path.exists() and error_file_path.stat().st_size == 0:
                error_file_path.unlink()
                logging.info(f"Empty error file {error_file_path} removed.")
            elif error_file_path.exists():
                 logging.info(f"ORCA stderr (though successful run, not empty) saved to {error_file_path}")


    except FileNotFoundError:
        logging.error(f"ORCA executable not found at {orca_executable_path}. Please check the path.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during ORCA execution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single ORCA test calculation for a water molecule.")
    parser.add_argument(
        "--orca_path",
        type=str,
        default=os.environ.get("ORCA_PATH"),
        help="Full path to the ORCA executable. Must be provided via this argument or the ORCA_PATH environment variable."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="orca_water_test_script_output",
        help="Directory to save ORCA output files."
    )
    args = parser.parse_args()

    if not args.orca_path or not Path(args.orca_path).is_file():
        logging.error(f"ORCA executable not found at the specified path: {args.orca_path}")
        logging.error("Please provide the ORCA path using --orca_path or set the ORCA_PATH environment variable.")
    else:
        logging.info(f"Using ORCA executable from: {args.orca_path}")
        run_orca_water_test(orca_executable_path=args.orca_path, output_dir_name=args.output_dir)