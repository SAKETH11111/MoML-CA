# This file makes the 'parser' directory a Python package.

"""
QM parser package for extracting data from quantum chemistry output files.
"""

from .orca_output_parser import parse_orca_output

__all__ = [
    "parse_orca_output"
]
