#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 MoML-CA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---
#
# Convert ORCA-parsed JSON to a single QM9-style .npz file.
"""
Convert ORCA-parsed JSON → single QM9-style .npz
Usage:
    python orca_json_to_qm9_npz.py data/qm_processed/*.json -o data/pfas_qm9.npz
"""
import argparse
import numpy as np

ANGSTROM3_TO_BOHR3 = 1 / 0.148184  # Å³ → a0³  (CODATA 2014)
KJMOL_TO_EV = 1 / 96.485332123  # kJ mol⁻¹ → eV
HARTREE_TO_EV = 27.211386245988  # Ha → eV
J_TO_CAL = 1 / 4.184  # J → cal
EV_GAP_PLACEHOLDER = np.nan


def convert_one(fp):
    R = []
    Z = []
    y = np.full(19, np.nan, dtype=np.float32)

    with open(fp, "r") as f:
        lines = f.readlines()

    # Extract atomic coordinates and numbers
    start_coords = False
    for line in lines:
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
            start_coords = True
            continue
        if start_coords and "---" in line:
            start_coords = False
            break
        if start_coords and line.strip():
            parts = line.split()
            try:
                Z.append(int(get_atomic_number(parts[0])))
                R.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except:
                pass

    R = np.array(R, dtype=np.float32)
    Z = np.array(Z, dtype=np.int8)

    # Extract energy
    for line in lines:
        if "FINAL SINGLE POINT ENERGY" in line:
            try:
                y[7] = float(line.split()[-2]) * HARTREE_TO_EV  # E_0K
            except ValueError:
                y[7] = float(line.split()[-1]) * HARTREE_TO_EV

    return R, Z, y


def get_atomic_number(symbol):
    atomic_numbers = {"H": 1, "C": 6, "O": 8, "F": 9, "Cl": 17, "Br": 35, "I": 53}
    return atomic_numbers.get(symbol, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("orca_path")
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args()

    Rs, Zs, Ys = [], [], []
    R, Z, y = convert_one(args.orca_path)
    Rs.append(R)
    Zs.append(Z)
    Ys.append(y)

    np.savez_compressed(args.out, R=Rs, Z=Zs, y=Ys)
    print(f"✅  {len(Rs)} molecules ➜ {args.out}")


if __name__ == "__main__":
    main()
