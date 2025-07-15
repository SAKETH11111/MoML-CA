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
# MoML-CA main package initializer.
"""
MoML: Molecular machine learning framework for PFAS analysis

Public API:
- __version__: package version
- core: core graph processing and descriptor facilities
- data: data loading, splitting, and molecule‑to‑graph utilities
- models: graph neural network layers, trainers, and predictors
- pipeline: end‑to‑end orchestrators for data⇒QM⇒graph pipelines
- simulation: quantum/MD simulation helpers and parsers
- utils: miscellaneous lightweight helper functions
"""

__version__ = "0.1.0"

# Subpackages with conditional imports
try:
    from . import core
except ImportError:
    import warnings
    warnings.warn(f"Could not import core module")
    core = None

try:
    from . import data
except ImportError:
    import warnings
    warnings.warn(f"Could not import data module")
    data = None

try:
    from . import models
except ImportError:
    import warnings
    warnings.warn(f"Could not import models module")
    models = None

try:
    from . import pipeline
except ImportError:
    import warnings
    warnings.warn(f"Could not import pipeline module")
    pipeline = None

try:
    from . import simulation
except ImportError:
    import warnings
    warnings.warn(f"Could not import simulation module")
    simulation = None

try:
    from . import utils
except ImportError:
    import warnings
    warnings.warn(f"Could not import utils module")
    utils = None

__all__ = [
    "__version__",
    "core",
    "data",
    "models",
    "pipeline",
    "simulation",
    "utils",
]
