# MoML-CA Implementation Plan (Revised based on Detailed Blueprint)
**Original Timeline: March 21 - May 31, 2025**
**Note:** This checklist is updated based on the detailed tasks in `MoML-CA Implementation Blueprint.txt` and available progress logs. Status is best-effort.

## PHASE 1: FOUNDATION (Blueprint Weeks 1–2)
---

### Week 1: Environment Setup & Data Acquisition
- **Day 1: Set up cloud environment and development tools.**
  - [ ] Provision VM (e.g., Google Cloud with GPU)
  - [x] System Prep (build-essential, MPI etc. - *assumed covered by Docker setup*)
  - [x] Miniconda Installation
  - [x] Python Libraries Installation (RDKit, OpenMM, PyTorch, PyG, DGL, MDTraj - *covered by environment.yml*)
  - [/[ORCA Quantum Software Installation (Tools like `orca_pfas_wrapper.py` exist, implying access/setup)
  - [x] Version Control (Git repository initialization and GitHub setup)
  
- **Day 2: Identify and gather molecular data for PFAS compounds.**
  - [x] Compile PFAS List (names or SMILES)
  - [x] Programmatic Fetch of molecular data (SDF/MOL2 from PubChem/ChemSpider)
  - [/[Gather Environmental Data (pH, temp, co-contaminants)
  - [ ] Obtain Real-world Data (e.g., from Cedar Falls/Waterloo utilities - *status unknown*)

- **Day 3: Data cleaning and normalization.**
  - [x] Molecular Data Cleaning (RDKit: sanitize, add Hs, 3D conformers)
  - [/[Property Label Assembly (log Kow, adsorption coefficients, toxicity - *initial data processing scripts exist*)
  - [/[Environmental Data Prep (standardize units, resample time-series)
  - [/[Data Splits (train/validation for MGNN)

- **Day 4: Initial force-field parameter label generation (for MGNN training).**
  - [x] Define QM Calculation Protocols (ωB97X-D, def2-TZVP etc. - *protocol defined in systemPatterns.md*)
  - [/[Quantum Approach (ORCA for charges, etc. - *wrapper/parser exist, `ml_training_data.json` exists*)
  - [ ] Force Field Reference (OpenFF Toolkit for parameters - *status unknown*)
  - [/[Data Storage for FF labels (JSON/dict structure - *implied by `ml_training_data.json`*)

- **Day 5: Data integration and pipeline prototype.**
  - [x] Unified Data Pipeline (`data_loader.py` equivalent - *`dataset_loader.py`, `molecular_graph_processor.py` exist*)
  - [x] Graph Data Structure (PyTorch Geometric `Data` objects)
  - [x] Initial Pipeline Test (load example, verify graph - *pipeline_orchestrator.py and tests exist*)

- **Day 6: Preliminary analysis and project plan review.**
  - [x] Exploratory Data Analysis (EDA - *analysis plots exist*)
  - [/[Check Data Suitability (QM9 for pre-training if needed - *pfas_qm9.npz exists*)
  - [ ] Resource Check (estimate compute time for MGNN, MD, LSTM)
  - [x] Plan Refinement (adjust timeline - *ongoing, implied by detailed blueprint*)

### Week 2: Model Prototype and Dataset Finalization
- **Day 7: MGNN model design (architecture planning).**
  - [x] Model Outline (hierarchical GNN, multi-task heads, env context integration - *hmgnn.py, djmgnn.py exist*)
  - [x] Architecture Example (Pseudo-code translated to actual code)
  - [x] Review & Feedback on design

- **Day 8: Implement the MGNN model and training routine.**
  - [x] Coding the Model (PyTorch Geometric - *hmgnn.py, djmgnn.py*)
  - [x] Training Loop Setup (`trainer.py`, `callbacks.py` - Adam, combined loss, metrics, checkpoints, early stopping)
  - [x] Initial Test Run (one epoch on tiny subset - *covered by unit testing of trainer*)

- **Day 9: Begin MGNN training (multi-task learning).**
  - [ ] Full Training Run (Launch on full dataset)
  - [ ] Monitor Training Duration & Resources
  - [ ] Overnight Training Plan

- **Day 10: MGNN training completion and evaluation.**
  - [ ] Complete Training Epochs
  - [ ] Hyperparameter Tuning (if results subpar)
  - [ ] Evaluate on Validation Set (FF param error, property prediction accuracy - *test_mgnn_metrics.py, test_mgnn_predictor.py exist for framework*)
  - [ ] Record Results & Checkpoint Model

- **Day 11: MGNN insights and physics-based refinements (if needed).**
  - [ ] Analyze MGNN Outputs (visualize predictions vs actuals)
  - [ ] Incorporate Domain Knowledge (physics-informed layers, charge conservation)
  - [ ] Retrain (if major changes made)
  - [ ] Final MGNN Check & Freeze Model

- **Day 12: MD simulation subsystem setup.**
  - [ ] OpenMM Familiarization (basic simulation with standard FF)
  - [x] Integrating MGNN Parameters into OpenMM System (`ForceFieldMapper` exists and tested)

- **Day 13: Run molecular dynamics simulations for PFAS scenarios.**
  - [ ] Simulation Plan (PFAS list, conditions, duration)
  - [ ] Automate Simulations (`run_simulations.py` equivalent)
  - [ ] Parallel Execution Strategy (if applicable)
  - [ ] Start Simulation Runs & Monitor

- **Day 14: MD trajectory post-processing.**
  - [ ] Extract Time-Series Data (MDAnalysis/mdtraj: distances, RoG, interaction energies)
  - [ ] Assemble LSTM Dataset (normalize features, define prediction target - next-step)
  - [ ] Data Windowing for LSTM sequences

## PHASE 2: CORE DEVELOPMENT (Blueprint Weeks 3–6)
---

### Week 3: MGNN Finalization & Data Augmentation
- **Day 15: Finalize MGNN training and save the final model.**
  - [ ] Continue/Complete MGNN training if extended
  - [ ] Load best checkpoint, confirm validation metrics
  - [ ] Freeze MGNN (save model, export if needed)
  - [ ] Log GPU hours for MGNN training

- **Day 16: Data augmentation or pre-training (optional, if MGNN underperformed).**
  - [ ] Pre-train on QM9 (if beneficial for FF learning)
  - [ ] Generate synthetic data points (if needed)
  - [ ] Develop utility `predict_parameters(molecule)`

### Week 4: LSTM Model Development
- **Day 17: LSTM model design for time-series prediction.**
  - [ ] LSTM Architecture (layers, hidden size, input/output shape)
  - [ ] Loss Function (MSE)
  - [ ] Input Normalization Strategy
  - [ ] Plan Training/Validation Strategy for LSTM

- **Day 18: Implement and test the LSTM training loop.**
  - [ ] Data loader for LSTM sequences
  - [ ] LSTM Training Loop (optimizer, batching, BPTT)
  - [ ] Test LSTM with dummy data (e.g., sine wave)
  - [ ] Commence Training on MD-derived dataset

- **Day 19: Complete LSTM training and evaluate.**
  - [ ] Finalize LSTM Training Epochs
  - [ ] Evaluation: Multi-step prediction test, RMSE calculation
  - [ ] Qualitative plot analysis (predicted vs. actual)
  - [ ] Tune LSTM if needed (sequence length, architecture)
  - [ ] Save final LSTM model

- **Day 20: Document core model performance and fallback considerations.**
  - [ ] Summarize MGNN Results (accuracy, MSE, challenges, refinements)
  - [ ] Summarize LSTM Results (prediction capability, RMSE, limitations)
  - [ ] Outline Fallback Plans (MGNN: blend with classical FF; MD: acknowledge limitations; LSTM: simplify target)
  - [ ] Team Checkpoint / Code Repository Sync

### Week 5: Preparation for Integration
- **Day 21: Refactor code for integration.**
  - [x] Unified Directory Structure (`mgnn/`, `md_simulation/`, `lstm/`, `utils/` - *current structure is similar*)
  - [ ] MGNN Prediction Function: `predict_forcefield(pfas_smiles, env_params)`
  - [ ] LSTM Prediction Function: `forecast_dynamics(initial_sequence)`
  - [ ] Ensure Feature Consistency between training and prediction functions

- **Day 22: Integration pipeline development – part 1 (MGNN -> MD -> LSTM input).**
  - [/[Integrate MGNN and MD (`pipeline_predict.py` equivalent - `PipelineOrchestrator` exists)
    - [ ] Stage 1: MGNN predicts FF parameters & properties.
    - [ ] Stage 2: Setup and run *short* OpenMM simulation with MGNN parameters.
    - [ ] Stage 3: Extract time-series features from short MD run for LSTM input.

- **Day 23: Integration pipeline development – part 2 (LSTM -> Interpretation & Output).**
  - [ ] Continue `pipeline_predict.py` equivalent:
    - [ ] Stage 4: LSTM forecasts dynamics using `initial_sequence`.
    - [ ] Stage 5: Interpret LSTM results (stability, conformational changes).
    - [ ] Stage 6: Aggregate outputs (MGNN properties, MD observations, LSTM trends, narrative).
  - [ ] Optimize Pipeline (speed, stability, memory)
  - [ ] Automated Validation (test pipeline on a few different inputs)
  - [ ] Logging Integration

### Week 6: Integration Testing & Preliminary Validation
- **Day 24: Validate integrated model against real-world or literature data.**
  - [ ] Set up Real-Scenario Simulation (e.g., GAC removal, compare MGNN adsorption property)
  - [ ] Quantitative Check (if possible, e.g., relative reactivity A vs B)
  - [ ] Perform Case Studies (easy vs. hard to remove PFAS, degradation correlations)
  - [ ] Document Results (model prediction vs. expectation for scenarios)

- **Day 25: Implement feedback loop for model improvement (optional, stretch goal).**
  - [ ] Simulate feedback: add a mispredicted scenario outcome to training data.
  - [ ] Implement `retrain_models_if_needed(discrepancy_threshold)` (partial/conceptual).
  - [ ] Test the loop with a manual example.

- **Day 26: Final integration testing and freeze of integrated system.**
  - [ ] Full Pipeline Dry Run (simulate user queries with diverse inputs)
  - [ ] Check Output Sensibility & Formatting (units, warnings, fallback usage)
  - [ ] Generate Sample Final Output for a known PFAS
  - [ ] Lockdown Code (Git tag v1.0)

- **Day 27: Buffer / extra improvements.**
  - [ ] User Interpretability (GNNExplainer, attention mechanisms - if time)
  - [ ] Visualization Scripts (molecule with charges, predicted vs. actual plots)
  - [ ] Code Quality Refinements (comments, PEP8, refactoring)
  - [ ] Additional MD Scenarios (if time and beneficial)

- **Day 28: Write integration & validation documentation.**
  - [ ] Summarize Phase 3 outcomes (integration process, assumptions, limitations)
  - [ ] Document tested scenarios and results
  - [ ] Include references for methodologies
  - [ ] Update project README or create separate report section

## PHASE 3: INTEGRATION & VALIDATION (Blueprint Weeks 7–9 - *tasks interwoven with Weeks 5-6*)
---
*(Tasks from Blueprint Days 29-30 are effectively covered by Days 24-28 if on accelerated schedule, or represent further refinement)*

### Week 7: Extended Testing & Real-world Validation (Continued)
- **Day 29: Extended pipeline testing on diverse inputs.**
  - [ ] Diversity Testing (perfluoroethers, short/long chain, unusual functional groups)
  - [ ] Edge Cases (very large/small PFAS)
  - [ ] Result Sanity Checks (flagging erroneous outputs)

- **Day 30: Fixes and adjustments from extended testing.**
  - [ ] Address issues from Day 29 (MGNN out-of-distribution handling, FF template issues)
  - [ ] MGNN Property Scaling/Calibration
  - [ ] Re-run failed cases to confirm fixes

### Week 8: Documentation & User Interface Planning
- **Day 31: Draft user-facing documentation and result explanations.**
  - [ ] User Manual (input, output, limitations, example)
  - [ ] Visual Aids (pipeline diagram, example plots)
  - [ ] Citations for documentation

- **Day 32: Web application UI design.**
  - [ ] Choose Web Framework (Streamlit decided)
  - [ ] UI Layout (input fields, output display: properties, time-series, interpretation)
  - [ ] Setup Streamlit App Structure (`app/streamlit_app.py`)

- **Day 33: Test and refine the web dashboard.**
  - [ ] Usability Testing (various inputs, error handling)
  - [ ] UI Polish (number formatting, plot labels, interpretation text clarity)
  - [ ] Mobile/Compatibility Check

## PHASE 4: FINALIZATION & DELIVERY (Blueprint Weeks 9-10)
---

### Week 9: Final Preparations and Deployment (Continued from Blueprint Week 8)
- **Day 34: Deployment setup on Google Cloud.**
  - [ ] Deploy Streamlit App on VM (open firewall port, run server)
  - [ ] Domain/Access Configuration (optional)
  - [ ] Docker Container for App (optional stretch goal)

- **Day 35: Final Project Report and Presentation Preparation.**
  - [ ] Compile Final Report (Intro, Methodology, Results, Discussion, Conclusion)
  - [ ] Prepare Presentation Materials (slides, demo plan)

- **Day 36: Buffer for report revision and app fine-tuning.**
  - [ ] Incorporate feedback on report/presentation
  - [ ] Final App Check
  - [ ] Budget Wrap-up

### Week 10: Finalization & Delivery (Continued from Blueprint Week 9)
- **Day 37: Final system audit and backup.**
  - [ ] Comprehensive Project Review (backups, sensitive info check, reproducibility)
  - [ ] Deliverables Checklist Verification
  - [ ] Ensure project handoff readiness

- **Day 38: Launch day – deployment to end-users.**
  - [ ] Ensure App Accessibility
  - [ ] Prepare Demo/User Guide for launch
  - [ ] Monitor Resources During Demo
  - [ ] Collect Initial User Feedback

- **Day 39: (Post-delivery) Project reflection and wrap-up.**
  - [ ] Team Debrief (lessons learned, future work)
  - [ ] Final Budget Statement
  - [ ] Acknowledgments
  - [ ] Project Closure (archiving, resource shutdown)

**Milestones (from original checklist, status updated where possible):**
- **Milestone 1:** Development environment configured, initial QM calculations running, and data structures established - `[/]` (Env mostly configured, QM calcs in progress/tools ready, data structures good)
- **Milestone 2:** Functional prototypes of all three main components (MGNN, MD, LSTM) - `[/]` (MGNN prototype strong, MD setup in progress, LSTM foundation to do)
- **Milestone 3:** Functioning MGNN model trained on initial dataset - `[ ]` (Training to be done/validated)
- **Milestone 4:** Transfer learning implemented, model trained on expanded dataset - `[ ]`
- **Milestone 5:** Complete MD simulations generating usable time-series data - `[ ]`
- **Milestone 6:** Functioning LSTM model capable of predicting PFAS behavior - `[ ]`
- **Milestone 7:** Functioning end-to-end pipeline from molecule input to treatment predictions - `[/]` (Orchestrator exists, full pipeline with all trained models pending)
- **Milestone 8:** Validated models with feedback mechanism implemented - `[ ]`
- **Milestone 9:** Deployable system with user interface - `[ ]`
- **Milestone 10:** Complete project delivered with all documentation - `[/]` (Initial docs, unit tests done)