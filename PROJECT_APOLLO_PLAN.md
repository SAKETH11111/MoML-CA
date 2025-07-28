# **Project Apollo: The Final, Definitive Master Plan**

## **1. Executive Summary & Unified Diagnosis**

This document outlines the final, synthesized master plan, "Project Apollo," derived from a deep analysis of three independent o3-pro expert agent responses. The plan provides a definitive roadmap to solve the persistent Phase 3 joint training failure and propel the DJMGNN model from its stable 78% R² plateau to our ultimate >95% accuracy target.

**The Unified Expert Diagnosis:** All three expert agents converged on the same fundamental root cause:

> **The joint training of the DJMGNN pipeline is failing due to a fundamental conflict between the representation needs of the general-purpose base model and the high-precision, geometrically-sensitive PIMEH head.** During joint training, the learning signal for the 18 "easy" chemistry tasks completely dominates and washes out the weak, noisy, and numerically sensitive gradient from the 3 rotational constant tasks. PIMEH cannot adapt to the "representational drift" of the backbone embeddings, causing its performance to collapse.

**The Unified Expert Solution:** A strategic shift from simultaneous joint optimization to a **decoupled, sequential fine-tuning paradigm.**

## **2. Synthesis of Agent Recommendations**

While all agents recommended a sequential approach, they each provided unique and valuable refinements that we will synthesize into one master strategy.

| Agent | Core Contribution & Key Insight |
| :--- | :--- |
| **Agent 1** | Proposed a **`pimeh_adapter` MLP** to bridge the gap between the frozen backbone embeddings and the PIMEH head, and identified the artificial `*0.01` scaling on the physics loss as a critical flaw. |
| **Agent 2** | Expanded on the adapter idea, recommending **dedicated lightweight GNN layers** for PIMEH to refine the embeddings. Also proposed an advanced, physics-informed loss function. |
| **Agent 3** | Focused on the **gradient dynamics**, advocating for a separate PIMEH-only training phase with its own high-learning-rate optimizer to overcome the naturally small physics gradients. |

## **3. The "Project Apollo" Master Plan: A New 4-Phase Schedule**

Project Apollo replaces the old curriculum with a new, four-phase sequential training schedule designed to eliminate task conflict.

---

### **Phase 0: Backbone Pre-training (Steps 0 → ~8,000)**
*   **Goal:** Train the main DJMGNN backbone until its performance on the 18 non-physics tasks converges.
*   **Action:**
    *   **Freeze PIMEH completely** (`pimeh_head.*.requires_grad = False`).
    *   Set the `physics_loss` weight to `0`.
    *   Train the model using the existing alternating curriculum (QM9 + SPICE).
    *   **Stop when the validation loss on the *non-physics*** tasks plateaus (estimated around 8,000 steps).

---

### **Phase 1: PIMEH Adaptation & Fine-Tuning (Steps ~8,000 → ~10,000)**
*   **Goal:** Train the specialist PIMEH head on a *stationary*, high-quality embedding distribution.
*   **Action:**
    *   **Freeze the entire DJMGNN backbone completely.**
    *   **Unfreeze only the `pimeh_head` and its new `pimeh_adapter` GNN layers.**
    *   Switch to a **new, PIMEH-only optimizer** (`torch.optim.AdamW`) with a dedicated **high learning rate** (e.g., `1e-3`).
    *   Set the `physics_loss` weight to `1.0` and all other loss weights to `0`.
    *   Train for a fixed number of steps (e.g., 2,000) or until the `physics_loss` validation plateaus.

---

### **Phase 2: Stability Check (A Quick Verification Step)**
*   **Goal:** Verify that the newly trained PIMEH head works with the frozen backbone without causing issues.
*   **Action:**
    *   Keep all model weights frozen.
    *   Run ~200 batches of inference and assert that the R² for rotational constants remains high and stable.

---

### **Phase 3: Gentle Joint Fine-Tuning (Optional Final Polish)**
*   **Goal:** Allow the entire model to make small, final adjustments together.
*   **Action:**
    *   Unfreeze all parameters.
    *   Switch back to the original **two-group optimizer**.
    *   Use a **very low learning rate** for both groups (e.g., `2e-5` for the base, `2e-6` for PIMEH).
    *   Re-enable **GradNorm** to balance the now-comparable loss signals.
    *   Train until the overall validation loss plateaus again.

---

## **4. Prioritized To-Do List for "Project Apollo"**

I will now delegate the following implementation tasks to `claude-code`.

1.  **[CRITICAL] Architectural Change: Implement the PIMEH Adapter.**
    *   In [`moml/models/mgnn/djmgnn.py`](/home/saketh/MoML-CA/moml/models/mgnn/djmgnn.py), add a new `pimeh_adapter` module consisting of two lightweight `GraphConvLayer`s.
    *   Modify the `forward` pass to route the backbone embeddings through this adapter before they are passed to the `pimeh_head`.

2.  **[CRITICAL] Training Logic Overhaul: Implement the New 4-Phase Curriculum.**
    *   In [`scripts/train_alternating_optimized.py`](/home/saketh/MoML-CA/scripts/train_alternating_optimized.py), completely refactor the `SimpleCurriculumManager` to implement the new Phase 0, 1, 2, and 3 logic.
    *   The manager must now control which parts of the model are frozen/unfrozen at each phase transition.

3.  **[HIGH] Optimizer Switching Logic.**
    *   In [`scripts/train_alternating_optimized.py`](/home/saketh/MoML-CA/scripts/train_alternating_optimized.py), add logic to create and switch between the `base_optimizer` (used in Phase 0) and the `pimeh_optimizer` (used in Phase 1).

4.  **[HIGH] Loss Function Correction.**
    *   In the `compute_losses` function, **permanently remove the `* 0.01` scaling** on the `physics_loss`. It will now be controlled by the curriculum phase weights.

5.  **[VERIFICATION] Launch the "Project Apollo" Training Run.**
    *   After all changes are implemented and verified, start a fresh training run from scratch and monitor its progress through the new four phases.