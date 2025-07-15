"""Example usage of :class:`PFASDataLoader`.

This script demonstrates how to use the data loader to obtain PyG
``Data`` objects and batched mini-batches ready for MGNN training.
"""

import os
from moml.data.data_loader import PFASDataLoader


def main():
    """
    Demonstrates loading and batching molecular graph data using PFASDataLoader.
    
    Loads a single molecule by ID and prints its labels, environmental features, and node count. Then creates a batch from a list of molecule IDs and prints the number of graphs in the batch. Exceptions during loading and batching are caught and reported.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "example_dataset")
    config = {
        "environmental_features": ["ph", "temperature", "flow_rate"],
        "label_types": ["force_field_params", "molecular_properties"],
    }
    loader = PFASDataLoader(data_dir, config=config)

    # Load a single molecule
    try:
        graph, label, env = loader.load_molecule_by_id("molecule1")
        print("Loaded molecule1")
        print(" labels:", label)
        print(" env features:", env)
        print(" nodes:", graph.num_nodes)
    except Exception as e:
        print(f"Could not load molecule1")

    # Create a batch from a list of molecule ids
    try:
        batch = loader.get_batch(["molecule1", "molecule2"], batch_size=2)
        print("Batch contains", batch.num_graphs, "graphs")
    except Exception as e:
        print(f"Batching failed")


if __name__ == "__main__":
    main()
