# conftest.py  (repo root)

import pytest, torch
from typing import List, Dict, Any
 
# Default sizes reused by many HM-GNN tests
_NODE_DIMS = [16, 32, 64]
_EDGE_DIMS = [4, 8, 16]
_NODES_PER_SCALE = [10, 5, 2]
_EDGES_PER_SCALE = [20, 8, 1]


def _make_scale_data(nodes_per_scale, edges_per_scale,
                     node_dims, edge_dims) -> List[Dict[str, Any]]:
    data = []
    for n_nodes, n_edges, n_dim, e_dim in zip(
            nodes_per_scale, edges_per_scale, node_dims, edge_dims):
        x = torch.randn(n_nodes, n_dim)
        edge_index = torch.randint(0, n_nodes, (2, n_edges)) if n_edges \
                     else torch.empty(2, 0, dtype=torch.long)
        edge_attr = (torch.randn(n_edges, e_dim) if e_dim and n_edges
                     else (torch.empty(0, e_dim) if e_dim else None))
        entry = {"x": x,
                 "edge_index": edge_index,
                 "batch": torch.zeros(n_nodes, dtype=torch.long)}
        if edge_attr is not None:
            entry["edge_attr"] = edge_attr
        data.append(entry)
    return data


@pytest.fixture
def dummy_hierarchical_graph_data() -> List[Dict[str, Any]]:
    """
    A generic single-graph, 3-scale hierarchical sample.
    Shape parameters match those used elsewhere in test_hmgnn.py.
    """
    return _make_scale_data(_NODES_PER_SCALE, _EDGES_PER_SCALE,
                            _NODE_DIMS, _EDGE_DIMS)
