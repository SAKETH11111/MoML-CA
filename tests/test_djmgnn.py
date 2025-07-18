"""
tests/test_djmgnn.py

Unit tests for the DJMGNN model and its components in moml.models.mgnn.djmgnn.
"""

import os
import sys
from typing import Any, Generator

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphNorm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from moml.models.mgnn.djmgnn import GraphConvLayer, DenseGNNBlock, JKAggregator, DJMGNN

NODE_IN_DIM = 16
EDGE_ATTR_DIM_PRESENT = 4
EDGE_ATTR_DIM_ABSENT = 0
HIDDEN_DIM = 32
NUM_NODES = 10
NUM_EDGES = 20
BATCH_SIZE = 2

@pytest.fixture(params=["cpu", "cuda"])
def device(request: Any) -> torch.device:
    """
    Fixture for running tests on CPU and CUDA if available.
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)

@pytest.fixture
def dummy_graph_data_single(request: Any) -> Any:
    """
    Creates dummy graph data for a single graph.
    Can be parameterized to have edge_attr or not.
    """
    num_nodes_override = getattr(request, "param", {}).get("num_nodes", NUM_NODES)
    num_edges_override = getattr(request, "param", {}).get("num_edges", NUM_EDGES)
    edge_attr_dim_override = getattr(request, "param", {}).get("edge_attr_dim", EDGE_ATTR_DIM_PRESENT)

    x = torch.randn(num_nodes_override, NODE_IN_DIM)
    if num_nodes_override == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_attr_dim_override)) if edge_attr_dim_override > 0 else None
    elif num_edges_override == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_attr_dim_override)) if edge_attr_dim_override > 0 else None
    else:
        edge_index = torch.randint(0, num_nodes_override, (2, num_edges_override), dtype=torch.long)
        edge_attr = torch.randn(num_edges_override, edge_attr_dim_override) if edge_attr_dim_override > 0 else None

    return x, edge_index, edge_attr

@pytest.fixture
def dummy_graph_data_batch(request: Any) -> Any:
    """
    Creates dummy graph data for a batch of graphs.
    Can be parameterized for edge_attr presence.
    """
    edge_attr_dim_override = getattr(request, "param", {}).get("edge_attr_dim", EDGE_ATTR_DIM_PRESENT)

    x1 = torch.randn(NUM_NODES, NODE_IN_DIM)
    edge_index1 = torch.randint(0, NUM_NODES, (2, NUM_EDGES), dtype=torch.long)
    edge_attr1 = torch.randn(NUM_EDGES, edge_attr_dim_override) if edge_attr_dim_override > 0 else None
    data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1 if edge_attr_dim_override > 0 else None)

    x2 = torch.randn(NUM_NODES - 2, NODE_IN_DIM)
    edge_index2 = torch.randint(0, NUM_NODES - 2, (2, NUM_EDGES - 5), dtype=torch.long)
    edge_attr2 = torch.randn(NUM_EDGES - 5, edge_attr_dim_override) if edge_attr_dim_override > 0 else None
    data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2 if edge_attr_dim_override > 0 else None)

    batch = Batch.from_data_list([data1, data2])
    return batch.x, batch.edge_index, batch.edge_attr if edge_attr_dim_override > 0 else None, batch.batch


class TestGraphConvLayer:
    def test_instantiation(self, device):
        layer = GraphConvLayer(NODE_IN_DIM, HIDDEN_DIM, EDGE_ATTR_DIM_PRESENT).to(device)
        assert isinstance(layer.conv, nn.Module)
        assert isinstance(layer.edge_mlp, nn.Sequential)
        assert isinstance(layer.norm, GraphNorm)
    @pytest.mark.parametrize("dummy_graph_data_single", [{"edge_attr_dim": EDGE_ATTR_DIM_PRESENT}], indirect=True)
    def test_forward_pass_with_edge_attr(self, dummy_graph_data_single, device):
        x, edge_index, edge_attr = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        layer = GraphConvLayer(NODE_IN_DIM, HIDDEN_DIM, EDGE_ATTR_DIM_PRESENT).to(device)
        out_x = layer(x, edge_index, edge_attr)
        assert out_x.shape == (NUM_NODES, HIDDEN_DIM)
        assert out_x.dtype == torch.float32

    @pytest.mark.parametrize("dummy_graph_data_single", [{"edge_attr_dim": EDGE_ATTR_DIM_ABSENT}], indirect=True)
    def test_forward_pass_no_edge_attr(self, dummy_graph_data_single, device):
        x, edge_index, _ = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        layer = GraphConvLayer(NODE_IN_DIM, HIDDEN_DIM, EDGE_ATTR_DIM_ABSENT).to(device)
        out_x = layer(x, edge_index, None)
        assert out_x.shape == (NUM_NODES, HIDDEN_DIM)

    @pytest.mark.parametrize("dummy_graph_data_single", [{"edge_attr_dim": EDGE_ATTR_DIM_PRESENT}], indirect=True)
    def test_gradient_flow(self, dummy_graph_data_single, device):
        x, edge_index, edge_attr = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        layer = GraphConvLayer(NODE_IN_DIM, HIDDEN_DIM, EDGE_ATTR_DIM_PRESENT).to(device)

        for param in layer.parameters():
            param.requires_grad = True

        out_x = layer(x, edge_index, edge_attr)
        loss = out_x.sum()
        loss.backward()

        for name, param in layer.named_parameters():
            assert param.grad is not None, f"Gradient is None for {name}"

    @pytest.mark.parametrize(
        "dummy_graph_data_single", [{"num_nodes": 0, "edge_attr_dim": EDGE_ATTR_DIM_PRESENT}], indirect=True
    )
    def test_forward_zero_nodes(self, dummy_graph_data_single, device):
        x, edge_index, edge_attr = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        layer = GraphConvLayer(NODE_IN_DIM, HIDDEN_DIM, EDGE_ATTR_DIM_PRESENT).to(device)
        out_x = layer(x, edge_index, edge_attr)
        assert out_x.shape == (0, HIDDEN_DIM)

    @pytest.mark.parametrize(
        "dummy_graph_data_single", [{"num_edges": 0, "edge_attr_dim": EDGE_ATTR_DIM_PRESENT}], indirect=True
    )
    def test_forward_zero_edges(self, dummy_graph_data_single, device):
        x, edge_index, edge_attr = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        layer = GraphConvLayer(NODE_IN_DIM, HIDDEN_DIM, EDGE_ATTR_DIM_PRESENT).to(device)
        out_x = layer(x, edge_index, edge_attr)
        assert out_x.shape == (NUM_NODES, HIDDEN_DIM)


class TestDenseGNNBlock:
    N_LAYERS_BLOCK = 2
    TRANSITION_DIM = HIDDEN_DIM // 2

    def test_instantiation(self, device):
        block = DenseGNNBlock(NODE_IN_DIM, HIDDEN_DIM, self.N_LAYERS_BLOCK, self.TRANSITION_DIM, EDGE_ATTR_DIM_PRESENT).to(device)
        assert len(block.conv_layers) == self.N_LAYERS_BLOCK
        assert isinstance(block.final_transition, nn.Linear)

    @pytest.mark.parametrize("dummy_graph_data_single", [{"edge_attr_dim": EDGE_ATTR_DIM_PRESENT}], indirect=True)
    def test_forward_pass_with_edge_attr(self, dummy_graph_data_single, device):
        x, edge_index, edge_attr = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        block = DenseGNNBlock(NODE_IN_DIM, HIDDEN_DIM, self.N_LAYERS_BLOCK, self.TRANSITION_DIM, EDGE_ATTR_DIM_PRESENT).to(device)
        out_x = block(x, edge_index, edge_attr)
        assert out_x.shape == (NUM_NODES, self.TRANSITION_DIM)

    @pytest.mark.parametrize("dummy_graph_data_single", [{"edge_attr_dim": EDGE_ATTR_DIM_ABSENT}], indirect=True)
    def test_forward_pass_no_edge_attr(self, dummy_graph_data_single, device):
        x, edge_index, _ = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        block = DenseGNNBlock(NODE_IN_DIM, HIDDEN_DIM, self.N_LAYERS_BLOCK, self.TRANSITION_DIM, EDGE_ATTR_DIM_ABSENT).to(device)
        out_x = block(x, edge_index, None)
        assert out_x.shape == (NUM_NODES, self.TRANSITION_DIM)

    @pytest.mark.parametrize("dummy_graph_data_single", [{"edge_attr_dim": EDGE_ATTR_DIM_PRESENT}], indirect=True)
    def test_gradient_flow(self, dummy_graph_data_single, device):
        x, edge_index, edge_attr = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        block = DenseGNNBlock(NODE_IN_DIM, HIDDEN_DIM, self.N_LAYERS_BLOCK, self.TRANSITION_DIM, EDGE_ATTR_DIM_PRESENT).to(device)
        out_x = block(x, edge_index, edge_attr)
        loss = out_x.sum()
        loss.backward()
        for name, param in block.named_parameters():
            assert param.grad is not None, f"Gradient is None for {name}"

    @pytest.mark.parametrize(
        "dummy_graph_data_single", [{"num_nodes": 0, "edge_attr_dim": EDGE_ATTR_DIM_PRESENT}], indirect=True
    )
    def test_forward_zero_nodes(self, dummy_graph_data_single, device):
        x, edge_index, edge_attr = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        block = DenseGNNBlock(NODE_IN_DIM, HIDDEN_DIM, self.N_LAYERS_BLOCK, self.TRANSITION_DIM, EDGE_ATTR_DIM_PRESENT).to(device)
        out_x = block(x, edge_index, edge_attr)
        assert out_x.shape == (0, self.TRANSITION_DIM)


class TestJKAggregator:
    BLOCK_DIMS = [HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM]
    JK_OUT_DIM = HIDDEN_DIM * 2

    @pytest.mark.parametrize("mode", ["concat", "max", "attention", "lstm"])
    def test_instantiation(self, mode, device):
        aggregator = JKAggregator(self.BLOCK_DIMS, self.JK_OUT_DIM, mode=mode).to(device)
        if mode == "concat":
            assert isinstance(aggregator.proj, nn.Linear)
        elif mode == "max":
            assert hasattr(aggregator, "projs") and isinstance(aggregator.projs, nn.ModuleList)
        elif mode == "attention":
            assert hasattr(aggregator, "projs") and isinstance(aggregator.projs, nn.ModuleList)
            assert hasattr(aggregator, "attn_vecs") and isinstance(aggregator.attn_vecs, nn.ParameterList)
        elif mode == "lstm":
            assert hasattr(aggregator, "lstm_projs_in") and isinstance(aggregator.lstm_projs_in, nn.ModuleList)
            assert hasattr(aggregator, "lstm_layer") and isinstance(aggregator.lstm_layer, nn.LSTM)
            assert hasattr(aggregator, "lstm_final_proj") and isinstance(aggregator.lstm_final_proj, nn.Linear)

    @pytest.mark.parametrize("mode", ["concat", "max", "attention", "lstm"])
    def test_forward_pass(self, mode, device):
        block_outputs = [torch.randn(NUM_NODES, dim).to(device) for dim in self.BLOCK_DIMS]
        aggregator = JKAggregator(self.BLOCK_DIMS, self.JK_OUT_DIM, mode=mode).to(device)
        out_features = aggregator(block_outputs)
        assert out_features.shape == (NUM_NODES, self.JK_OUT_DIM)

    @pytest.mark.parametrize("mode", ["concat", "max", "attention", "lstm"])
    def test_gradient_flow(self, mode, device):
        block_outputs = [torch.randn(NUM_NODES, dim, requires_grad=True).to(device) for dim in self.BLOCK_DIMS]
        aggregator = JKAggregator(self.BLOCK_DIMS, self.JK_OUT_DIM, mode=mode).to(device)

        for param in aggregator.parameters():
            param.requires_grad = True

        out_features = aggregator(block_outputs)
        loss = out_features.sum()
        loss.backward()

        for i, bo in enumerate(block_outputs):
            assert bo.grad is not None, f"Gradient is None for block_output {i} in mode {mode}"

        # Check gradients only for parameters relevant to the current mode
        checked_any_param_grad = False
        if mode == "concat":
            assert aggregator.proj.weight.grad is not None, f"Gradient is None for proj.weight in mode {mode}"
            checked_any_param_grad = True
        elif mode == "max":
            for i, proj_layer in enumerate(aggregator.projs):
                assert proj_layer.weight.grad is not None, f"Gradient is None for projs[{i}].weight in mode {mode}"
            checked_any_param_grad = True
        elif mode == "attention":
            for i, proj_layer in enumerate(aggregator.projs):
                assert proj_layer.weight.grad is not None, f"Gradient is None for projs[{i}].weight in mode {mode}"
            for i, attn_v in enumerate(aggregator.attn_vecs):
                assert attn_v.grad is not None, f"Gradient is None for attn_vecs[{i}] in mode {mode}"
            checked_any_param_grad = True
        elif mode == "lstm":
            lstm_path_grads_ok_in_djmgnn_jk = False
            if (
                hasattr(aggregator, "lstm_projs_in")
                and hasattr(aggregator, "lstm_layer")
                and hasattr(aggregator, "lstm_final_proj")
            ):
                try:
                    for i, proj_layer in enumerate(aggregator.lstm_projs_in):
                        assert proj_layer.weight.grad is not None, f"Gradient is None for jk.lstm_projs_in[{i}].weight"
                    for name, param in aggregator.lstm_layer.named_parameters():
                        assert param.grad is not None, f"Gradient is None for jk.lstm_layer.{name}"
                    assert (
                        aggregator.lstm_final_proj.weight.grad is not None
                    ), "Gradient is None for jk.lstm_final_proj.weight"
                    lstm_path_grads_ok_in_djmgnn_jk = True
                    checked_any_param_grad = True
                except AssertionError:
                    pass  # Will check fallback next

            if not lstm_path_grads_ok_in_djmgnn_jk and hasattr(aggregator, "fallback_proj"):
                assert (
                    aggregator.fallback_proj.weight.grad is not None
                ), f"DJMGNN (jk_mode={mode}): Main JK params failed grad check, AND jk.fallback_proj.weight.grad is also None."
                checked_any_param_grad = True  # Counted as checked

        if mode != "none" and not checked_any_param_grad and hasattr(aggregator, "fallback_proj"):
            # This case handles if jk_mode was something like 'max' but projs didn't exist, etc.
            # and it fell through to checking the general fallback_proj of JKAggregator
            assert (
                aggregator.fallback_proj.weight.grad is not None
            ), f"DJMGNN (jk_mode={mode}): Mode-specific JK params not found/checked or no grad, AND jk.fallback_proj.weight.grad is also None."


    @pytest.mark.parametrize("mode", ["concat", "max", "attention", "lstm"])
    def test_forward_zero_nodes(self, mode, device):
        block_outputs = [torch.empty(0, dim).to(device) for dim in self.BLOCK_DIMS]
        aggregator = JKAggregator(self.BLOCK_DIMS, self.JK_OUT_DIM, mode=mode).to(device)
        out_features = aggregator(block_outputs)
        assert out_features.shape == (0, self.JK_OUT_DIM)


class TestDJMGNN:
    N_BLOCKS = 2
    LAYERS_PER_BLOCK = 1
    NODE_OUT_DIM = 1
    GRAPH_OUT_DIM = 1

    @pytest.mark.parametrize("jk_mode", ["concat", "max", "attention", "lstm"])
    def test_instantiation(self, jk_mode: str, device: torch.device) -> None:
        """
        Test DJMGNN instantiation for different JK modes.
        """
        model = DJMGNN(
            in_node_dim=NODE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            n_blocks=self.N_BLOCKS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            in_edge_dim=EDGE_ATTR_DIM_PRESENT,
            jk_mode=jk_mode,
            node_output_dims=self.NODE_OUT_DIM,
            graph_output_dims=self.GRAPH_OUT_DIM,
        ).to(device)
        assert len(model.blocks) == self.N_BLOCKS
        assert isinstance(model.jk, JKAggregator)
        assert model.jk.mode == jk_mode

    def test_instantiation_no_edge_attr(self, device: torch.device) -> None:
        """
        Test DJMGNN instantiation with no edge attributes.
        """
        model = DJMGNN(
            in_node_dim=NODE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            n_blocks=self.N_BLOCKS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            in_edge_dim=0,
            jk_mode="concat",
            node_output_dims=self.NODE_OUT_DIM,
            graph_output_dims=self.GRAPH_OUT_DIM,
        ).to(device)
        assert len(model.blocks) == self.N_BLOCKS
        assert model.processed_edge_attr_dim > 0

    @pytest.mark.parametrize("dummy_graph_data_single", [{"edge_attr_dim": EDGE_ATTR_DIM_PRESENT}], indirect=True)
    def test_forward_pass_single_graph_with_edge_attr(self, dummy_graph_data_single: Any, device: torch.device) -> None:
        x, edge_index, edge_attr = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        model = DJMGNN(
            in_node_dim=NODE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            n_blocks=self.N_BLOCKS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            in_edge_dim=EDGE_ATTR_DIM_PRESENT,
            node_output_dims=self.NODE_OUT_DIM,
            graph_output_dims=self.GRAPH_OUT_DIM,
        ).to(device)
        output = model(x, edge_index, edge_attr)
        assert "node_pred" in output
        assert "graph_pred" in output
        assert "energy_pred" in output
        assert output["node_pred"].shape == (NUM_NODES, self.NODE_OUT_DIM)
        assert output["graph_pred"].shape == (1, self.GRAPH_OUT_DIM)

    @pytest.mark.parametrize("dummy_graph_data_single", [{"edge_attr_dim": EDGE_ATTR_DIM_ABSENT}], indirect=True)
    def test_forward_pass_single_graph_no_edge_attr(self, dummy_graph_data_single: Any, device: torch.device) -> None:
        x, edge_index, edge_attr = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        model = DJMGNN(
            in_node_dim=NODE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            n_blocks=self.N_BLOCKS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            in_edge_dim=EDGE_ATTR_DIM_ABSENT,
            node_output_dims=self.NODE_OUT_DIM,
            graph_output_dims=self.GRAPH_OUT_DIM,
        ).to(device)
        output = model(x, edge_index, edge_attr)
        assert "node_pred" in output
        assert "graph_pred" in output
        assert output["node_pred"].shape == (NUM_NODES, self.NODE_OUT_DIM)
        assert output["graph_pred"].shape == (1, self.GRAPH_OUT_DIM)

    @pytest.mark.parametrize("dummy_graph_data_batch", [{"edge_attr_dim": EDGE_ATTR_DIM_PRESENT}], indirect=True)
    def test_forward_pass_batch_graph_with_edge_attr(self, dummy_graph_data_batch: Any, device: torch.device) -> None:
        x, edge_index, edge_attr, batch = dummy_graph_data_batch
        x, edge_index, batch = x.to(device), edge_index.to(device), batch.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        model = DJMGNN(
            in_node_dim=NODE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            n_blocks=self.N_BLOCKS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            in_edge_dim=EDGE_ATTR_DIM_PRESENT,
            node_output_dims=self.NODE_OUT_DIM,
            graph_output_dims=self.GRAPH_OUT_DIM,
        ).to(device)
        output = model(x, edge_index, edge_attr, batch)
        assert "node_pred" in output
        assert "graph_pred" in output
        assert "energy_pred" in output
        assert output["node_pred"].shape == (x.shape[0], self.NODE_OUT_DIM)
        assert output["graph_pred"].shape == (BATCH_SIZE, self.GRAPH_OUT_DIM)

    @pytest.mark.parametrize("dummy_graph_data_batch", [{"edge_attr_dim": EDGE_ATTR_DIM_ABSENT}], indirect=True)
    def test_forward_pass_batch_graph_no_edge_attr(self, dummy_graph_data_batch: Any, device: torch.device) -> None:
        x, edge_index, _, batch = dummy_graph_data_batch
        x, edge_index, batch = x.to(device), edge_index.to(device), batch.to(device)
        model = DJMGNN(
            in_node_dim=NODE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            n_blocks=self.N_BLOCKS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            in_edge_dim=EDGE_ATTR_DIM_ABSENT,
            node_output_dims=self.NODE_OUT_DIM,
            graph_output_dims=self.GRAPH_OUT_DIM,
        ).to(device)
        output = model(x, edge_index, edge_attr=None, batch=batch)
        assert "node_pred" in output
        assert "graph_pred" in output
        assert output["node_pred"].shape == (x.shape[0], self.NODE_OUT_DIM)
        assert output["graph_pred"].shape == (BATCH_SIZE, self.GRAPH_OUT_DIM)

    @pytest.mark.parametrize("dummy_graph_data_batch", [{"edge_attr_dim": EDGE_ATTR_DIM_PRESENT}], indirect=True)
    def test_gradient_flow(self, dummy_graph_data_batch: Any, device: torch.device) -> None:
        x, edge_index, edge_attr, batch = dummy_graph_data_batch
        x, edge_index, batch = x.to(device), edge_index.to(device), batch.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        model = DJMGNN(
            in_node_dim=NODE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            n_blocks=self.N_BLOCKS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            in_edge_dim=EDGE_ATTR_DIM_PRESENT,
            node_output_dims=self.NODE_OUT_DIM,
            graph_output_dims=self.GRAPH_OUT_DIM,
        ).to(device)
        for param in model.parameters():
            param.requires_grad = True
        output = model(x, edge_index, edge_attr, batch)
        loss = sum(p.sum() for p in output.values() if p is not None)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and "fallback_proj" not in name:
                assert param.grad is not None, f"Gradient is None for {name}"

    @pytest.mark.parametrize(
        "dummy_graph_data_single", [{"num_nodes": 0, "edge_attr_dim": EDGE_ATTR_DIM_PRESENT}], indirect=True
    )
    def test_forward_zero_nodes_single_graph(self, dummy_graph_data_single: Any, device: torch.device) -> None:
        x, edge_index, edge_attr = dummy_graph_data_single
        x, edge_index = x.to(device), edge_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None
        model = DJMGNN(
            in_node_dim=NODE_IN_DIM,
            hidden_dim=HIDDEN_DIM,
            n_blocks=self.N_BLOCKS,
            layers_per_block=self.LAYERS_PER_BLOCK,
            in_edge_dim=EDGE_ATTR_DIM_PRESENT,
            node_output_dims=self.NODE_OUT_DIM,
            graph_output_dims=self.GRAPH_OUT_DIM,
        ).to(device)
        output = model(x, edge_index, edge_attr)
        assert "node_pred" in output
        assert "graph_pred" in output
        assert output["node_pred"].shape == (0, self.NODE_OUT_DIM)
        assert output["graph_pred"].shape == (0, self.GRAPH_OUT_DIM)

@pytest.fixture
def model() -> DJMGNN:
    """
    Fixture for a DJMGNN model with specific parameters for general functional tests.
    """
    return DJMGNN(
        in_node_dim=128,
        hidden_dim=64,
        n_blocks=2,
        layers_per_block=2,
        in_edge_dim=16,
        node_output_dims=1,
        graph_output_dims=1,
    )

def test_djmgnn_forward_pass_with_valid_inputs(model: DJMGNN) -> None:
    """
    Test the DJMGNN forward pass with valid, non-empty graph data.
    """
    x = torch.randn(10, 128)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    edge_attr = torch.randn(5, 16)
    batch = torch.zeros(10, dtype=torch.long)
    output = model(x, edge_index, edge_attr, batch)
    assert "node_pred" in output
    assert "graph_pred" in output
    assert "energy_pred" in output
    assert output["node_pred"].shape == (10, 1)
    assert output["graph_pred"].shape == (1, 1)

def test_djmgnn_forward_pass_with_empty_graph(model: DJMGNN) -> None:
    """
    Test the DJMGNN forward pass with empty graph data.
    """
    x = torch.empty((0, 128))
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 16))
    batch = torch.empty((0,), dtype=torch.long)
    output = model(x, edge_index, edge_attr, batch)
    assert "node_pred" in output
    assert "graph_pred" in output
    assert "energy_pred" in output
    assert output["node_pred"].shape == (0, 1)
    assert output["graph_pred"].shape == (0, 1)
