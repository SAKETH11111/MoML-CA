#!/usr/bin/env python3
"""
Quick PIMEH Position Parameter Verification

Fast test to verify the position parameter fix without full dataset loading.
Creates synthetic test data to check PIMEH behavior.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.models.mgnn.djmgnn import DJMGNN

console = Console()

# QM9 rotational constant indices (properties 16, 17, 18 are A, B, C)
ROTATIONAL_INDICES = [16, 17, 18]
ROTATIONAL_NAMES = ['A', 'B', 'C']

def create_synthetic_batch(batch_size: int = 2, num_atoms_per_molecule: int = 10):
    """Create synthetic molecular batch data for testing."""
    total_atoms = batch_size * num_atoms_per_molecule
    
    # Node features (33-dimensional: 29 original + 4 positional)
    x = torch.randn(total_atoms, 33)
    
    # 3D positions (critical for PIMEH)
    pos = torch.randn(total_atoms, 3) * 2.0  # Realistic molecular coordinates scale
    
    # Edge connections (simple ring + some random edges)
    edge_index = []
    for mol_idx in range(batch_size):
        start_atom = mol_idx * num_atoms_per_molecule
        # Create ring connections
        for i in range(num_atoms_per_molecule):
            current = start_atom + i
            next_atom = start_atom + ((i + 1) % num_atoms_per_molecule)
            edge_index.extend([[current, next_atom], [next_atom, current]])
        
        # Add some random connections
        for _ in range(5):
            atom1 = start_atom + np.random.randint(0, num_atoms_per_molecule)  
            atom2 = start_atom + np.random.randint(0, num_atoms_per_molecule)
            if atom1 != atom2:
                edge_index.extend([[atom1, atom2], [atom2, atom1]])
    
    edge_index = torch.tensor(edge_index).T
    
    # Edge attributes (should be None for in_edge_dim=0)
    edge_attr = None
    
    # Batch assignment
    batch = torch.repeat_interleave(torch.arange(batch_size), num_atoms_per_molecule)
    
    # Target values (19 QM9 properties)
    y = torch.randn(batch_size, 19)
    
    # Create batch object
    class Batch:
        def __init__(self):
            self.x = x
            self.pos = pos
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.batch = batch
            self.y = y
        
        def to(self, device):
            new_batch = Batch()
            new_batch.x = self.x.to(device)
            new_batch.pos = self.pos.to(device)
            new_batch.edge_index = self.edge_index.to(device)
            new_batch.edge_attr = self.edge_attr.to(device) if self.edge_attr is not None else None
            new_batch.batch = self.batch.to(device)
            new_batch.y = self.y.to(device)
            return new_batch
    
    return Batch()

def load_model_checkpoint():
    """Load DJMGNN model from checkpoint."""
    console.print("🤖 Loading model checkpoint...")
    
    # Try different checkpoint locations
    checkpoint_paths = [
        "checkpoints_optimized/best_checkpoint.pt",
        "checkpoints/best_checkpoint.pt",
        "checkpoints_optimized/checkpoint_step_2000.pt",
        "checkpoints_optimized/checkpoint_step_1000.pt"
    ]
    
    model = None
    for checkpoint_path in checkpoint_paths:
        try:
            if Path(checkpoint_path).exists():
                console.print(f"📁 Found checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Extract model state
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    model_state = checkpoint['model']
                else:
                    continue
                
                # Create model with same config as training script
                model = DJMGNN(
                    in_node_dim=33,  # DEFAULT_NODE_FEATURE_DIM
                    hidden_dim=160,  # mgnn_config.get("hidden_channels", 160)
                    n_blocks=4,      # mgnn_config.get("num_layers", 4)
                    in_edge_dim=0,   # mgnn_config.get("in_edge_dim", 0)
                    node_output_dims=3,
                    graph_output_dims=19,
                    energy_output_dims=1
                )
                
                model.load_state_dict(model_state)
                model.eval()
                
                console.print("✅ Model loaded successfully")
                console.print(f"🧠 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
                return model
                
        except Exception as e:
            console.print(f"⚠️ Failed to load {checkpoint_path}: {e}")
            continue
    
    if model is None:
        console.print("❌ No valid checkpoint found - creating new model for structure test")
        model = DJMGNN(
            in_node_dim=33,  # DEFAULT_NODE_FEATURE_DIM  
            hidden_dim=160,  # mgnn_config.get("hidden_channels", 160)
            n_blocks=4,      # mgnn_config.get("num_layers", 4)  
            in_edge_dim=0,   # mgnn_config.get("in_edge_dim", 0)
            node_output_dims=3,
            graph_output_dims=19,
            energy_output_dims=1
        )
        model.eval()
    
    return model

def test_pimeh_position_fix():
    """Test the PIMEH position parameter fix."""
    console.print(Panel(
        "[bold cyan]QUICK PIMEH POSITION VERIFICATION[/bold cyan]\n"
        "Testing position parameter fix with synthetic data",
        title="🔬 PIMEH Test"
    ))
    
    # Load model
    model = load_model_checkpoint()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test batch
    console.print("\n🧪 Creating synthetic test batch...")
    batch = create_synthetic_batch(batch_size=3, num_atoms_per_molecule=8)
    batch = batch.to(device)
    
    console.print(f"📊 Test batch: {batch.batch.max().item() + 1} molecules, {batch.x.shape[0]} atoms")
    console.print(f"📍 Positions shape: {batch.pos.shape}")
    console.print(f"🔗 Edges: {batch.edge_index.shape[1]}")
    
    results = {}
    
    with torch.no_grad():
        # Test 1: WITH positions (FIXED version)
        console.print("\n🟢 [bold green]Test 1: Model WITH positions (FIXED)[/bold green]")
        try:
            output_with_pos = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                pos=batch.pos  # THIS IS THE CRITICAL FIX
            )
            
            # Handle dictionary output (check actual structure)
            if isinstance(output_with_pos, dict):
                if 'graph_pred' in output_with_pos:
                    graph_pred = output_with_pos['graph_pred']
                    console.print(f"📊 Graph pred shape: {graph_pred.shape}")
                    rot_constants_fixed = graph_pred[:, ROTATIONAL_INDICES]
                else:
                    console.print(f"❌ Unexpected output keys: {list(output_with_pos.keys())}")
                    return False
            else:
                console.print(f"📊 Direct tensor shape: {output_with_pos.shape}")
                rot_constants_fixed = output_with_pos[:, ROTATIONAL_INDICES]
            
            results['with_pos'] = rot_constants_fixed
            
            console.print("✅ Model call successful")
            
            # Display rotational constants
            table = Table(title="Rotational Constants WITH Positions")
            table.add_column("Molecule", style="cyan")
            for name in ROTATIONAL_NAMES:
                table.add_column(f"Rot {name}", style="green")
            
            for mol_idx in range(rot_constants_fixed.shape[0]):
                row = [f"Mol {mol_idx}"]
                for rot_idx in range(3):
                    value = rot_constants_fixed[mol_idx, rot_idx].item()
                    row.append(f"{value:.4f}")
                table.add_row(*row)
            
            console.print(table)
            
            # Statistics
            for i, name in enumerate(ROTATIONAL_NAMES):
                values = rot_constants_fixed[:, i]
                console.print(f"🌀 Rot {name}: min={values.min():.4f}, max={values.max():.4f}, std={values.std():.4f}")
            
        except Exception as e:
            console.print(f"❌ Error in model call with positions: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 2: WITHOUT positions (BROKEN version)
        console.print("\n🔴 [bold red]Test 2: Model WITHOUT positions (BROKEN)[/bold red]")
        try:
            output_without_pos = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                pos=None  # This should trigger fallback values
            )
            
            # Handle dictionary output (check actual structure)
            if isinstance(output_without_pos, dict):
                if 'graph_pred' in output_without_pos:
                    graph_pred = output_without_pos['graph_pred']
                    rot_constants_broken = graph_pred[:, ROTATIONAL_INDICES]
                else:
                    console.print(f"❌ Unexpected output keys: {list(output_without_pos.keys())}")
                    return False
            else:
                rot_constants_broken = output_without_pos[:, ROTATIONAL_INDICES]
            
            results['without_pos'] = rot_constants_broken
            
            console.print("✅ Model call successful")
            
            # Display rotational constants
            table = Table(title="Rotational Constants WITHOUT Positions")
            table.add_column("Molecule", style="cyan")
            for name in ROTATIONAL_NAMES:
                table.add_column(f"Rot {name}", style="red")
            
            for mol_idx in range(rot_constants_broken.shape[0]):
                row = [f"Mol {mol_idx}"]
                for rot_idx in range(3):
                    value = rot_constants_broken[mol_idx, rot_idx].item()
                    row.append(f"{value:.4f}")
                table.add_row(*row)
            
            console.print(table)
            
            # Check for fallback pattern (should be ~10.0 GHz)
            fallback_threshold = 0.1
            is_fallback = torch.all(torch.abs(rot_constants_broken - 10.0) < fallback_threshold)
            
            if is_fallback:
                console.print("⚠️ [orange]All values are ~10.0 GHz (expected fallback behavior)[/orange]")
            else:
                console.print("🤔 [yellow]Values are not fallbacks - unexpected![/yellow]")
                for i, name in enumerate(ROTATIONAL_NAMES):
                    values = rot_constants_broken[:, i]
                    console.print(f"🌀 Rot {name}: min={values.min():.4f}, max={values.max():.4f}, std={values.std():.4f}")
            
        except Exception as e:
            console.print(f"❌ Error in model call without positions: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Comparison analysis
    if 'with_pos' in results and 'without_pos' in results:
        console.print("\n📊 [bold blue]COMPARISON ANALYSIS[/bold blue]")
        
        with_pos = results['with_pos']
        without_pos = results['without_pos']
        
        # Check if they're different
        are_different = not torch.allclose(with_pos, without_pos, atol=0.01)
        
        console.print(f"🔍 Outputs are different: {'✅ YES' if are_different else '❌ NO'}")
        
        if are_different:
            console.print("✅ [green]SUCCESS: Position parameter affects PIMEH output![/green]")
            
            # Show differences
            diff_table = Table(title="Difference Analysis")
            diff_table.add_column("Property", style="bold")
            diff_table.add_column("WITH Pos", style="green")
            diff_table.add_column("WITHOUT Pos", style="red")
            diff_table.add_column("Difference", style="yellow")
            
            for i, name in enumerate(ROTATIONAL_NAMES):
                with_vals = with_pos[:, i]
                without_vals = without_pos[:, i]
                diff = torch.abs(with_vals - without_vals).mean()
                
                diff_table.add_row(
                    f"Rot {name}",
                    f"μ={with_vals.mean():.4f}, σ={with_vals.std():.4f}",
                    f"μ={without_vals.mean():.4f}, σ={without_vals.std():.4f}",
                    f"{diff:.4f}"
                )
            
            console.print(diff_table)
            
        else:
            console.print("❌ [red]FAILURE: Position parameter has no effect![/red]")
    
    # Final verdict
    console.print("\n🎯 [bold white]VERIFICATION SUMMARY[/bold white]")
    
    success_criteria = [
        ("Model loads successfully", model is not None),
        ("Model runs with positions", 'with_pos' in results),
        ("Model runs without positions", 'without_pos' in results),
        ("Outputs are different", 'with_pos' in results and 'without_pos' in results and 
         not torch.allclose(results['with_pos'], results['without_pos'], atol=0.01))
    ]
    
    all_passed = True
    for criterion, passed in success_criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        console.print(f"  {criterion}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        console.print("\n🎉 [bold green]OVERALL: POSITION FIX IS WORKING![/bold green]")
        console.print("🚀 Ready to run full validation with fixed model")
    else:
        console.print("\n💥 [bold red]OVERALL: POSITION FIX NEEDS DEBUGGING![/bold red]")
    
    return all_passed

if __name__ == "__main__":
    test_pimeh_position_fix()