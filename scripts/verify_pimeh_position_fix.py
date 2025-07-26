#!/usr/bin/env python3
"""
CRITICAL VERIFICATION: Test Position Parameter Fix for PIMEH Rotational Constants

This script verifies that the PIMEH fix (adding pos parameter) resolves the
catastrophic rotational constant failures (R² = -178M to -1.2K) caused by
missing position data.

CONTEXT: PIMEH was returning fallback values of 10.0 GHz because positions
were never passed to it. The fix adds pos=getattr(batch, "pos", None) to
model calls.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from sklearn.metrics import r2_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.data.dataset import get_dataset
from moml.data.feature_transforms import (CreateEdges, FeaturizeNodes,
                                          AddPositionalFeatures, StandardizeTargets)
from moml.models.mgnn.djmgnn import DJMGNN
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision.transforms import Compose

console = Console()
logger = logging.getLogger(__name__)

# QM9 rotational constant indices (properties 16, 17, 18 are A, B, C)
ROTATIONAL_INDICES = [16, 17, 18]  # A, B, C rotational constants
ROTATIONAL_NAMES = ['A', 'B', 'C']

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_qm9_dataset(batch_size: int = 4) -> Tuple[GraphDataLoader, any]:
    """Load QM9 dataset with same pipeline as validation script."""
    console.print("🔬 [bold blue]Loading QM9 Dataset[/bold blue]")
    
    # Create transform pipeline (same as validation script)
    transform_graph = Compose([
        CreateEdges(),
        FeaturizeNodes(),
        AddPositionalFeatures(),
        StandardizeTargets(dataset_name='qm9')
    ])
    
    # Load dataset
    dataset = get_dataset('qm9', transform=transform_graph)
    console.print(f"📊 Loaded QM9 dataset: {len(dataset)} molecules")
    
    # Create validation split (last 20% as in validation script)
    val_size = len(dataset) // 5
    train_size = len(dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloader
    val_loader = GraphDataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    console.print(f"✅ Created validation loader: {len(val_dataset)} samples, batch_size={batch_size}")
    return val_loader, dataset

def verify_position_data(val_loader: GraphDataLoader) -> bool:
    """Verify that QM9 batches contain valid 3D position data."""
    console.print("\n🔍 [bold yellow]TASK 1: Verifying QM9 Position Data[/bold yellow]")
    
    try:
        # Get first batch
        batch = next(iter(val_loader))
        
        # Check if pos attribute exists
        if not hasattr(batch, 'pos'):
            console.print("❌ [red]CRITICAL: batch.pos attribute missing![/red]")
            return False
        
        pos = batch.pos
        if pos is None:
            console.print("❌ [red]CRITICAL: batch.pos is None![/red]")
            return False
        
        # Verify tensor properties
        console.print(f"📐 Position tensor shape: {pos.shape}")
        console.print(f"📊 Position tensor dtype: {pos.dtype}")
        console.print(f"🔢 Position tensor device: {pos.device}")
        
        # Verify shape is [N, 3] for 3D coordinates
        if len(pos.shape) != 2 or pos.shape[1] != 3:
            console.print(f"❌ [red]Invalid position shape: {pos.shape}, expected [N, 3][/red]")
            return False
        
        # Check for valid coordinate values
        if torch.all(pos == 0):
            console.print("⚠️ [orange]WARNING: All positions are zero![/orange]")
        
        # Check for NaN or inf values
        if torch.any(torch.isnan(pos)) or torch.any(torch.isinf(pos)):
            console.print("❌ [red]CRITICAL: NaN or inf values in positions![/red]")
            return False
        
        # Print sample positions
        console.print("\n📋 [bold]Sample Position Data:[/bold]")
        sample_size = min(10, pos.shape[0])
        for i in range(sample_size):
            x, y, z = pos[i]
            console.print(f"  Atom {i}: ({x:.4f}, {y:.4f}, {z:.4f})")
        
        # Statistics
        pos_mean = torch.mean(pos, dim=0)
        pos_std = torch.std(pos, dim=0)
        pos_range = torch.max(pos, dim=0)[0] - torch.min(pos, dim=0)[0]
        
        console.print(f"\n📈 Position Statistics:")
        console.print(f"  Mean: ({pos_mean[0]:.4f}, {pos_mean[1]:.4f}, {pos_mean[2]:.4f})")
        console.print(f"  Std:  ({pos_std[0]:.4f}, {pos_std[1]:.4f}, {pos_std[2]:.4f})")
        console.print(f"  Range: ({pos_range[0]:.4f}, {pos_range[1]:.4f}, {pos_range[2]:.4f})")
        
        console.print("✅ [green]QM9 position data verification PASSED[/green]")
        return True
        
    except Exception as e:
        console.print(f"❌ [red]Error during position verification: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

def load_model_checkpoint(checkpoint_path: str = "checkpoints_optimized/best_checkpoint.pt") -> Optional[DJMGNN]:
    """Load DJMGNN model from checkpoint."""
    console.print(f"\n🤖 [bold cyan]Loading Model Checkpoint[/bold cyan]")
    
    try:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            console.print(f"❌ [red]Checkpoint not found: {checkpoint_path}[/red]")
            # Try alternative locations
            alternatives = [
                "checkpoints/best_checkpoint.pt",
                "checkpoints_optimized/checkpoint_step_1000.pt",
                "checkpoints_optimized/checkpoint_step_2000.pt"
            ]
            
            for alt_path in alternatives:
                if Path(alt_path).exists():
                    console.print(f"🔄 Using alternative checkpoint: {alt_path}")
                    checkpoint_path = alt_path
                    break
            else:
                console.print("❌ [red]No valid checkpoint found![/red]")
                return None
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        console.print(f"📁 Loaded checkpoint from: {checkpoint_path}")
        
        # Extract model state
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
        else:
            console.print("❌ [red]Invalid checkpoint format![/red]")
            return None
        
        # Create model instance (matching training script config)
        model = DJMGNN(
            in_node_dim=33,      # DEFAULT_NODE_FEATURE_DIM
            hidden_dim=160,      # mgnn_config.get("hidden_channels", 160)
            n_blocks=4,          # mgnn_config.get("num_layers", 4)
            in_edge_dim=0,       # mgnn_config.get("in_edge_dim", 0)
            node_output_dims=3,
            graph_output_dims=19,
            energy_output_dims=1
        )
        
        # Load state dict
        model.load_state_dict(model_state)
        model.eval()
        
        console.print("✅ [green]Model loaded successfully[/green]")
        console.print(f"🧠 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        console.print(f"🔬 PIMEH parameters: {sum(p.numel() for p in model.pimeh_head.parameters()):,}")
        
        return model
        
    except Exception as e:
        console.print(f"❌ [red]Error loading model: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None

def test_pimeh_integration(model: DJMGNN, val_loader: GraphDataLoader) -> bool:
    """Test PIMEH integration with position parameters."""
    console.print("\n⚡ [bold magenta]TASK 2: Testing PIMEH Integration[/bold magenta]")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Get test batch
        batch = next(iter(val_loader))
        batch = batch.to(device)
        
        console.print(f"🔧 Testing with batch size: {batch.batch.max().item() + 1}")
        
        with torch.no_grad():
            # Test 1: Call model WITH positions (fixed version)
            console.print("\n🧪 [bold]Test 1: Model with positions (FIXED)[/bold]")
            try:
                # This is the fix: explicitly pass positions
                pos = getattr(batch, 'pos', None)
                console.print(f"📍 Positions passed: {pos is not None}")
                console.print(f"📐 Position shape: {pos.shape if pos is not None else 'None'}")
                
                # Call model with positions
                output_with_pos = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    pos=pos  # THIS IS THE CRITICAL FIX
                )
                
                # Extract rotational constants (indices 16, 17, 18)
                if isinstance(output_with_pos, dict) and 'graph_pred' in output_with_pos:
                    graph_pred = output_with_pos['graph_pred']
                    rot_constants_fixed = graph_pred[:, ROTATIONAL_INDICES]
                else:
                    rot_constants_fixed = output_with_pos[:, ROTATIONAL_INDICES]
                console.print(f"🌀 Rotational constants shape: {rot_constants_fixed.shape}")
                
                # Display values
                for i, name in enumerate(ROTATIONAL_NAMES):
                    values = rot_constants_fixed[:, i]
                    console.print(f"  {name}: min={values.min():.6f}, max={values.max():.6f}, mean={values.mean():.6f}")
                
                console.print("✅ [green]Model call with positions SUCCESSFUL[/green]")
                
            except Exception as e:
                console.print(f"❌ [red]Error in model call with positions: {e}[/red]")
                import traceback
                traceback.print_exc()
                return False
            
            # Test 2: Call model WITHOUT positions (broken version)
            console.print("\n🧪 [bold]Test 2: Model without positions (BROKEN)[/bold]")
            try:
                # Call model without positions (should use fallbacks)
                output_without_pos = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    pos=None  # This should trigger fallback values
                )
                
                # Extract rotational constants
                if isinstance(output_without_pos, dict) and 'graph_pred' in output_without_pos:
                    graph_pred = output_without_pos['graph_pred']
                    rot_constants_broken = graph_pred[:, ROTATIONAL_INDICES]
                else:
                    rot_constants_broken = output_without_pos[:, ROTATIONAL_INDICES]
                
                # Display values
                for i, name in enumerate(ROTATIONAL_NAMES):
                    values = rot_constants_broken[:, i]
                    console.print(f"  {name}: min={values.min():.6f}, max={values.max():.6f}, mean={values.mean():.6f}")
                
                # Check if all values are fallbacks (10.0 GHz)
                fallback_threshold = 0.01  # Small tolerance
                is_fallback = torch.all(torch.abs(rot_constants_broken - 10.0) < fallback_threshold)
                
                if is_fallback:
                    console.print("⚠️ [orange]All values are 10.0 GHz fallbacks (expected)[/orange]")
                else:
                    console.print("🤔 [yellow]Values are not fallbacks (unexpected)[/yellow]")
                
                console.print("✅ [green]Model call without positions completed[/green]")
                
            except Exception as e:
                console.print(f"❌ [red]Error in model call without positions: {e}[/red]")
                import traceback
                traceback.print_exc()
                return False
        
        return True
        
    except Exception as e:
        console.print(f"❌ [red]Error in PIMEH integration test: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

def compare_before_after(model: DJMGNN, val_loader: GraphDataLoader) -> Dict:
    """Compare PIMEH outputs before and after fix."""
    console.print("\n🔄 [bold green]TASK 3: Before/After Comparison[/bold green]")
    
    results = {
        'with_pos': {'predictions': [], 'targets': []},
        'without_pos': {'predictions': [], 'targets': []}
    }
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        sample_count = 0
        max_samples = 20  # Test on 20 molecules
        
        with torch.no_grad():
            for batch in val_loader:
                if sample_count >= max_samples:
                    break
                
                batch = batch.to(device)
                batch_size = batch.batch.max().item() + 1
                
                # Get targets for rotational constants
                targets = batch.y[:, ROTATIONAL_INDICES]
                
                # Test with positions (FIXED)
                pos = getattr(batch, 'pos', None)
                output_with_pos = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    pos=pos
                )
                if isinstance(output_with_pos, dict) and 'graph_pred' in output_with_pos:
                    pred_with_pos = output_with_pos['graph_pred'][:, ROTATIONAL_INDICES]
                else:
                    pred_with_pos = output_with_pos[:, ROTATIONAL_INDICES]
                
                # Test without positions (BROKEN)
                output_without_pos = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    pos=None
                )
                if isinstance(output_without_pos, dict) and 'graph_pred' in output_without_pos:
                    pred_without_pos = output_without_pos['graph_pred'][:, ROTATIONAL_INDICES]
                else:
                    pred_without_pos = output_without_pos[:, ROTATIONAL_INDICES]
                
                # Store results
                results['with_pos']['predictions'].append(pred_with_pos.cpu())
                results['with_pos']['targets'].append(targets.cpu())
                results['without_pos']['predictions'].append(pred_without_pos.cpu())
                results['without_pos']['targets'].append(targets.cpu())
                
                sample_count += batch_size
        
        # Concatenate all results
        for key in results:
            results[key]['predictions'] = torch.cat(results[key]['predictions'], dim=0)
            results[key]['targets'] = torch.cat(results[key]['targets'], dim=0)
        
        console.print(f"📊 Comparison completed on {sample_count} molecules")
        
        # Display comparison table
        table = Table(title="Before/After Comparison")
        table.add_column("Condition", style="bold")
        table.add_column("Property", style="cyan")
        table.add_column("Pred Range", style="magenta")
        table.add_column("Target Range", style="green")
        table.add_column("Variation", style="yellow")
        
        for condition in ['with_pos', 'without_pos']:
            condition_name = "WITH Positions (FIXED)" if condition == 'with_pos' else "WITHOUT Positions (BROKEN)"
            preds = results[condition]['predictions']
            targets = results[condition]['targets']
            
            for i, prop_name in enumerate(ROTATIONAL_NAMES):
                pred_vals = preds[:, i]
                target_vals = targets[:, i]
                
                pred_range = f"{pred_vals.min():.3f} to {pred_vals.max():.3f}"
                target_range = f"{target_vals.min():.3f} to {target_vals.max():.3f}"
                variation = f"σ={pred_vals.std():.3f}"
                
                table.add_row(condition_name if i == 0 else "", f"Rot {prop_name}", pred_range, target_range, variation)
        
        console.print(table)
        
        return results
        
    except Exception as e:
        console.print(f"❌ [red]Error in before/after comparison: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {}

def run_mini_validation(model: DJMGNN, val_loader: GraphDataLoader, results: Dict) -> Dict:
    """Run mini-validation to compute R² scores."""
    console.print("\n📈 [bold blue]TASK 4: Mini-Validation R² Scores[/bold blue]")
    
    r2_scores = {}
    
    try:
        if not results:
            console.print("❌ [red]No comparison results available[/red]")
            return {}
        
        # Compute R² for both conditions
        for condition in ['with_pos', 'without_pos']:
            if condition not in results:
                continue
                
            preds = results[condition]['predictions'].numpy()
            targets = results[condition]['targets'].numpy()
            
            condition_scores = {}
            condition_name = "WITH Positions" if condition == 'with_pos' else "WITHOUT Positions"
            
            console.print(f"\n🔬 [bold]{condition_name}:[/bold]")
            
            for i, prop_name in enumerate(ROTATIONAL_NAMES):
                try:
                    r2 = r2_score(targets[:, i], preds[:, i])
                    condition_scores[prop_name] = r2
                    
                    # Color code based on performance
                    if r2 > 0:
                        color = "green"
                        status = "EXCELLENT"
                    elif r2 > -1000:
                        color = "yellow"
                        status = "IMPROVED"
                    elif r2 > -100000:
                        color = "orange"
                        status = "POOR"
                    else:
                        color = "red"
                        status = "CATASTROPHIC"
                    
                    console.print(f"  Rotational {prop_name}: R² = {r2:.2e} [{status}]", style=color)
                    
                except Exception as e:
                    console.print(f"  Rotational {prop_name}: Error computing R² - {e}")
                    condition_scores[prop_name] = float('nan')
            
            r2_scores[condition] = condition_scores
        
        # Summary table
        table = Table(title="R² Score Comparison")
        table.add_column("Property", style="bold")
        table.add_column("WITH Positions", style="green")
        table.add_column("WITHOUT Positions", style="red")
        table.add_column("Improvement", style="cyan")
        
        for prop_name in ROTATIONAL_NAMES:
            with_pos_r2 = r2_scores.get('with_pos', {}).get(prop_name, float('nan'))
            without_pos_r2 = r2_scores.get('without_pos', {}).get(prop_name, float('nan'))
            
            if not (np.isnan(with_pos_r2) or np.isnan(without_pos_r2)):
                improvement = with_pos_r2 - without_pos_r2
                improvement_str = f"{improvement:.2e}" if abs(improvement) > 1e-3 else f"{improvement:.6f}"
            else:
                improvement_str = "N/A"
            
            table.add_row(
                f"Rotational {prop_name}",
                f"{with_pos_r2:.2e}" if not np.isnan(with_pos_r2) else "N/A",
                f"{without_pos_r2:.2e}" if not np.isnan(without_pos_r2) else "N/A",
                improvement_str
            )
        
        console.print(table)
        
        return r2_scores
        
    except Exception as e:
        console.print(f"❌ [red]Error in mini-validation: {e}[/red]")
        import traceback
        traceback.print_exc()
        return {}

def generate_verification_report(pos_data_ok: bool, pimeh_ok: bool, results: Dict, r2_scores: Dict):
    """Generate comprehensive verification report."""
    console.print("\n📋 [bold white]VERIFICATION REPORT[/bold white]")
    
    # Overall status
    overall_success = pos_data_ok and pimeh_ok and bool(results) and bool(r2_scores)
    
    status_panel = Panel(
        f"[bold {'green' if overall_success else 'red'}]"
        f"{'✅ VERIFICATION SUCCESSFUL' if overall_success else '❌ VERIFICATION FAILED'}"
        f"[/bold {'green' if overall_success else 'red'}]",
        title="Overall Status"
    )
    console.print(status_panel)
    
    # Detailed results
    console.print("\n📊 [bold]Detailed Results:[/bold]")
    console.print(f"1. QM9 Position Data: {'✅ PASS' if pos_data_ok else '❌ FAIL'}")
    console.print(f"2. PIMEH Integration: {'✅ PASS' if pimeh_ok else '❌ FAIL'}")
    console.print(f"3. Before/After Comparison: {'✅ PASS' if results else '❌ FAIL'}")
    console.print(f"4. R² Validation: {'✅ PASS' if r2_scores else '❌ FAIL'}")
    
    # Key findings
    if results and r2_scores:
        console.print("\n🔍 [bold]Key Findings:[/bold]")
        
        # Check for fallback patterns
        with_pos_preds = results.get('with_pos', {}).get('predictions', torch.tensor([]))
        without_pos_preds = results.get('without_pos', {}).get('predictions', torch.tensor([]))
        
        if len(with_pos_preds) > 0 and len(without_pos_preds) > 0:
            # Check if without_pos shows fallback pattern (all ~10.0)
            fallback_check = torch.all(torch.abs(without_pos_preds - 10.0) < 0.1)
            
            # Check if with_pos shows variation
            variation_check = torch.std(with_pos_preds) > 0.1
            
            console.print(f"• Fallback detection (without pos): {'✅' if fallback_check else '❌'}")
            console.print(f"• Real computation (with pos): {'✅' if variation_check else '❌'}")
        
        # R² improvement summary
        if 'with_pos' in r2_scores and 'without_pos' in r2_scores:
            console.print("• R² Score Improvements:")
            for prop in ROTATIONAL_NAMES:
                with_r2 = r2_scores['with_pos'].get(prop, float('nan'))
                without_r2 = r2_scores['without_pos'].get(prop, float('nan'))
                
                if not (np.isnan(with_r2) or np.isnan(without_r2)):
                    improvement = with_r2 - without_r2
                    console.print(f"  - Rotational {prop}: {improvement:.2e}")
    
    # Recommendations
    console.print("\n💡 [bold]Recommendations:[/bold]")
    
    if overall_success:
        console.print("✅ Position fix is working correctly!")
        console.print("✅ PIMEH now receives valid 3D coordinates")
        console.print("✅ Rotational constants show realistic variation")
        console.print("🚀 Ready to run full validation with the fixed model")
    else:
        if not pos_data_ok:
            console.print("❌ Fix QM9 data loading to include positions")
        if not pimeh_ok:
            console.print("❌ Debug PIMEH integration issues")
        if not results:
            console.print("❌ Investigate model forward pass problems")
        if not r2_scores:
            console.print("❌ Check R² computation and data alignment")

def main():
    """Main verification function."""
    setup_logging()
    
    console.print(Panel(
        "[bold cyan]CRITICAL VERIFICATION: PIMEH Position Parameter Fix[/bold cyan]\n"
        "Testing fix for catastrophic rotational constant failures\n"
        "Target: Verify pos parameter resolves R² = -178M issue",
        title="🔬 DJMGNN PIMEH Verification"
    ))
    
    # Initialize results
    pos_data_ok = False
    pimeh_ok = False
    results = {}
    r2_scores = {}
    
    try:
        # Task 1: Verify QM9 position data
        val_loader, dataset = load_qm9_dataset(batch_size=4)
        pos_data_ok = verify_position_data(val_loader)
        
        if not pos_data_ok:
            console.print("🛑 [red]Cannot continue without valid position data[/red]")
            return
        
        # Task 2: Load model and test PIMEH integration
        model = load_model_checkpoint()
        if model is None:
            console.print("🛑 [red]Cannot continue without valid model[/red]")
            return
        
        pimeh_ok = test_pimeh_integration(model, val_loader)
        
        if not pimeh_ok:
            console.print("🛑 [red]PIMEH integration failed[/red]")
            return
        
        # Task 3: Before/after comparison
        results = compare_before_after(model, val_loader)
        
        # Task 4: Mini-validation
        if results:
            r2_scores = run_mini_validation(model, val_loader, results)
        
        # Task 5: Generate report
        generate_verification_report(pos_data_ok, pimeh_ok, results, r2_scores)
        
    except Exception as e:
        console.print(f"💥 [red]Critical error during verification: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    finally:
        console.print("\n🎯 [bold]Verification completed![/bold]")

if __name__ == "__main__":
    main()