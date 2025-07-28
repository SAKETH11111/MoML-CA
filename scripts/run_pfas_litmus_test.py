#!/usr/bin/env python3
"""
PROJECT GALAHAD - PFAS Fine-Tuning Litmus Test

This script performs a critical transfer learning validation to determine if our
QM9-pretrained DJMGNN backbone (from PROJECT APOLLO) provides useful features
for PFAS force field parameter prediction.

LITMUS TEST OBJECTIVE:
- Load pre-trained checkpoint_step_8000.pt (Phase 1: PIMEH adaptation complete)
- Freeze the backbone, only train new output heads
- Replace 19-property QM9 head with PFAS molecular descriptor head  
- Train on small PFAS sample and observe learning signal

SUCCESS CRITERIA: Clear downward trend in validation loss
FAILURE CRITERIA: Loss stagnation/random fluctuation (backbone not transferable)

This single experiment determines our entire subsequent strategy.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torchvision.transforms import Compose

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RDKit for SMILES processing
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available. PFAS dataset creation will be limited.")

try:
    from moml.data.dataset import get_dataset
    from moml.data.feature_transforms import CreateEdges, FeaturizeNodes, AddPositionalFeatures
    from moml.models.mgnn.djmgnn import DJMGNN
    from moml.utils.dataset_utils import SubsetWrapper
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import required modules: {e}")
    print("Ensure you're running from the project root with proper environment.")
    sys.exit(1)

# Constants
CHECKPOINT_PATH = "checkpoints_apollo_final/checkpoint_step_8000.pt"
NODE_FEATURE_DIM = 33  # After featurization (checkpoint was trained with this)
PFAS_DESCRIPTOR_DIM = 19  # Molecular descriptors as FF parameter proxies
TEST_BATCH_SIZE = 16
MAX_EPOCHS = 100
MIN_PFAS_SAMPLES = 10  # Minimum for litmus test (reduced for available data)
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PFASForceFieldHead(nn.Module):
    """
    Simplified output head for PFAS molecular descriptor prediction.
    
    This head replaces the original 19-property QM9 head and predicts
    PFAS molecular descriptors that serve as proxies for force field
    parameters (molecular weight, LogP, H-bond properties, etc.).
    """
    
    def __init__(self, hidden_dim: int, output_dim: int = PFAS_DESCRIPTOR_DIM):
        """
        Initialize the PFAS force field head.
        
        Args:
            hidden_dim: Input feature dimension from backbone
            output_dim: Number of molecular descriptors to predict
        """
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights for stable training
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PFAS head."""
        return self.head(x)


class LitmusTestRunner:
    """
    PROJECT GALAHAD Litmus Test Runner.
    
    Orchestrates the critical transfer learning experiment to validate
    whether QM9-pretrained backbone can transfer to PFAS prediction.
    """
    
    def __init__(self):
        self.model = None
        self.pfas_data = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.loss_history = []
        
    def load_pretrained_backbone(self, checkpoint_path: str = CHECKPOINT_PATH) -> DJMGNN:
        """
        Load the pre-trained DJMGNN backbone from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            DJMGNN model with frozen backbone parameters
        """
        logger.info(f"🔄 Loading pre-trained backbone from {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"❌ CRITICAL: Checkpoint not found at {checkpoint_path}\n"
                f"PROJECT GALAHAD requires the Phase 1 checkpoint from PROJECT APOLLO."
            )
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            logger.info(f"Checkpoint loaded: step={checkpoint.get('step', 'unknown')}")
            
            # Create model with same architecture as training
            model = DJMGNN(
                in_node_dim=NODE_FEATURE_DIM,
                hidden_dim=160,  # PROJECT APOLLO architecture
                n_blocks=4,
                layers_per_block=6,
                in_edge_dim=0,
                node_output_dims=3,
                graph_output_dims=19,  # Will be replaced
                energy_output_dims=1
            )
            
            # Load trained weights with error handling for mismatched keys
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                if "Unexpected key(s)" in str(e):
                    logger.warning("Loading with strict=False due to architecture mismatch")
                    # Load only matching keys
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    raise e
            
            model.to(DEVICE)
            
            logger.info(f"✅ Backbone loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
            return model
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load backbone: {e}")
    
    def freeze_backbone_and_replace_head(self, model: DJMGNN) -> DJMGNN:
        """
        Freeze backbone parameters and replace graph head for PFAS prediction.
        
        Args:
            model: Loaded DJMGNN model
            
        Returns:
            Modified model with frozen backbone and new PFAS head
        """
        logger.info("🧊 Freezing backbone parameters...")
        
        # Freeze all parameters first
        frozen_params = 0
        for param in model.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        # Replace the graph head with PFAS head
        model.graph_head = PFASForceFieldHead(
            hidden_dim=model.hidden_dim,
            output_dim=PFAS_DESCRIPTOR_DIM
        ).to(DEVICE)
        
        # Unfreeze only the new head
        trainable_params = 0
        for param in model.graph_head.parameters():
            param.requires_grad = True
            trainable_params += param.numel()
        
        logger.info(f"✅ Backbone frozen: {frozen_params:,} parameters")
        logger.info(f"✅ New PFAS head: {trainable_params:,} trainable parameters")
        logger.info(f"Transfer ratio: {trainable_params / frozen_params * 100:.2f}% trainable")
        
        return model
    
    def load_pfas_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load PFAS dataset and create train/validation loaders.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("📊 Loading PFAS dataset...")
        
        try:
            # Load PFAS molecules from the CSV file with SMILES strings
            pfas_csv = 'data/processed/chemical_list/pfas20_standardized.csv'
            
            if not Path(pfas_csv).exists():
                raise FileNotFoundError(
                    f"❌ CRITICAL: PFAS CSV data not found at {pfas_csv}"
                )
            
            logger.info(f"Loading PFAS SMILES data from {pfas_csv}")
            
            # Generate molecular data from SMILES
            dataset = self._create_pfas_dataset_from_smiles(pfas_csv)
            
            dataset_size = len(dataset)
            logger.info(f"PFAS dataset size: {dataset_size} molecules")
            
            if dataset_size < MIN_PFAS_SAMPLES:
                raise ValueError(
                    f"❌ CRITICAL: Dataset too small ({dataset_size} < {MIN_PFAS_SAMPLES})\n"
                    f"PROJECT GALAHAD requires at least {MIN_PFAS_SAMPLES} PFAS molecules."
                )
            
            # Use all available data for litmus test (small dataset)  
            test_size = dataset_size
            
            # Split into train/validation indices
            train_size = int(0.8 * test_size)
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, test_size))
            
            # Create Subset objects first, then wrap them
            from torch.utils.data import Subset
            train_subset = SubsetWrapper(Subset(dataset, train_indices))
            val_subset = SubsetWrapper(Subset(dataset, val_indices))
            
            # Create data loaders
            train_loader = DataLoader(
                train_subset, 
                batch_size=TEST_BATCH_SIZE, 
                shuffle=True
            )
            val_loader = DataLoader(
                val_subset, 
                batch_size=TEST_BATCH_SIZE, 
                shuffle=False
            )
            
            logger.info(f"✅ Data split: {len(train_subset)} train, {len(val_subset)} validation")
            
            # Verify first batch to ensure compatibility
            first_batch = next(iter(train_loader))
            
            # Check if we have atomic numbers (z) or node features (x)
            if hasattr(first_batch, 'z'):
                logger.info(f"Sample batch: {first_batch.z.shape[0]} atoms, target shape: {first_batch.y.shape}")
            elif hasattr(first_batch, 'x'):
                logger.info(f"Sample batch: {first_batch.x.shape} nodes, target shape: {first_batch.y.shape}")
            else:
                logger.warning("Unusual batch structure - no z or x attributes found")
            
            # Check target dimension compatibility  
            global PFAS_DESCRIPTOR_DIM
            expected_dim = PFAS_DESCRIPTOR_DIM
            
            # For batched data, check individual molecule targets
            if len(first_batch.y.shape) == 2:
                # Shape [batch_size, descriptor_dim]
                actual_dim = first_batch.y.shape[1]
            else:
                # Single molecule or flattened - need to check per-molecule size
                # Assume batch size and compute per-molecule size
                batch_size = first_batch.batch.max().item() + 1 if hasattr(first_batch, 'batch') else 1
                actual_dim = first_batch.y.shape[0] // batch_size
            
            if actual_dim != expected_dim:
                logger.warning(
                    f"Target dimension mismatch: expected {expected_dim}, got {actual_dim}. "
                    f"Adjusting model output dimension."
                )
                # Update the global constant for this session
                PFAS_DESCRIPTOR_DIM = actual_dim
            
            return train_loader, val_loader
            
        except Exception as e:
            if "No such file" in str(e) or "not Found" in str(e):
                raise FileNotFoundError(
                    f"❌ CRITICAL: PFAS dataset not found.\n"
                    f"Error: {e}\n"
                    f"This indicates no PFAS SDF files are available for testing.\n"
                    f"PROJECT GALAHAD cannot proceed without PFAS data."
                )
            else:
                raise RuntimeError(f"❌ Failed to load PFAS data: {e}")
    
    def _create_pfas_dataset_from_smiles(self, csv_path: str):
        """
        Create PFAS dataset from SMILES strings in CSV file.
        
        Args:
            csv_path: Path to CSV file containing PFAS data with SMILES
            
        Returns:
            Dataset with molecular graphs and descriptors
        """
        if not HAS_RDKIT:
            raise ImportError("RDKit is required for SMILES processing")
        
        logger.info("Creating PFAS dataset from SMILES strings...")
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} PFAS molecules from CSV")
        
        # Create molecular graphs from SMILES
        data_list = []
        successful_molecules = 0
        
        for idx, row in df.iterrows():
            try:
                smiles = row['canonical_smiles'] if 'canonical_smiles' in row else row['SMILES']
                if pd.isna(smiles):
                    continue
                    
                # Convert SMILES to molecular graph
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.debug(f"Failed to parse SMILES: {smiles}")
                    continue
                
                # Add hydrogens and generate 3D coordinates
                mol = Chem.AddHs(mol)
                
                # Simple 2D coordinates (RDKit can generate them)
                from rdkit.Chem import rdDepictor
                rdDepictor.Compute2DCoords(mol)
                
                # Convert to PyG Data object
                data = self._mol_to_pyg_data(mol, idx)
                if data is not None:
                    data_list.append(data)
                    successful_molecules += 1
                    
            except Exception as e:
                logger.debug(f"Error processing molecule {idx}: {e}")
                continue
        
        logger.info(f"Successfully created {successful_molecules} molecular graphs")
        
        if not data_list:
            raise ValueError("No valid PFAS molecules could be processed")
        
        # Create a simple dataset wrapper
        from torch_geometric.data import InMemoryDataset
        
        class PFASSMILESDataset(InMemoryDataset):
            def __init__(self, data_list, transform=None):
                super().__init__(transform=transform)
                self.data, self.slices = self.collate(data_list)
        
        # Apply transforms (matching QM9 training pipeline)
        transforms = Compose([
            FeaturizeNodes(),
            CreateEdges(cutoff=5.0),
            AddPositionalFeatures()
        ])
        
        return PFASSMILESDataset(data_list, transform=transforms)
    
    def _mol_to_pyg_data(self, mol, mol_id: int) -> Optional[Data]:
        """
        Convert RDKit molecule to PyTorch Geometric Data object.
        
        Args:
            mol: RDKit molecule object
            mol_id: Molecule identifier
            
        Returns:
            PyG Data object or None if conversion fails
        """
        try:
            # Get atomic information
            atoms = mol.GetAtoms()
            num_atoms = len(atoms)
            
            if num_atoms == 0:
                return None
            
            # Extract atomic numbers
            z = torch.tensor([atom.GetAtomicNum() for atom in atoms], dtype=torch.long)
            
            # Extract coordinates from conformer
            conf = mol.GetConformer()
            positions = []
            for i in range(num_atoms):
                pos = conf.GetAtomPosition(i)
                positions.append([pos.x, pos.y, 0.0])  # 2D coords with z=0
            
            pos = torch.tensor(positions, dtype=torch.float)
            
            # Generate molecular descriptors as targets (similar to PFAS SDF dataset)
            descriptors = self._compute_molecular_descriptors(mol)
            
            return Data(
                z=z,
                pos=pos,
                y=descriptors
            )
            
        except Exception as e:
            logger.debug(f"Error converting molecule {mol_id}: {e}")
            return None
    
    def _compute_molecular_descriptors(self, mol) -> torch.Tensor:
        """
        Compute molecular descriptors for PFAS molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Tensor of molecular descriptors
        """
        try:
            # Compute key molecular descriptors (matching PFAS SDF dataset)
            descriptors = [
                Descriptors.MolWt(mol),                    # Molecular weight
                Descriptors.ExactMolWt(mol),               # Exact molecular weight  
                Crippen.MolLogP(mol),                      # LogP (lipophilicity)
                Descriptors.TPSA(mol),                     # Topological polar surface area
                Descriptors.NumHAcceptors(mol),            # H-bond acceptors
                Descriptors.NumHDonors(mol),               # H-bond donors
                Descriptors.NumRotatableBonds(mol),        # Rotatable bonds
                Descriptors.NumAromaticRings(mol),         # Aromatic rings
                Descriptors.NumSaturatedRings(mol),        # Saturated rings
                mol.GetNumHeavyAtoms(),                    # Heavy atom count
                Descriptors.BalabanJ(mol),                 # Balaban J index
                Descriptors.BertzCT(mol),                  # Bertz complexity
                Descriptors.HallKierAlpha(mol),            # Hall-Kier alpha
                Descriptors.Kappa1(mol),                   # Kappa shape index 1
                Descriptors.Kappa2(mol),                   # Kappa shape index 2
                Descriptors.Kappa3(mol),                   # Kappa shape index 3
                Descriptors.LabuteASA(mol),                # Labute accessible surface area
                Descriptors.NumHeteroatoms(mol),           # Heteroatom count
                len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F']),  # Fluorine count
            ]
            
            # Clean descriptors (handle NaN/inf values)
            clean_descriptors = []
            for desc in descriptors:
                try:
                    value = float(desc)
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    clean_descriptors.append(value)
                except (TypeError, ValueError):
                    clean_descriptors.append(0.0)
            
            return torch.tensor(clean_descriptors, dtype=torch.float)
            
        except Exception as e:
            logger.warning(f"Error computing molecular descriptors: {e}")
            # Return zeros if computation fails
            return torch.zeros(PFAS_DESCRIPTOR_DIM, dtype=torch.float)
    
    def run_training_loop(self, max_epochs: int = MAX_EPOCHS) -> List[float]:
        """
        Run the critical training loop to detect learning signal.
        
        Args:
            max_epochs: Maximum training epochs
            
        Returns:
            List of validation losses per epoch
        """
        logger.info(f"🚀 Starting PROJECT GALAHAD training loop ({max_epochs} epochs max)")
        
        # Setup optimizer (only for trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
        
        # Setup loss function
        criterion = nn.MSELoss()
        
        # Training loop
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20  # Early stopping
        
        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch in self.train_loader:
                batch = batch.to(DEVICE)
                
                self.optimizer.zero_grad()
                
                # Forward pass (only graph prediction needed)
                output = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=None,
                    batch=batch.batch
                )
                
                # Extract graph predictions
                graph_pred = output['graph_pred']
                
                # Reshape targets to match prediction shape [batch_size, descriptor_dim]
                batch_size = graph_pred.shape[0]
                target_reshaped = batch.y.view(batch_size, -1)
                
                loss = criterion(graph_pred, target_reshaped)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in self.val_loader:
                    batch = batch.to(DEVICE)
                    
                    output = self.model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=None,
                        batch=batch.batch
                    )
                    
                    graph_pred = output['graph_pred']
                    
                    # Reshape targets to match prediction shape
                    batch_size = graph_pred.shape[0]
                    target_reshaped = batch.y.view(batch_size, -1)
                    
                    loss = criterion(graph_pred, target_reshaped) 
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # Progress logging
            if epoch % 10 == 0 or epoch < 10:
                logger.info(
                    f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, "
                    f"val_loss={avg_val_loss:.6f}"
                )
            
            # Early stopping based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch} (patience exceeded)")
                break
        
        logger.info(f"✅ Training complete. Best validation loss: {best_val_loss:.6f}")
        return val_losses
    
    def analyze_results(self, val_losses: List[float]) -> Dict[str, Any]:
        """
        Analyze training results to determine transfer learning success.
        
        Args:
            val_losses: List of validation losses per epoch
            
        Returns:
            Analysis results dictionary
        """
        logger.info("📊 Analyzing PROJECT GALAHAD results...")
        
        if len(val_losses) < 5:
            return {
                'success': False,
                'reason': 'Insufficient training epochs',
                'recommendation': 'Training failed too early'
            }
        
        # Compute learning metrics
        initial_loss = np.mean(val_losses[:3])
        final_loss = np.mean(val_losses[-3:])
        improvement = initial_loss - final_loss
        improvement_pct = (improvement / initial_loss) * 100
        
        # Trend analysis (simple linear regression slope)
        epochs = np.arange(len(val_losses))
        slope = np.corrcoef(epochs, val_losses)[0, 1] * np.std(val_losses) / np.std(epochs)
        
        # Success criteria
        min_improvement_pct = 5.0  # At least 5% improvement
        max_slope = -0.001  # Negative slope indicates learning
        
        success = (improvement_pct >= min_improvement_pct) and (slope <= max_slope)
        
        results = {
            'success': success,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'slope': slope,
            'epochs_trained': len(val_losses),
            'val_losses': val_losses
        }
        
        # Generate recommendation
        if success:
            results['reason'] = 'Clear learning signal detected'
            results['recommendation'] = (
                'SUCCESS: QM9-pretrained backbone transfers to PFAS! '
                'Proceed with full-scale PFAS fine-tuning.'
            )
        else:
            if improvement_pct < min_improvement_pct:
                results['reason'] = f'Insufficient improvement ({improvement_pct:.1f}% < {min_improvement_pct}%)'
            else:
                results['reason'] = f'No learning trend (slope={slope:.6f})'
            
            results['recommendation'] = (
                'FAILURE: No transfer learning signal. '
                'QM9 backbone may not be suitable for PFAS prediction.'
            )
        
        return results
    
    def run_litmus_test(self, checkpoint_path: str = CHECKPOINT_PATH, max_epochs: int = MAX_EPOCHS) -> Dict[str, Any]:
        """
        Execute the complete PROJECT GALAHAD litmus test.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            max_epochs: Maximum training epochs
        
        Returns:
            Complete test results and recommendations
        """
        logger.info("🧪 PROJECT GALAHAD - PFAS Transfer Learning Litmus Test")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Load and prepare model
            self.model = self.load_pretrained_backbone(checkpoint_path)
            self.model = self.freeze_backbone_and_replace_head(self.model)
            
            # Step 2: Load PFAS data
            self.train_loader, self.val_loader = self.load_pfas_data()
            
            # Step 3: Run training loop
            val_losses = self.run_training_loop(max_epochs)
            
            # Step 4: Analyze results
            results = self.analyze_results(val_losses)
            
            # Add timing info
            results['duration_minutes'] = (time.time() - start_time) / 60
            
            return results
            
        except Exception as e:
            logger.error(f"❌ PROJECT GALAHAD FAILED: {e}")
            return {
                'success': False,
                'reason': str(e),
                'recommendation': 'Fix the reported error and retry the litmus test.',
                'duration_minutes': (time.time() - start_time) / 60
            }


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted litmus test results."""
    print("\n" + "=" * 80)
    print("🧪 PROJECT GALAHAD - LITMUS TEST RESULTS")
    print("=" * 80)
    
    success = results.get('success', False)
    status = "✅ SUCCESS" if success else "❌ FAILURE"
    
    print(f"Status: {status}")
    print(f"Duration: {results.get('duration_minutes', 0):.1f} minutes")
    print(f"Reason: {results.get('reason', 'Unknown')}")
    print()
    
    if 'initial_loss' in results:
        print("📈 Learning Metrics:")
        print(f"  Initial Loss: {results['initial_loss']:.6f}")
        print(f"  Final Loss: {results['final_loss']:.6f}")
        print(f"  Improvement: {results['improvement_pct']:.1f}%")
        print(f"  Trend Slope: {results['slope']:.6f}")
        print(f"  Epochs: {results['epochs_trained']}")
        print()
    
    print("🎯 Recommendation:")
    print(f"  {results.get('recommendation', 'No recommendation available')}")
    print()
    
    if success:
        print("🚀 NEXT STEPS:")
        print("  1. Proceed with full PFAS fine-tuning")
        print("  2. Experiment with different learning rates")
        print("  3. Try unfreezing more backbone layers")
        print("  4. Scale up to full PFAS dataset")
    else:
        print("🔧 TROUBLESHOOTING:")
        print("  1. Check PFAS dataset quality and size")
        print("  2. Verify checkpoint contains good embeddings")
        print("  3. Consider different model architectures")
        print("  4. Investigate domain gap between QM9 and PFAS")
    
    print("=" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="PROJECT GALAHAD - PFAS Transfer Learning Litmus Test"
    )
    parser.add_argument(
        '--max_epochs', 
        type=int, 
        default=MAX_EPOCHS,
        help=f'Maximum training epochs (default: {MAX_EPOCHS})'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default=CHECKPOINT_PATH,
        help=f'Path to checkpoint file (default: {CHECKPOINT_PATH})'
    )
    
    args = parser.parse_args()
    
    # Run the litmus test
    runner = LitmusTestRunner()
    results = runner.run_litmus_test(
        checkpoint_path=args.checkpoint,
        max_epochs=args.max_epochs
    )
    
    # Print results
    print_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results.get('success', False) else 1)


if __name__ == "__main__":
    main()