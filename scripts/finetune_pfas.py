import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader as GraphDataLoader

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moml.data.dataset import get_dataset
from moml.models.mgnn.djmgnn import DJMGNN
from moml.data.feature_transforms import CreateEdges, FeaturizeNodes, StandardizeTargets
from torchvision.transforms import Compose

def main():
    parser = argparse.ArgumentParser(description='Fine-tune DJMGNN on PFAS dataset.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the pre-trained model checkpoint.')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum fine-tuning steps.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for fine-tuning.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for fine-tuning.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for fine-tuning (cuda/cpu).')
    parser.add_argument('--config_path', type=str, default='config/training_config.template.yaml', help='Path to training config YAML file')
    parser.add_argument('--save_path', type=str, default='checkpoints/finetuned_pfas.pt', help='Path to save the fine-tuned model.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save a checkpoint every N steps.')
    args = parser.parse_args()

    print(f"Loading configuration from {args.config_path}...")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Pre-trained Model ---
    print(f"Loading pre-trained model from checkpoint: {args.ckpt}")
    mgnn_config = config.get('mgnn', {})
    model = DJMGNN(
        in_node_dim=29,
        in_edge_dim=mgnn_config.get('in_edge_dim', 0),
        node_output_dims=mgnn_config.get('node_output_dims', 3),
        graph_output_dims=mgnn_config.get('graph_output_dims', 19),
        energy_output_dims=mgnn_config.get('energy_output_dims', 1),
        hidden_dim=mgnn_config.get('hidden_channels', 128),
        n_blocks=mgnn_config.get('num_layers', 4)
    )
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # --- Load PFAS Dataset ---
    print("Loading PFAS dataset...")
    transform = Compose([
        CreateEdges(),
        FeaturizeNodes(),
        StandardizeTargets(dataset_name="pfas"),
    ])
    full_dataset = get_dataset("pfas", root="data", transform=transform)
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Fine-tuning Setup ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.L1Loss()  # Use MAE loss
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)

    # --- Fine-tuning Loop ---
    print(f"Starting fine-tuning for {args.max_steps} steps...")
    model.train()
    step = 0
    ema_loss = None
    beta = 0.95
    done = False
    best_val_loss = float('inf')
    patience_counter = 0

    while not done:
        for batch in train_loader:
            if step >= args.max_steps:
                done = True
                break
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch
            )
            
            preds = out['graph_pred']
            targets = batch.y.view(preds.shape)
            
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            # Update EMA loss
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss = beta * ema_loss + (1 - beta) * loss.item()
            
            if step % 20 == 0:
                print(f"Step {step:5d} | Loss: {loss.item():.6f} | EMA Loss: {ema_loss:.6f}")

            if step > 0 and step % args.save_every == 0:
                # Validation step
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device)
                        val_out = model(
                            x=val_batch.x,
                            edge_index=val_batch.edge_index,
                            batch=val_batch.batch
                        )
                        val_preds = val_out['graph_pred']
                        val_targets = val_batch.y.view(val_preds.shape)
                        val_loss = loss_fn(val_preds, val_targets)
                        val_losses.append(val_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"Validation Loss at step {step}: {avg_val_loss:.6f}")
                
                scheduler.step(avg_val_loss) # Update learning rate scheduler

                # Early stopping logic
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    intermediate_save_path = args.save_path.replace('.pt', f'_step_{step}_best.pt')
                    os.makedirs(os.path.dirname(intermediate_save_path), exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step,
                        'loss': loss.item(),
                        'val_loss': avg_val_loss,
                    }, intermediate_save_path)
                    print(f"Saved best intermediate checkpoint to {intermediate_save_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience: # Assuming args.patience is defined
                        print(f"Early stopping triggered at step {step} due to no improvement in validation loss.")
                        done = True
                        break
                
                model.train() # Set model back to training mode

            step += 1

    # --- Save Fine-tuned Model ---
    print(f"\nFine-tuning complete. Saving model to {args.save_path}")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, args.save_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()