"""
Checkpoint manager for handling simulation restarts and integrity verification.
"""

from pathlib import Path
from typing import Optional, Dict, List
import hashlib
import json
import structlog
import git
from datetime import datetime

logger = structlog.get_logger()

class CheckpointManager:
    """Manages simulation checkpoints and restarts."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint metadata
        self.metadata_path = self.checkpoint_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata."""
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                return json.load(f)
        return {
            "checkpoints": [],
            "last_checkpoint": None,
            "git_revision": self._get_git_revision()
        }
    
    def _get_git_revision(self) -> Optional[str]:
        """Get current git revision."""
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha
        except git.InvalidGitRepositoryError:
            return None
    
    def save_checkpoint(self,
                       checkpoint_data: bytes,
                       step: int,
                       system_hash: str) -> Path:
        """Save a new checkpoint."""
        # Generate checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{step}_{timestamp}_{system_hash[:8]}.chk"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            f.write(checkpoint_data)
        
        # Update metadata
        checkpoint_info = {
            "path": str(checkpoint_path),
            "step": step,
            "timestamp": timestamp,
            "system_hash": system_hash,
            "checksum": self._compute_checksum(checkpoint_data)
        }
        
        self.metadata["checkpoints"].append(checkpoint_info)
        self.metadata["last_checkpoint"] = checkpoint_info
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info("checkpoint_saved",
                   path=str(checkpoint_path),
                   step=step,
                   system_hash=system_hash)
        
        return checkpoint_path
    
    def load_latest_checkpoint(self) -> Optional[bytes]:
        """Load the latest checkpoint."""
        if not self.metadata["last_checkpoint"]:
            return None
        
        checkpoint_info = self.metadata["last_checkpoint"]
        checkpoint_path = Path(checkpoint_info["path"])
        
        if not checkpoint_path.exists():
            logger.error("checkpoint_not_found", path=str(checkpoint_path))
            return None
        
        # Verify checkpoint
        if not self.verify_checkpoint(checkpoint_path, checkpoint_info["checksum"]):
            logger.error("checkpoint_verification_failed", path=str(checkpoint_path))
            return None
        
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = f.read()
        
        logger.info("checkpoint_loaded",
                   path=str(checkpoint_path),
                   step=checkpoint_info["step"])
        
        return checkpoint_data
    
    def verify_checkpoint(self, checkpoint_path: Path, expected_checksum: str) -> bool:
        """Verify checkpoint integrity."""
        try:
            with open(checkpoint_path, 'rb') as f:
                data = f.read()
                checksum = self._compute_checksum(data)
                return checksum == expected_checksum
        except Exception as e:
            logger.error("checkpoint_verification_error",
                        path=str(checkpoint_path),
                        error=str(e))
            return False
    
    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()
    
    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints."""
        return self.metadata["checkpoints"]
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = self.metadata["checkpoints"]
        if len(checkpoints) <= keep_last:
            return
        
        # Sort checkpoints by step
        checkpoints.sort(key=lambda x: x["step"])
        
        # Remove old checkpoints
        for checkpoint in checkpoints[:-keep_last]:
            path = Path(checkpoint["path"])
            if path.exists():
                path.unlink()
                logger.info("checkpoint_removed", path=str(path))
        
        # Update metadata
        self.metadata["checkpoints"] = checkpoints[-keep_last:]
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2) 