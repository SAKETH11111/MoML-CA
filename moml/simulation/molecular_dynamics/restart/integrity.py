"""
Integrity verification module for ensuring data consistency.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import hashlib
import zlib
import structlog
import git
import xml.etree.ElementTree as ET
import json
from datetime import datetime

logger = structlog.get_logger()

class IntegrityVerifier:
    """Verifies integrity of simulation data and checkpoints."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def verify_system(self, system_xml: Path) -> Tuple[bool, Optional[str]]:
        """Verify system XML integrity."""
        try:
            # Parse XML
            tree = ET.parse(system_xml)
            root = tree.getroot()
            
            # Compute CRC-32
            with open(system_xml, 'rb') as f:
                data = f.read()
                crc32 = zlib.crc32(data)
            
            # Verify XML structure
            if not self._verify_xml_structure(root):
                return False, "Invalid XML structure"
            
            return True, f"{crc32:08x}"
            
        except Exception as e:
            logger.error("system_verification_failed",
                        path=str(system_xml),
                        error=str(e))
            return False, str(e)
    
    def verify_trajectory(self, trajectory_path: Path) -> Tuple[bool, Optional[str]]:
        """Verify trajectory file integrity."""
        try:
            # Compute MD5 hash
            md5_hash = hashlib.md5()
            with open(trajectory_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    md5_hash.update(chunk)
            
            return True, md5_hash.hexdigest()
            
        except Exception as e:
            logger.error("trajectory_verification_failed",
                        path=str(trajectory_path),
                        error=str(e))
            return False, str(e)
    
    def verify_checkpoint(self, checkpoint_path: Path) -> Tuple[bool, Optional[str]]:
        """Verify checkpoint file integrity."""
        try:
            # Compute SHA-256 hash
            sha256_hash = hashlib.sha256()
            with open(checkpoint_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(chunk)
            
            return True, sha256_hash.hexdigest()
            
        except Exception as e:
            logger.error("checkpoint_verification_failed",
                        path=str(checkpoint_path),
                        error=str(e))
            return False, str(e)
    
    def verify_git_state(self) -> Tuple[bool, Optional[str]]:
        """Verify git repository state."""
        try:
            repo = git.Repo(search_parent_directories=True)
            
            # Check for uncommitted changes
            if repo.is_dirty():
                return False, "Uncommitted changes detected"
            
            # Get current revision
            revision = repo.head.object.hexsha
            
            return True, revision
            
        except git.InvalidGitRepositoryError:
            return False, "Not a git repository"
        except Exception as e:
            logger.error("git_verification_failed", error=str(e))
            return False, str(e)
    
    def _verify_xml_structure(self, root: ET.Element) -> bool:
        """Verify XML structure is valid."""
        required_sections = {
            'HarmonicBondForce',
            'HarmonicAngleForce',
            'PeriodicTorsionForce',
            'NonbondedForce'
        }
        
        # Check required sections
        for section in required_sections:
            if root.find(section) is None:
                logger.error("missing_xml_section", section=section)
                return False
        
        return True
    
    def verify_run_consistency(self, run_dir: Path) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Verify consistency of an entire run directory."""
        results = {}
        
        # Verify system XML
        system_xml = run_dir / "system.xml"
        if system_xml.exists():
            results["system_xml"] = self.verify_system(system_xml)
        
        # Verify trajectory
        trajectory_path = run_dir / "trajectory.dcd"
        if trajectory_path.exists():
            results["trajectory"] = self.verify_trajectory(trajectory_path)
        
        # Verify checkpoints
        checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir.exists():
            for checkpoint in checkpoint_dir.glob("*.chk"):
                results[f"checkpoint_{checkpoint.name}"] = self.verify_checkpoint(checkpoint)
        
        # Verify git state
        results["git_state"] = self.verify_git_state()
        
        return results
    
    def generate_integrity_report(self, run_dir: Path) -> Dict:
        """Generate a comprehensive integrity report."""
        results = self.verify_run_consistency(run_dir)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "run_directory": str(run_dir),
            "verification_results": results,
            "overall_status": all(status for status, _ in results.values())
        }
        
        # Save report
        report_path = run_dir / "integrity_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("integrity_report_generated",
                   path=str(report_path),
                   status=report["overall_status"])
        
        return report 