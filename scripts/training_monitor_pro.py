#!/usr/bin/env python3
"""
Professional ML Training Monitor CLI

A comprehensive real-time machine learning training monitoring system that:
- Displays live training metrics with visual charts and performance indicators
- Automatically detects and evaluates new checkpoints
- Provides clear good/bad status indicators and actionable recommendations
- Includes anomaly detection for training issues
- Shows resource utilization and predictive estimates
- All in a beautiful CLI interface

Usage:
    python scripts/training_monitor_pro.py --auto-detect production_checkpoints/
"""

import argparse
import time
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import signal
import threading
from datetime import datetime, timedelta
from collections import deque
import json
import subprocess
import psutil
import os

import pandas as pd
import numpy as np
import torch

# Rich imports for beautiful CLI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.rule import Rule
from rich.tree import Tree


class CheckpointEvaluator:
    """Automatically evaluate model checkpoints."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.evaluated_checkpoints = set()
        self.evaluation_history = deque(maxlen=50)
        
    def detect_new_checkpoints(self) -> List[Path]:
        """Detect new checkpoints that haven't been evaluated."""
        if not self.checkpoint_dir.exists():
            return []
            
        # Look for checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        checkpoint_files.extend(list(self.checkpoint_dir.glob("best_checkpoint.pt")))
        
        new_checkpoints = []
        for ckpt in checkpoint_files:
            if ckpt not in self.evaluated_checkpoints:
                new_checkpoints.append(ckpt)
                
        return new_checkpoints
    
    def evaluate_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Evaluate a checkpoint and return performance metrics."""
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract metrics
            step = checkpoint.get('step', 0)
            loss = checkpoint.get('loss', float('inf'))
            timestamp = checkpoint.get('timestamp', time.time())
            
            # Simple health check
            model_state = checkpoint.get('model_state_dict', {})
            
            # Check for NaN/Inf in model parameters
            has_nan = False
            param_stats = {"min": float('inf'), "max": float('-inf'), "mean": 0, "std": 0}
            
            if model_state:
                all_params = []
                for name, param in model_state.items():
                    if isinstance(param, torch.Tensor):
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            has_nan = True
                        all_params.extend(param.flatten().tolist())
                
                if all_params and not has_nan:
                    all_params = np.array(all_params)
                    param_stats = {
                        "min": float(all_params.min()),
                        "max": float(all_params.max()),
                        "mean": float(all_params.mean()),
                        "std": float(all_params.std())
                    }
            
            # Determine health status
            health_score = self._calculate_health_score(loss, has_nan, param_stats)
            status = "GOOD" if health_score > 0.7 else "WARNING" if health_score > 0.4 else "BAD"
            
            evaluation = {
                "checkpoint": checkpoint_path.name,
                "step": step,
                "loss": loss,
                "timestamp": timestamp,
                "health_score": health_score,
                "status": status,
                "has_nan": has_nan,
                "param_stats": param_stats,
                "evaluated_at": time.time()
            }
            
            # Add to history
            self.evaluation_history.append(evaluation)
            self.evaluated_checkpoints.add(checkpoint_path)
            
            return evaluation
            
        except Exception as e:
            return {
                "checkpoint": checkpoint_path.name,
                "step": 0,
                "loss": float('inf'),
                "status": "ERROR",
                "error": str(e),
                "evaluated_at": time.time()
            }
    
    def _calculate_health_score(self, loss: float, has_nan: bool, param_stats: Dict) -> float:
        """Calculate a health score from 0-1 based on various indicators."""
        if has_nan or loss == float('inf'):
            return 0.0
        
        score = 1.0
        
        # Penalize very high losses
        if loss > 10:
            score *= 0.3
        elif loss > 5:
            score *= 0.5
        elif loss > 1:
            score *= 0.8
        
        # Check parameter statistics
        if param_stats["std"] == 0:  # All parameters the same
            score *= 0.2
        elif param_stats["std"] > 10:  # Very high variance
            score *= 0.6
        elif abs(param_stats["mean"]) > 5:  # Very high mean
            score *= 0.7
            
        return max(0.0, min(1.0, score))


class AnomalyDetector:
    """Detect training anomalies and issues."""
    
    def __init__(self):
        self.loss_history = deque(maxlen=1000)
        self.lr_history = deque(maxlen=1000)
        self.speed_history = deque(maxlen=100)
        
    def update(self, metrics: Dict[str, float]):
        """Update with new metrics."""
        if 'total_loss' in metrics:
            self.loss_history.append(metrics['total_loss'])
        if 'learning_rate' in metrics:
            self.lr_history.append(metrics['learning_rate'])
        if 'steps_per_sec' in metrics and metrics['steps_per_sec'] > 0:
            self.speed_history.append(metrics['steps_per_sec'])
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect various training anomalies."""
        anomalies = []
        
        # Check for gradient explosion
        if len(self.loss_history) >= 10:
            recent_losses = list(self.loss_history)[-10:]
            if len(recent_losses) >= 2:
                # Check for sudden spikes
                max_recent = max(recent_losses)
                avg_before = np.mean(list(self.loss_history)[-20:-10]) if len(self.loss_history) >= 20 else max_recent
                
                if max_recent > avg_before * 5:  # 5x increase
                    anomalies.append({
                        "type": "GRADIENT_EXPLOSION",
                        "severity": "HIGH",
                        "message": f"Loss spiked from {avg_before:.4f} to {max_recent:.4f}",
                        "recommendation": "Consider reducing learning rate or gradient clipping"
                    })
        
        # Check for loss plateau
        if len(self.loss_history) >= 100:
            recent_100 = list(self.loss_history)[-100:]
            if np.std(recent_100) < np.mean(recent_100) * 0.01:  # Very low variance
                anomalies.append({
                    "type": "LOSS_PLATEAU",
                    "severity": "MEDIUM",
                    "message": f"Loss plateaued at {np.mean(recent_100):.4f} for 100+ steps",
                    "recommendation": "Consider learning rate scheduling or architecture changes"
                })
        
        # Check for oscillating loss
        if len(self.loss_history) >= 50:
            recent_50 = np.array(list(self.loss_history)[-50:])
            # Simple oscillation detection
            diff = np.diff(recent_50)
            sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
            if sign_changes > 30:  # Too many direction changes
                anomalies.append({
                    "type": "OSCILLATING_LOSS",
                    "severity": "MEDIUM",
                    "message": "Loss oscillating frequently",
                    "recommendation": "Learning rate might be too high"
                })
        
        # Check training speed
        if len(self.speed_history) >= 10:
            recent_speed = list(self.speed_history)[-10:]
            avg_speed = np.mean(recent_speed)
            if avg_speed < 0.5:  # Very slow
                anomalies.append({
                    "type": "SLOW_TRAINING",
                    "severity": "LOW",
                    "message": f"Training speed very low: {avg_speed:.2f} steps/sec",
                    "recommendation": "Check batch size, hardware utilization"
                })
        
        return anomalies


class PerformanceAssessor:
    """Assess overall training performance and provide recommendations."""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        
    def update(self, metrics: Dict[str, float]):
        """Update with new metrics."""
        metrics_with_timestamp = metrics.copy()
        metrics_with_timestamp['timestamp'] = time.time()
        self.metrics_history.append(metrics_with_timestamp)
    
    def assess_training(self) -> Dict[str, Any]:
        """Assess overall training performance."""
        if len(self.metrics_history) < 10:
            return {"status": "INSUFFICIENT_DATA", "score": 0.0}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 points
        losses = [m['total_loss'] for m in recent_metrics if 'total_loss' in m]
        
        if not losses:
            return {"status": "NO_LOSS_DATA", "score": 0.0}
        
        # Calculate improvement trend
        if len(losses) >= 20:
            early_avg = np.mean(losses[:10])
            recent_avg = np.mean(losses[-10:])
            improvement = (early_avg - recent_avg) / early_avg if early_avg > 0 else 0
        else:
            improvement = 0
        
        # Calculate stability
        stability = 1.0 / (1.0 + np.std(losses[-20:]) / np.mean(losses[-20:])) if len(losses) >= 20 else 0.5
        
        # Calculate convergence rate
        if len(losses) >= 50:
            # Simple convergence check - are we still improving?
            recent_trend = np.mean(losses[-10:]) - np.mean(losses[-20:-10])
            convergence = max(0, -recent_trend / np.mean(losses[-20:-10])) if np.mean(losses[-20:-10]) > 0 else 0
        else:
            convergence = 0.5
        
        # Overall score
        score = (improvement * 0.4 + stability * 0.3 + convergence * 0.3)
        score = max(0.0, min(1.0, score))
        
        # Determine status
        if score > 0.8:
            status = "EXCELLENT"
        elif score > 0.6:
            status = "GOOD"
        elif score > 0.4:
            status = "AVERAGE"
        elif score > 0.2:
            status = "POOR"
        else:
            status = "CRITICAL"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(improvement, stability, convergence, losses)
        
        return {
            "status": status,
            "score": score,
            "improvement": improvement,
            "stability": stability,
            "convergence": convergence,
            "current_loss": losses[-1] if losses else 0,
            "best_loss": min(losses) if losses else float('inf'),
            "recommendations": recommendations
        }
    
    def _generate_recommendations(self, improvement: float, stability: float, convergence: float, losses: List[float]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if improvement < 0.1:
            recommendations.append("⚠️  Loss not improving - consider learning rate decay or different optimizer")
        
        if stability < 0.5:
            recommendations.append("⚠️  Training unstable - try gradient clipping or smaller learning rate")
        
        if convergence < 0.3:
            recommendations.append("⚠️  Slow convergence - experiment with learning rate scheduling")
        
        if len(losses) >= 20:
            recent_losses = losses[-20:]
            if max(recent_losses) > min(recent_losses) * 3:
                recommendations.append("⚠️  High loss variance - check batch size and data quality")
        
        if not recommendations:
            recommendations.append("✅ Training looks healthy - continue current settings")
        
        return recommendations


class SystemMonitor:
    """Monitor system resources."""
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available // (1024**3)  # GB
            
            # GPU usage (if available)
            gpu_stats = self._get_gpu_stats()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free // (1024**3)  # GB
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_gb": memory_available,
                "disk_percent": disk_percent,
                "disk_free_gb": disk_free,
                "gpu_stats": gpu_stats
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics if available."""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for i, line in enumerate(lines):
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 4:
                        gpus.append({
                            "id": i,
                            "utilization": int(parts[0]),
                            "memory_used": int(parts[1]),
                            "memory_total": int(parts[2]),
                            "temperature": int(parts[3])
                        })
                return {"gpus": gpus, "available": True}
        except Exception:
            pass
        
        return {"available": False}


class TrainingMonitorPro:
    """Professional ML training monitoring system."""
    
    def __init__(self, metrics_csv: Path, phase_csv: Path, checkpoint_dir: Path, update_interval: float = 5.0):
        self.console = Console()
        self.metrics_csv = metrics_csv
        self.phase_csv = phase_csv
        self.checkpoint_dir = checkpoint_dir
        self.update_interval = update_interval
        
        # Components
        self.checkpoint_evaluator = CheckpointEvaluator(checkpoint_dir)
        self.anomaly_detector = AnomalyDetector()
        self.performance_assessor = PerformanceAssessor()
        self.system_monitor = SystemMonitor()
        
        # Data storage
        self.metrics_df = pd.DataFrame()
        self.phase_df = pd.DataFrame()
        self.last_metrics_size = 0
        self.last_phase_size = 0
        
        # State tracking
        self.alerts = deque(maxlen=20)
        self.latest_evaluation = None
        self.training_start_time = time.time()
        
        # Layout
        self.layout = Layout()
        self._setup_layout()
        
        # Control
        self.is_running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print("🎯 Professional ML Training Monitor Starting...")
        print(f"📊 Monitoring: {metrics_csv.name}")
        print(f"📈 Phase Data: {phase_csv.name}")
        print(f"📁 Checkpoints: {checkpoint_dir}")
        print(f"🔄 Update Interval: {update_interval}s")
        print("Press Ctrl+C to stop\n")
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        self.console.print("\n🛑 Stopping monitor...", style="bold red")
        self.is_running = False
        sys.exit(0)
    
    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=4)
        )
        
        self.layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        self.layout["left"].split_column(
            Layout(name="status", size=8),
            Layout(name="metrics", size=12),
            Layout(name="checkpoints")
        )
        
        self.layout["right"].split_column(
            Layout(name="performance", size=10),
            Layout(name="anomalies", size=8),
            Layout(name="system", size=8)
        )
    
    def load_data(self) -> bool:
        """Load data from CSV files."""
        try:
            # Check if files have new data
            metrics_size = self.metrics_csv.stat().st_size if self.metrics_csv.exists() else 0
            phase_size = self.phase_csv.stat().st_size if self.phase_csv.exists() else 0
            
            has_new_data = (metrics_size != self.last_metrics_size or 
                           phase_size != self.last_phase_size)
            
            if not has_new_data and not self.metrics_df.empty:
                return False
                
            # Load metrics data
            if self.metrics_csv.exists() and metrics_size > 0:
                try:
                    self.metrics_df = pd.read_csv(self.metrics_csv)
                    if 'timestamp' in self.metrics_df.columns:
                        self.metrics_df['datetime'] = pd.to_datetime(self.metrics_df['timestamp'], unit='s')
                    self.last_metrics_size = metrics_size
                    
                    # Update components with latest data
                    if not self.metrics_df.empty:
                        latest = self.metrics_df.iloc[-1].to_dict()
                        self.anomaly_detector.update(latest)
                        self.performance_assessor.update(latest)
                        
                except Exception as e:
                    self.console.print(f"Warning: Could not load metrics CSV: {e}", style="yellow")
                    return False
                    
            # Load phase data
            if self.phase_csv.exists() and phase_size > 0:
                try:
                    self.phase_df = pd.read_csv(self.phase_csv)
                    if 'timestamp' in self.phase_df.columns:
                        self.phase_df['datetime'] = pd.to_datetime(self.phase_df['timestamp'], unit='s')
                    self.last_phase_size = phase_size
                except Exception as e:
                    self.console.print(f"Warning: Could not load phase CSV: {e}", style="yellow")
                    
            return has_new_data
            
        except Exception as e:
            self.console.print(f"Error loading data: {e}", style="bold red")
            return False
    
    def check_new_checkpoints(self):
        """Check for and evaluate new checkpoints."""
        new_checkpoints = self.checkpoint_evaluator.detect_new_checkpoints()
        
        for checkpoint in new_checkpoints:
            self.console.print(f"🔍 Evaluating new checkpoint: {checkpoint.name}", style="cyan")
            evaluation = self.checkpoint_evaluator.evaluate_checkpoint(checkpoint)
            self.latest_evaluation = evaluation
            
            # Add alert for new checkpoint
            status_color = "green" if evaluation["status"] == "GOOD" else "yellow" if evaluation["status"] == "WARNING" else "red"
            self.alerts.append({
                "timestamp": time.time(),
                "type": "CHECKPOINT",
                "message": f"New checkpoint: {checkpoint.name} - Status: {evaluation['status']}",
                "color": status_color
            })
    
    def create_header(self):
        """Create header panel."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.training_start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        header_text = Text()
        header_text.append("🎯 Professional ML Training Monitor", style="bold cyan")
        header_text.append(f" - {now} - Runtime: {elapsed_str}", style="dim")
        return Panel(Align.center(header_text), style="bold blue")
    
    def create_status_panel(self):
        """Create training status panel."""
        if self.metrics_df.empty:
            return Panel("⏳ Waiting for training data...", title="Training Status")
        
        latest = self.metrics_df.iloc[-1]
        step = int(latest['step'])
        phase = latest['phase'].upper()
        phase_emoji = "🧬" if phase == "NODE" else "📈"
        
        # Get performance assessment
        assessment = self.performance_assessor.assess_training()
        
        # Status color based on assessment
        status_colors = {
            "EXCELLENT": "bright_green",
            "GOOD": "green", 
            "AVERAGE": "yellow",
            "POOR": "red",
            "CRITICAL": "bright_red"
        }
        status_color = status_colors.get(assessment["status"], "white")
        
        status_text = Text()
        status_text.append("🎯 TRAINING STATUS\n\n", style="bold")
        status_text.append(f"Step: {step:,}\n", style="cyan")
        status_text.append(f"Phase: {phase_emoji} {phase}\n", style="bold cyan" if phase == "NODE" else "bold green")
        status_text.append(f"Status: ", style="white")
        status_text.append(f"{assessment['status']}\n", style=f"bold {status_color}")
        status_text.append(f"Score: {assessment['score']:.2f}/1.0\n", style="yellow")
        status_text.append(f"Current Loss: {latest['total_loss']:.6f}\n", style="magenta")
        if 'best_loss' in assessment:
            status_text.append(f"Best Loss: {assessment['best_loss']:.6f}", style="bright_green")
        
        return Panel(status_text, title=f"🎯 Status", border_style=status_color)
    
    def create_metrics_panel(self):
        """Create detailed metrics panel."""
        if self.metrics_df.empty:
            return Panel("No metrics data", title="Metrics")
        
        latest = self.metrics_df.iloc[-1]
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan", width=14)
        table.add_column("Current", style="yellow", width=12)
        table.add_column("Trend", style="blue", width=8)
        
        # Add metrics with trend indicators
        metrics_to_show = [
            ("Total Loss", "total_loss"),
            ("Node Loss", "node_loss"),
            ("Graph Loss", "graph_loss"),
            ("Energy Loss", "energy_loss")
        ]
        
        for name, col in metrics_to_show:
            if col in latest:
                current = latest[col]
                # Simple trend calculation
                if len(self.metrics_df) >= 10:
                    recent = self.metrics_df[col].tail(10)
                    trend = "📉" if recent.iloc[-1] < recent.iloc[0] else "📈" if recent.iloc[-1] > recent.iloc[0] else "➡️"
                else:
                    trend = "➡️"
                
                table.add_row(name, f"{current:.6f}", trend)
        
        # Add other metrics
        if 'learning_rate' in latest:
            table.add_row("Learning Rate", f"{latest['learning_rate']:.2e}", "")
        if 'steps_per_sec' in latest and latest['steps_per_sec'] > 0:
            table.add_row("Speed", f"{latest['steps_per_sec']:.2f} steps/s", "")
        
        return Panel(table, title="📊 Current Metrics", border_style="green")
    
    def create_checkpoints_panel(self):
        """Create checkpoint evaluation panel."""
        if not self.checkpoint_evaluator.evaluation_history:
            return Panel("No checkpoints evaluated yet", title="Checkpoints")
        
        table = Table(show_header=True, box=None)
        table.add_column("Checkpoint", style="cyan", width=20)
        table.add_column("Status", style="bold", width=8)
        table.add_column("Loss", style="yellow", width=10)
        table.add_column("Score", style="green", width=6)
        
        # Show last 5 evaluations
        recent_evals = list(self.checkpoint_evaluator.evaluation_history)[-5:]
        for eval_data in recent_evals:
            status_style = "bright_green" if eval_data["status"] == "GOOD" else "yellow" if eval_data["status"] == "WARNING" else "red"
            table.add_row(
                eval_data["checkpoint"][:18] + "..." if len(eval_data["checkpoint"]) > 20 else eval_data["checkpoint"],
                eval_data["status"],
                f"{eval_data.get('loss', 0):.4f}",
                f"{eval_data.get('health_score', 0):.2f}"
            )
        
        return Panel(table, title="📁 Recent Checkpoints", border_style="blue")
    
    def create_performance_panel(self):
        """Create performance assessment panel."""
        assessment = self.performance_assessor.assess_training()
        
        # Create performance bars
        performance_text = Text()
        performance_text.append("🎯 PERFORMANCE ANALYSIS\n\n", style="bold")
        
        metrics = [
            ("Improvement", assessment.get("improvement", 0)),
            ("Stability", assessment.get("stability", 0)), 
            ("Convergence", assessment.get("convergence", 0))
        ]
        
        for name, value in metrics:
            bar_length = int(value * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            color = "green" if value > 0.7 else "yellow" if value > 0.4 else "red"
            performance_text.append(f"{name:12} │{bar}│ {value:.2f}\n", style=color)
        
        performance_text.append("\n📋 RECOMMENDATIONS:\n", style="bold cyan")
        recommendations = assessment.get("recommendations", ["No recommendations"])
        for rec in recommendations[:3]:  # Show top 3
            performance_text.append(f"• {rec}\n", style="dim")
        
        return Panel(performance_text, title="🎯 Performance", border_style="magenta")
    
    def create_anomalies_panel(self):
        """Create anomalies detection panel."""
        anomalies = self.anomaly_detector.detect_anomalies()
        
        if not anomalies:
            return Panel("✅ No anomalies detected", title="🔍 Anomaly Detection", border_style="green")
        
        anomaly_text = Text()
        anomaly_text.append("⚠️  DETECTED ISSUES:\n\n", style="bold red")
        
        for anomaly in anomalies[-5:]:  # Show last 5
            severity_color = "red" if anomaly["severity"] == "HIGH" else "yellow" if anomaly["severity"] == "MEDIUM" else "blue"
            anomaly_text.append(f"• {anomaly['type']}\n", style=f"bold {severity_color}")
            anomaly_text.append(f"  {anomaly['message']}\n", style="dim")
            anomaly_text.append(f"  💡 {anomaly['recommendation']}\n\n", style="dim cyan")
        
        return Panel(anomaly_text, title="🔍 Anomaly Detection", border_style="red")
    
    def create_system_panel(self):
        """Create system monitoring panel."""
        stats = self.system_monitor.get_system_stats()
        
        if "error" in stats:
            return Panel(f"❌ System monitoring error: {stats['error']}", title="System")
        
        system_text = Text()
        system_text.append("🖥️  SYSTEM RESOURCES\n\n", style="bold")
        
        # CPU
        cpu_color = "red" if stats.get("cpu_percent", 0) > 90 else "yellow" if stats.get("cpu_percent", 0) > 70 else "green"
        system_text.append(f"CPU: {stats.get('cpu_percent', 0):.1f}%\n", style=cpu_color)
        
        # Memory
        mem_color = "red" if stats.get("memory_percent", 0) > 90 else "yellow" if stats.get("memory_percent", 0) > 70 else "green"
        system_text.append(f"Memory: {stats.get('memory_percent', 0):.1f}% ({stats.get('memory_available_gb', 0):.1f}GB free)\n", style=mem_color)
        
        # Disk
        disk_color = "red" if stats.get("disk_percent", 0) > 95 else "yellow" if stats.get("disk_percent", 0) > 80 else "green"
        system_text.append(f"Disk: {stats.get('disk_percent', 0):.1f}% ({stats.get('disk_free_gb', 0):.1f}GB free)\n", style=disk_color)
        
        # GPU if available
        gpu_stats = stats.get("gpu_stats", {})
        if gpu_stats.get("available"):
            system_text.append("\n🎮 GPU:\n", style="bold blue")
            for i, gpu in enumerate(gpu_stats.get("gpus", [])):
                gpu_color = "red" if gpu["utilization"] > 95 else "green" if gpu["utilization"] > 50 else "yellow"
                system_text.append(f"  GPU{i}: {gpu['utilization']}% ({gpu['memory_used']}/{gpu['memory_total']}MB)\n", style=gpu_color)
        
        return Panel(system_text, title="🖥️  System", border_style="blue")
    
    def create_footer(self):
        """Create footer with alerts and status."""
        footer_text = Text()
        
        # Recent alerts
        if self.alerts:
            recent_alerts = list(self.alerts)[-3:]  # Last 3 alerts
            footer_text.append("🚨 Recent Alerts: ", style="bold")
            for alert in recent_alerts:
                alert_time = datetime.fromtimestamp(alert["timestamp"]).strftime("%H:%M:%S")
                footer_text.append(f"[{alert_time}] {alert['message']} ", style=alert["color"])
        else:
            footer_text.append("✅ No alerts", style="green")
        
        footer_text.append(f" • 🔄 Next update in {self.update_interval}s • Press Ctrl+C to stop", style="dim")
        
        return Panel(Align.center(footer_text), style="dim")
    
    def update_layout(self):
        """Update the layout with current data."""
        try:
            self.layout["header"].update(self.create_header())
            self.layout["status"].update(self.create_status_panel())
            self.layout["metrics"].update(self.create_metrics_panel())
            self.layout["checkpoints"].update(self.create_checkpoints_panel())
            self.layout["performance"].update(self.create_performance_panel())
            self.layout["anomalies"].update(self.create_anomalies_panel())
            self.layout["system"].update(self.create_system_panel())
            self.layout["footer"].update(self.create_footer())
        except Exception as e:
            # Fallback error display
            error_panel = Panel(f"❌ Layout update error: {e}", style="red")
            self.layout["header"].update(error_panel)
    
    def start(self):
        """Start the professional training monitor."""
        with Live(self.layout, console=self.console, refresh_per_second=1) as live:
            while self.is_running:
                try:
                    # Load new training data
                    has_new_data = self.load_data()
                    
                    # Check for new checkpoints
                    self.check_new_checkpoints()
                    
                    # Update display
                    self.update_layout()
                    
                    # Sleep until next update
                    time.sleep(self.update_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"Monitor error: {e}", style="red")
                    time.sleep(1)


def find_latest_csv_files(directory: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the most recent training CSV files in a directory."""
    metrics_files = list(directory.glob("training_metrics_*.csv"))
    phase_files = list(directory.glob("phase_summary_*.csv"))
    
    metrics_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    phase_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    metrics_csv = metrics_files[0] if metrics_files else None
    phase_csv = phase_files[0] if phase_files else None
    
    return metrics_csv, phase_csv


def main():
    parser = argparse.ArgumentParser(
        description="Professional ML Training Monitor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect files and start monitoring
    python scripts/training_monitor_pro.py --auto-detect production_checkpoints/
    
    # Custom update interval
    python scripts/training_monitor_pro.py --auto-detect production_checkpoints/ --update-interval 3.0
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--metrics-csv', type=Path, help='Path to training_metrics.csv file')
    group.add_argument('--auto-detect', type=Path, help='Directory to auto-detect latest CSV files and checkpoints')
    
    parser.add_argument('--phase-csv', type=Path, help='Path to phase_summary.csv file (required with --metrics-csv)')
    parser.add_argument('--checkpoint-dir', type=Path, help='Directory containing model checkpoints (required with --metrics-csv)')
    parser.add_argument('--update-interval', type=float, default=5.0, help='Update interval in seconds (default: 5.0)')
    
    args = parser.parse_args()
    
    # Determine paths
    if args.auto_detect:
        if not args.auto_detect.exists() or not args.auto_detect.is_dir():
            print(f"❌ Directory not found: {args.auto_detect}")
            sys.exit(1)
            
        print(f"🔍 Auto-detecting files in {args.auto_detect}...")
        metrics_csv, phase_csv = find_latest_csv_files(args.auto_detect)
        checkpoint_dir = args.auto_detect  # Same directory
        
        if not metrics_csv:
            print("❌ No training_metrics_*.csv files found")
            sys.exit(1)
        if not phase_csv:
            print("❌ No phase_summary_*.csv files found")
            sys.exit(1)
            
        print(f"✅ Found metrics file: {metrics_csv.name}")
        print(f"✅ Found phase file: {phase_csv.name}")
        print(f"✅ Monitoring checkpoints in: {checkpoint_dir}")
        
    else:
        metrics_csv = args.metrics_csv
        phase_csv = args.phase_csv
        checkpoint_dir = args.checkpoint_dir
        
        if not phase_csv or not checkpoint_dir:
            print("❌ --phase-csv and --checkpoint-dir are required when using --metrics-csv")
            sys.exit(1)
            
        for path, name in [(metrics_csv, "metrics"), (phase_csv, "phase"), (checkpoint_dir, "checkpoint_dir")]:
            if not path.exists():
                print(f"❌ {name} path not found: {path}")
                sys.exit(1)
    
    # Create and start monitor
    try:
        monitor = TrainingMonitorPro(metrics_csv, phase_csv, checkpoint_dir, args.update_interval)
        monitor.start()
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped by user")
    except Exception as e:
        print(f"❌ Monitor error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()