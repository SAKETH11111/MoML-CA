#!/usr/bin/env python3
"""
Real-time Training Dashboard CLI

A beautiful terminal-based dashboard for monitoring training progress that works
perfectly over SSH and on servers without displays. Uses Rich for beautiful formatting.

Usage:
    python scripts/training_dashboard_cli.py --auto-detect production_checkpoints/
"""

import argparse
import time
import sys
from pathlib import Path
from typing import Optional, Tuple
import signal
from datetime import datetime
from collections import deque

import pandas as pd
import numpy as np

# Rich imports for beautiful terminal UI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.live import Live
from rich.layout import Layout
from rich.align import Align

class TrainingDashboardCLI:
    """Rich-based CLI training dashboard."""
    
    def __init__(self, metrics_csv: Path, phase_csv: Path, update_interval: float = 2.0):
        self.metrics_csv = metrics_csv
        self.phase_csv = phase_csv
        self.update_interval = update_interval
        
        # Console setup
        self.console = Console()
        
        # Data storage
        self.metrics_df = pd.DataFrame()
        self.phase_df = pd.DataFrame()
        self.last_metrics_size = 0
        self.last_phase_size = 0
        
        # History for trends
        self.loss_history = deque(maxlen=50)
        
        # Control
        self.is_running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Layout
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        self.layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        self.layout["left"].split_column(
            Layout(name="progress", size=8),
            Layout(name="metrics")
        )
        self.layout["right"].split_column(
            Layout(name="trends", size=12),
            Layout(name="stats")
        )
        
        print("🎯 Training Dashboard CLI Starting...")
        print(f"📊 Monitoring: {metrics_csv.name}")
        print(f"📈 Phase Data: {phase_csv.name}")
        print(f"🔄 Update Interval: {update_interval}s")
        print("Press Ctrl+C to stop\n")
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        self.console.print("\n🛑 Stopping dashboard...", style="bold red")
        self.is_running = False
        sys.exit(0)
    
    def load_data(self) -> bool:
        """Load data from CSV files. Returns True if new data found."""
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
                    
                    # Update loss history
                    if not self.metrics_df.empty:
                        self.loss_history.append(self.metrics_df['total_loss'].iloc[-1])
                        
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
    
    def create_header(self):
        """Create header panel."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = Text()
        header_text.append("🚀 DJMGNN Training Dashboard", style="bold cyan")
        header_text.append(f" - Live at {now}", style="dim")
        return Panel(Align.center(header_text), style="bold blue")
    
    def create_progress_panel(self):
        """Create progress panel."""
        if self.metrics_df.empty:
            return Panel("⏳ Waiting for training data...", title="Progress")
        
        latest = self.metrics_df.iloc[-1]
        step = int(latest['step'])
        max_steps = 40000  # Default
        
        # Progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        
        task = progress.add_task(f"Step {step:,}/{max_steps:,}", 
                                total=max_steps, completed=step)
        
        # Current phase info
        phase = latest['phase'].upper()
        phase_emoji = "🧬" if phase == "NODE" else "📈"
        
        progress_text = Text()
        progress_text.append(f"\nCurrent Phase: {phase_emoji} ", style="bold")
        progress_text.append(f"{phase}", style="bold cyan" if phase == "NODE" else "bold green")
        
        content = Columns([progress, progress_text])
        return Panel(content, title="📊 Training Progress", border_style="blue")
    
    def create_metrics_panel(self):
        """Create current metrics panel."""
        if self.metrics_df.empty:
            return Panel("No metrics data", title="Current Metrics")
        
        latest = self.metrics_df.iloc[-1]
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Total Loss", f"{latest['total_loss']:.6f}")
        table.add_row("Node Loss", f"{latest['node_loss']:.6f}")
        table.add_row("Graph Loss", f"{latest['graph_loss']:.6f}")
        table.add_row("Energy Loss", f"{latest['energy_loss']:.6f}")
        
        if 'learning_rate' in latest:
            table.add_row("Learning Rate", f"{latest['learning_rate']:.2e}")
        if 'steps_per_sec' in latest and latest['steps_per_sec'] > 0:
            table.add_row("Speed", f"{latest['steps_per_sec']:.2f} steps/sec")
        
        return Panel(table, title="📊 Current Metrics", border_style="green")
    
    def create_loss_trend_sparkline(self, data, width=40):
        """Create ASCII sparkline for loss trends."""
        if len(data) < 2:
            return "─" * width
        
        # Take last 'width' points
        recent_data = list(data)[-width:] if len(data) > width else list(data)
        
        if len(recent_data) == 0:
            return "─" * width
        
        # Normalize to 0-7 range for unicode block characters
        min_val, max_val = min(recent_data), max(recent_data)
        if max_val == min_val:
            return "─" * len(recent_data)
        
        normalized = [(val - min_val) / (max_val - min_val) * 7 for val in recent_data]
        
        # Unicode block characters for different heights
        blocks = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        
        sparkline = ''.join(blocks[min(int(val), 7)] for val in normalized)
        
        # Pad to desired width
        if len(sparkline) < width:
            sparkline = sparkline + '─' * (width - len(sparkline))
        
        return sparkline[:width]
    
    def create_trends_panel(self):
        """Create loss trends panel."""
        if self.metrics_df.empty:
            return Panel("No trend data", title="Loss Trends")
        
        table = Table(show_header=True, box=None)
        table.add_column("Loss Type", style="cyan", width=12)
        table.add_column("Trend (Last 40)", style="yellow", width=42)
        table.add_column("Current", style="green", width=10)
        
        # Get recent data for sparklines
        recent_data = self.metrics_df.tail(40)
        
        if not recent_data.empty:
            total_sparkline = self.create_loss_trend_sparkline(recent_data['total_loss'])
            node_sparkline = self.create_loss_trend_sparkline(recent_data['node_loss'])
            graph_sparkline = self.create_loss_trend_sparkline(recent_data['graph_loss'])
            energy_sparkline = self.create_loss_trend_sparkline(recent_data['energy_loss'])
            
            latest = recent_data.iloc[-1]
            
            table.add_row("Total", total_sparkline, f"{latest['total_loss']:.4f}")
            table.add_row("Node", node_sparkline, f"{latest['node_loss']:.4f}")
            table.add_row("Graph", graph_sparkline, f"{latest['graph_loss']:.4f}")
            table.add_row("Energy", energy_sparkline, f"{latest['energy_loss']:.4f}")
        
        return Panel(table, title="📈 Loss Trends", border_style="yellow")
    
    def create_stats_panel(self):
        """Create statistics panel."""
        if self.metrics_df.empty or len(self.metrics_df) < 10:
            return Panel("Insufficient data for stats", title="Statistics")
        
        # Recent statistics
        recent_loss = self.metrics_df['total_loss'].tail(100)
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Stat", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Best Loss", f"{recent_loss.min():.6f}")
        table.add_row("Worst Loss", f"{recent_loss.max():.6f}")
        table.add_row("Avg Loss", f"{recent_loss.mean():.6f}")
        table.add_row("Loss Std", f"{recent_loss.std():.6f}")
        
        # Phase distribution
        if not self.metrics_df.empty:
            node_steps = len(self.metrics_df[self.metrics_df['phase'] == 'node'])
            graph_steps = len(self.metrics_df[self.metrics_df['phase'] == 'graph'])
            total_steps = len(self.metrics_df)
            
            table.add_row("", "")  # Spacer
            table.add_row("NODE Steps", f"{node_steps:,} ({node_steps/total_steps*100:.1f}%)")
            table.add_row("GRAPH Steps", f"{graph_steps:,} ({graph_steps/total_steps*100:.1f}%)")
        
        # Trend analysis
        if len(recent_loss) > 20:
            recent_trend = recent_loss.tail(10).mean() - recent_loss.head(10).mean()
            trend_emoji = "📉" if recent_trend < 0 else "📈" if recent_trend > 0 else "➡️"
            trend_text = "Improving" if recent_trend < 0 else "Worsening" if recent_trend > 0 else "Stable"
            table.add_row("", "")  # Spacer
            table.add_row("Trend", f"{trend_emoji} {trend_text}")
        
        return Panel(table, title="📊 Statistics", border_style="magenta")
    
    def create_footer(self):
        """Create footer panel."""
        footer_text = Text()
        footer_text.append(f"🔄 Auto-updating every {self.update_interval}s", style="dim")
        footer_text.append(" • Press Ctrl+C to stop", style="dim red")
        
        # Add recent phase pattern if available
        if len(self.metrics_df) > 1:
            recent_phases = self.metrics_df['phase'].tail(20).values
            pattern = ''.join('N' if p == 'node' else 'G' for p in recent_phases)
            footer_text.append(f" • Recent: {pattern}", style="dim blue")
        
        return Panel(Align.center(footer_text), style="dim")
    
    def update_layout(self):
        """Update the layout with current data."""
        self.layout["header"].update(self.create_header())
        self.layout["progress"].update(self.create_progress_panel())
        self.layout["metrics"].update(self.create_metrics_panel())
        self.layout["trends"].update(self.create_trends_panel())
        self.layout["stats"].update(self.create_stats_panel())
        self.layout["footer"].update(self.create_footer())
    
    def start(self):
        """Start the CLI dashboard."""
        with Live(self.layout, console=self.console, refresh_per_second=2) as live:
            while self.is_running:
                try:
                    if self.load_data():
                        self.update_layout()
                    time.sleep(self.update_interval)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"Error: {e}", style="bold red")
                    time.sleep(self.update_interval)


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
        description="Real-time DJMGNN Training Dashboard CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect latest files in directory
    python scripts/training_dashboard_cli.py --auto-detect production_checkpoints/
    
    # Custom update interval
    python scripts/training_dashboard_cli.py --auto-detect production_checkpoints/ --update-interval 3.0
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--metrics-csv', type=Path, help='Path to training_metrics.csv file')
    group.add_argument('--auto-detect', type=Path, help='Directory to auto-detect latest CSV files')
    
    parser.add_argument('--phase-csv', type=Path, help='Path to phase_summary.csv file (required with --metrics-csv)')
    parser.add_argument('--update-interval', type=float, default=2.0, help='Update interval in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    # Determine CSV file paths
    if args.auto_detect:
        if not args.auto_detect.exists() or not args.auto_detect.is_dir():
            print(f"❌ Directory not found: {args.auto_detect}")
            sys.exit(1)
            
        print(f"🔍 Auto-detecting CSV files in {args.auto_detect}...")
        metrics_csv, phase_csv = find_latest_csv_files(args.auto_detect)
        
        if not metrics_csv:
            print("❌ No training_metrics_*.csv files found")
            sys.exit(1)
        if not phase_csv:
            print("❌ No phase_summary_*.csv files found")
            sys.exit(1)
            
        print(f"✅ Found metrics file: {metrics_csv.name}")
        print(f"✅ Found phase file: {phase_csv.name}")
        
    else:
        metrics_csv = args.metrics_csv
        phase_csv = args.phase_csv
        
        if not phase_csv:
            print("❌ --phase-csv is required when using --metrics-csv")
            sys.exit(1)
            
        if not metrics_csv.exists():
            print(f"❌ Metrics file not found: {metrics_csv}")
            sys.exit(1)
        if not phase_csv.exists():
            print(f"❌ Phase file not found: {phase_csv}")
            sys.exit(1)
    
    # Create and start dashboard
    try:
        dashboard = TrainingDashboardCLI(metrics_csv, phase_csv, args.update_interval)
        dashboard.start()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()