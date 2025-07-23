#!/usr/bin/env python3
"""
Real-time Training Dashboard CLI

A beautiful real-time dashboard that monitors training progress by reading CSV logs
and displaying live updating graphs of losses, learning rate, phase transitions, and more.

Usage:
    python scripts/training_dashboard.py --metrics-csv path/to/training_metrics.csv --phase-csv path/to/phase_summary.csv
    python scripts/training_dashboard.py --auto-detect production_checkpoints/  # Auto-find latest CSV files
"""

import argparse
import time
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import signal

import pandas as pd
import numpy as np

# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
import seaborn as sns
from datetime import datetime

# Set up matplotlib for better appearance
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    # Fallback if seaborn style not available
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

sns.set_palette("husl")

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

class TrainingDashboard:
    """Real-time training dashboard with live updating graphs."""
    
    def __init__(self, metrics_csv: Path, phase_csv: Path, update_interval: float = 2.0):
        self.metrics_csv = metrics_csv
        self.phase_csv = phase_csv
        self.update_interval = update_interval
        
        # Data storage
        self.metrics_df = pd.DataFrame()
        self.phase_df = pd.DataFrame()
        self.last_metrics_size = 0
        self.last_phase_size = 0
        
        # Setup matplotlib figure
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('🚀 DJMGNN Training Dashboard - Real-Time Monitoring', 
                         fontsize=16, fontweight='bold')
        
        # Create subplots with GridSpec for better layout
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Main loss plot (spans 2 columns)
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        self.ax_loss_components = self.fig.add_subplot(gs[0, 2:])
        
        # Phase and learning rate plots
        self.ax_phase = self.fig.add_subplot(gs[1, :2])
        self.ax_lr = self.fig.add_subplot(gs[1, 2])
        self.ax_speed = self.fig.add_subplot(gs[1, 3])
        
        # Advanced analytics
        self.ax_loss_distribution = self.fig.add_subplot(gs[2, 0])
        self.ax_phase_stats = self.fig.add_subplot(gs[2, 1])
        self.ax_weights = self.fig.add_subplot(gs[2, 2])
        self.ax_convergence = self.fig.add_subplot(gs[2, 3])
        
        # Style improvements
        for ax in self.fig.get_axes():
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            
        # Animation
        self.ani = None
        self.is_running = True
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print("🎯 Training Dashboard Starting...")
        print(f"📊 Monitoring: {metrics_csv.name}")
        print(f"📈 Phase Data: {phase_csv.name}")
        print(f"🔄 Update Interval: {update_interval}s")
        print("Press Ctrl+C to stop")
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n🛑 Stopping dashboard...")
        self.is_running = False
        if self.ani:
            self.ani.event_source.stop()
        plt.close('all')
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
                except Exception as e:
                    print(f"Warning: Could not load metrics CSV: {e}")
                    
            # Load phase data
            if self.phase_csv.exists() and phase_size > 0:
                try:
                    self.phase_df = pd.read_csv(self.phase_csv)
                    if 'timestamp' in self.phase_df.columns:
                        self.phase_df['datetime'] = pd.to_datetime(self.phase_df['timestamp'], unit='s')
                    self.last_phase_size = phase_size
                except Exception as e:
                    print(f"Warning: Could not load phase CSV: {e}")
                    
            return has_new_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def update_plots(self, frame):
        """Update all plots with latest data."""
        if not self.is_running:
            return
            
        # Load new data
        if not self.load_data() or self.metrics_df.empty:
            return
            
        # Clear all axes
        for ax in self.fig.get_axes():
            ax.clear()
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
        
        try:
            self._plot_main_loss()
            self._plot_loss_components()
            self._plot_phase_transitions()
            self._plot_learning_rate()
            self._plot_training_speed()
            self._plot_loss_distribution()
            self._plot_phase_statistics()
            self._plot_gradnorm_weights()
            self._plot_convergence_analysis()
            
            # Update title with current stats
            if not self.metrics_df.empty:
                latest_step = self.metrics_df['step'].iloc[-1]
                latest_loss = self.metrics_df['total_loss'].iloc[-1]
                current_phase = self.metrics_df['phase'].iloc[-1].upper()
                
                self.fig.suptitle(
                    f'🚀 DJMGNN Training Dashboard - Step {latest_step:,} | {current_phase} Phase | Loss: {latest_loss:.4f}',
                    fontsize=16, fontweight='bold'
                )
                
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def _plot_main_loss(self):
        """Plot main loss curve with phase coloring."""
        if self.metrics_df.empty:
            return
            
        # Separate by phase for coloring
        node_data = self.metrics_df[self.metrics_df['phase'] == 'node']
        graph_data = self.metrics_df[self.metrics_df['phase'] == 'graph']
        
        # Plot with different colors for each phase
        if not node_data.empty:
            self.ax_loss.scatter(node_data['step'], node_data['total_loss'], 
                               c='#ff6b6b', alpha=0.6, s=10, label='NODE Phase')
        if not graph_data.empty:
            self.ax_loss.scatter(graph_data['step'], graph_data['total_loss'], 
                               c='#4ecdc4', alpha=0.6, s=10, label='GRAPH Phase')
        
        # Add trend line
        if len(self.metrics_df) > 10:
            # Use rolling average for trend
            window = min(100, len(self.metrics_df) // 10)
            rolling_avg = self.metrics_df['total_loss'].rolling(window=window).mean()
            self.ax_loss.plot(self.metrics_df['step'], rolling_avg, 'k-', alpha=0.8, 
                            linewidth=2, label=f'Trend (MA-{window})')
        
        self.ax_loss.set_title('📊 Total Loss Over Time', fontweight='bold')
        self.ax_loss.set_xlabel('Training Step')
        self.ax_loss.set_ylabel('Total Loss')
        self.ax_loss.legend(loc='upper right')
        self.ax_loss.set_yscale('log')
    
    def _plot_loss_components(self):
        """Plot individual loss components."""
        if self.metrics_df.empty:
            return
            
        loss_cols = ['node_loss', 'graph_loss', 'energy_loss']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        for col, color in zip(loss_cols, colors):
            if col in self.metrics_df.columns:
                # Use rolling average to smooth the curves
                window = min(50, len(self.metrics_df) // 10)
                if len(self.metrics_df) > window:
                    smooth_data = self.metrics_df[col].rolling(window=window).mean()
                    self.ax_loss_components.plot(self.metrics_df['step'], smooth_data, 
                                               color=color, alpha=0.8, linewidth=2,
                                               label=col.replace('_', ' ').title())
        
        self.ax_loss_components.set_title('📈 Loss Components', fontweight='bold')
        self.ax_loss_components.set_xlabel('Training Step')
        self.ax_loss_components.set_ylabel('Loss Value')
        self.ax_loss_components.legend()
        self.ax_loss_components.set_yscale('log')
    
    def _plot_phase_transitions(self):
        """Plot phase transitions over time."""
        if self.metrics_df.empty:
            return
        
        # Create phase transition visualization
        phases = self.metrics_df['phase'].copy()
        phase_numeric = phases.map({'node': 0, 'graph': 1})
        
        # Color the background by phase
        for i in range(len(phase_numeric) - 1):
            color = '#ffcccc' if phase_numeric.iloc[i] == 0 else '#ccffcc'
            self.ax_phase.axvspan(self.metrics_df['step'].iloc[i], 
                                self.metrics_df['step'].iloc[i+1], 
                                alpha=0.3, color=color)
        
        # Plot phase changes as step function
        self.ax_phase.step(self.metrics_df['step'], phase_numeric, where='post', 
                         linewidth=2, color='darkblue')
        
        self.ax_phase.set_title('⚡ Training Phase Transitions', fontweight='bold')
        self.ax_phase.set_xlabel('Training Step')
        self.ax_phase.set_ylabel('Phase')
        self.ax_phase.set_yticks([0, 1])
        self.ax_phase.set_yticklabels(['NODE 🧬', 'GRAPH 📈'])
        self.ax_phase.set_ylim(-0.1, 1.1)
    
    def _plot_learning_rate(self):
        """Plot learning rate schedule."""
        if self.metrics_df.empty or 'learning_rate' not in self.metrics_df.columns:
            return
            
        self.ax_lr.plot(self.metrics_df['step'], self.metrics_df['learning_rate'], 
                       'purple', linewidth=2)
        self.ax_lr.set_title('📚 Learning Rate', fontweight='bold')
        self.ax_lr.set_xlabel('Training Step')
        self.ax_lr.set_ylabel('Learning Rate')
        self.ax_lr.set_yscale('log')
    
    def _plot_training_speed(self):
        """Plot training speed (steps per second)."""
        if self.metrics_df.empty or 'steps_per_sec' not in self.metrics_df.columns:
            return
            
        # Filter out unrealistic values
        speed_data = self.metrics_df['steps_per_sec']
        speed_data = speed_data[(speed_data > 0) & (speed_data < 100)]
        
        if not speed_data.empty:
            # Use rolling average for smoother display
            window = min(20, len(speed_data) // 5)
            if len(speed_data) > window:
                smooth_speed = speed_data.rolling(window=window).mean()
                steps_subset = self.metrics_df.loc[speed_data.index, 'step']
                self.ax_speed.plot(steps_subset, smooth_speed, 'orange', linewidth=2)
        
        self.ax_speed.set_title('⚡ Training Speed', fontweight='bold')
        self.ax_speed.set_xlabel('Training Step')
        self.ax_speed.set_ylabel('Steps/Second')
    
    def _plot_loss_distribution(self):
        """Plot loss distribution histogram."""
        if self.metrics_df.empty:
            return
            
        # Get recent losses (last 1000 steps)
        recent_losses = self.metrics_df['total_loss'].tail(1000)
        
        self.ax_loss_distribution.hist(recent_losses, bins=30, alpha=0.7, 
                                     color='skyblue', edgecolor='black')
        self.ax_loss_distribution.axvline(recent_losses.mean(), color='red', 
                                        linestyle='--', label=f'Mean: {recent_losses.mean():.4f}')
        self.ax_loss_distribution.set_title('📊 Loss Distribution\n(Last 1K Steps)', fontweight='bold')
        self.ax_loss_distribution.set_xlabel('Total Loss')
        self.ax_loss_distribution.set_ylabel('Frequency')
        self.ax_loss_distribution.legend()
    
    def _plot_phase_statistics(self):
        """Plot phase statistics."""
        if self.phase_df.empty:
            return
            
        if 'phase_count_node' in self.phase_df.columns and 'phase_count_graph' in self.phase_df.columns:
            latest_phase_data = self.phase_df.iloc[-1]
            
            phases = ['NODE 🧬', 'GRAPH 📈']
            counts = [latest_phase_data['phase_count_node'], latest_phase_data['phase_count_graph']]
            colors = ['#ff6b6b', '#4ecdc4']
            
            wedges, texts, autotexts = self.ax_phase_stats.pie(counts, labels=phases, colors=colors, 
                                                             autopct='%1.1f%%', startangle=90)
            self.ax_phase_stats.set_title('⚖️ Phase Distribution', fontweight='bold')
    
    def _plot_gradnorm_weights(self):
        """Plot GradNorm loss weights over time."""
        if self.metrics_df.empty:
            return
            
        weight_cols = ['weight_node', 'weight_graph', 'weight_energy']
        colors = ['#ff6b6b', '#4ecdc4', '#ffa726']
        labels = ['Node Weight', 'Graph Weight', 'Energy Weight']
        
        for col, color, label in zip(weight_cols, colors, labels):
            if col in self.metrics_df.columns:
                # Smooth the weights
                window = min(50, len(self.metrics_df) // 10)
                if len(self.metrics_df) > window:
                    smooth_weights = self.metrics_df[col].rolling(window=window).mean()
                    self.ax_weights.plot(self.metrics_df['step'], smooth_weights, 
                                       color=color, linewidth=2, label=label)
        
        self.ax_weights.set_title('⚖️ GradNorm Weights', fontweight='bold')
        self.ax_weights.set_xlabel('Training Step')
        self.ax_weights.set_ylabel('Weight Value')
        self.ax_weights.legend()
    
    def _plot_convergence_analysis(self):
        """Plot convergence analysis."""
        if self.metrics_df.empty:
            return
            
        # Calculate improvement rate (negative derivative of loss)
        if len(self.metrics_df) > 50:
            window = 50
            loss_smooth = self.metrics_df['total_loss'].rolling(window=window).mean()
            improvement_rate = -loss_smooth.diff()
            
            self.ax_convergence.plot(self.metrics_df['step'][window:], 
                                   improvement_rate[window:], 
                                   'green', alpha=0.7, linewidth=1)
            
            # Add zero line
            self.ax_convergence.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
        self.ax_convergence.set_title('📈 Convergence Rate', fontweight='bold')
        self.ax_convergence.set_xlabel('Training Step')
        self.ax_convergence.set_ylabel('Loss Improvement Rate')
    
    def start(self):
        """Start the real-time dashboard."""
        # Initial data load
        self.load_data()
        
        # Initial plot update
        self.update_plots(0)
        
        # Setup animation with explicit parameters to avoid warnings
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plots, 
            interval=int(self.update_interval*1000),
            blit=False, repeat=True, save_count=None, cache_frame_data=False
        )
        
        # Maximize window
        mng = plt.get_current_fig_manager()
        try:
            # Try different ways to maximize based on backend
            if hasattr(mng, 'window'):
                if hasattr(mng.window, 'state'):
                    mng.window.state('zoomed')  # Tkinter
                elif hasattr(mng.window, 'showMaximized'):
                    mng.window.showMaximized()  # Qt
        except Exception as e:
            print(f"Note: Could not maximize window: {e}")
        
        # Adjust layout with error handling
        try:
            plt.tight_layout()
        except Exception as e:
            print(f"Note: Layout adjustment failed: {e}")
        
        print("🎨 Dashboard displayed! Close window or press Ctrl+C to stop.")
        
        # Show the plot - this will block until window is closed
        plt.show()
        
        # Keep animation reference to prevent garbage collection
        return self.ani


def find_latest_csv_files(directory: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the most recent training CSV files in a directory."""
    metrics_files = list(directory.glob("training_metrics_*.csv"))
    phase_files = list(directory.glob("phase_summary_*.csv"))
    
    # Sort by modification time (most recent first)
    metrics_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    phase_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    metrics_csv = metrics_files[0] if metrics_files else None
    phase_csv = phase_files[0] if phase_files else None
    
    return metrics_csv, phase_csv


def main():
    parser = argparse.ArgumentParser(
        description="Real-time DJMGNN Training Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Specify CSV files directly
    python scripts/training_dashboard.py --metrics-csv production_checkpoints/training_metrics_20250722_215450.csv --phase-csv production_checkpoints/phase_summary_20250722_215450.csv
    
    # Auto-detect latest files in directory
    python scripts/training_dashboard.py --auto-detect production_checkpoints/
    
    # Custom update interval
    python scripts/training_dashboard.py --auto-detect production_checkpoints/ --update-interval 1.0
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
    
    # Check if running in a display environment
    if not matplotlib.get_backend():
        print("❌ No display backend available. Run with X11 forwarding or on a machine with display.")
        sys.exit(1)
    
    # Create and start dashboard
    try:
        dashboard = TrainingDashboard(metrics_csv, phase_csv, args.update_interval)
        ani = dashboard.start()  # Keep reference to animation
        
        # Keep the program running
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
            
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()