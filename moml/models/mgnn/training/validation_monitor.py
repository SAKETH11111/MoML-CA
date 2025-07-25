"""
moml/models/mgnn/training/validation_monitor.py



from __future__ import annotations  # Postpone evaluation of type annotations to avoid runtime errors when optional deps are missing

"""

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from collections import defaultdict, deque
from enum import Enum
import logging
import json
import threading
from datetime import datetime, timedelta

import torch
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter

# Optional imports for visualization and monitoring
try:
    import types

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = types.SimpleNamespace(Figure=object)  # dummy placeholder when matplotlib missing
    import matplotlib.animation as animation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    import types
    plt = types.SimpleNamespace(Figure=object, Axes=object)
    sns = None

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    sp = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .callbacks import Callback

logger = logging.getLogger(__name__)


class VisualizationTheme(Enum):
    """Enumeration for visualization themes."""
    LIGHT = "light"
    DARK = "dark"
    COLORBLIND = "colorblind"
    PUBLICATION = "publication"
    MINIMAL = "minimal"


class AlertLevel(Enum):
    """Enumeration for alert levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DashboardConfig:
    """
    Configuration class for the validation monitoring dashboard.
    
    Attributes:
        # Display Settings
        theme: Visualization theme for plots and dashboard
        figure_size: Default figure size (width, height) in inches
        dpi: Resolution for saved figures
        update_interval: Update interval for real-time plots (seconds)
        max_points_display: Maximum number of points to display in plots
        
        # Metrics Configuration
        primary_metrics: List of primary metrics to monitor prominently
        secondary_metrics: List of secondary metrics for detailed view
        custom_metrics: Dictionary of custom derived metrics to compute
        metric_colors: Custom color mapping for metrics
        
        # Visualization Options
        show_trend_lines: Whether to show trend lines in plots
        show_confidence_intervals: Whether to show confidence intervals
        show_moving_averages: Whether to show moving averages
        moving_average_window: Window size for moving averages
        smooth_curves: Whether to apply smoothing to curves
        
        # Alert Configuration
        enable_alerts: Whether to enable the alert system
        alert_thresholds: Thresholds for different alert levels
        alert_cooldown: Cooldown period between similar alerts (seconds)
        
        # Export Settings
        export_dir: Directory for exporting plots and reports
        auto_export: Whether to automatically export plots
        export_formats: List of formats to export ('png', 'pdf', 'svg', 'html')
        
        # Integration Settings
        log_to_wandb: Whether to log visualizations to Weights & Biases
        log_to_tensorboard: Whether to log visualizations to TensorBoard
        create_html_dashboard: Whether to create standalone HTML dashboard
        dashboard_port: Port for live dashboard server (if enabled)
    """
    
    # Display Settings
    theme: VisualizationTheme = VisualizationTheme.LIGHT
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    update_interval: float = 5.0
    max_points_display: int = 1000
    
    # Metrics Configuration
    primary_metrics: List[str] = field(default_factory=lambda: ["train_loss", "val_loss"])
    secondary_metrics: List[str] = field(default_factory=lambda: ["learning_rate", "grad_norm"])
    custom_metrics: Dict[str, str] = field(default_factory=dict)  # name: formula
    metric_colors: Dict[str, str] = field(default_factory=dict)
    
    # Visualization Options
    show_trend_lines: bool = True
    show_confidence_intervals: bool = True
    show_moving_averages: bool = True
    moving_average_window: int = 10
    smooth_curves: bool = True
    
    # Alert Configuration
    enable_alerts: bool = True
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    alert_cooldown: float = 300.0  # 5 minutes
    
    # Export Settings
    export_dir: str = "monitoring_exports"
    auto_export: bool = False
    export_formats: List[str] = field(default_factory=lambda: ["png", "html"])
    
    # Integration Settings
    log_to_wandb: bool = WANDB_AVAILABLE
    log_to_tensorboard: bool = TENSORBOARD_AVAILABLE
    create_html_dashboard: bool = True
    dashboard_port: int = 8050
    
    def __post_init__(self):
        """Validate and normalize configuration parameters."""
        # Ensure export directory exists
        Path(self.export_dir).mkdir(parents=True, exist_ok=True)
        
        # Set default color scheme based on theme
        if not self.metric_colors:
            self.metric_colors = self._get_default_colors()
        
        # Validate alert thresholds format
        for metric, thresholds in self.alert_thresholds.items():
            required_keys = ["warning", "error", "critical"]
            if not all(key in thresholds for key in required_keys):
                logger.warning(f"Incomplete alert thresholds for {metric}")
        
        # Disable features if dependencies not available
        if self.log_to_wandb and not WANDB_AVAILABLE:
            warnings.warn("wandb not available, disabling wandb logging")
            self.log_to_wandb = False
        
        if self.log_to_tensorboard and not TENSORBOARD_AVAILABLE:
            warnings.warn("tensorboard not available, disabling tensorboard logging")
            self.log_to_tensorboard = False
        
        if self.create_html_dashboard and not PLOTLY_AVAILABLE:
            warnings.warn("plotly not available, disabling HTML dashboard")
            self.create_html_dashboard = False
    
    def _get_default_colors(self) -> Dict[str, str]:
        """Get default color scheme based on theme."""
        if self.theme == VisualizationTheme.DARK:
            return {
                "train_loss": "#FF6B6B",
                "val_loss": "#4ECDC4",
                "learning_rate": "#45B7D1",
                "grad_norm": "#FFA07A",
                "accuracy": "#98D8C8",
                "mae": "#F7DC6F",
                "mse": "#BB8FCE",
                "r2": "#85C1E9",
            }
        elif self.theme == VisualizationTheme.COLORBLIND:
            return {
                "train_loss": "#E69F00",
                "val_loss": "#56B4E9",
                "learning_rate": "#009E73",
                "grad_norm": "#F0E442",
                "accuracy": "#0072B2",
                "mae": "#D55E00",
                "mse": "#CC79A7",
                "r2": "#999999",
            }
        else:  # Light, Publication, or Minimal themes
            return {
                "train_loss": "#1f77b4",
                "val_loss": "#ff7f0e",
                "learning_rate": "#2ca02c",
                "grad_norm": "#d62728",
                "accuracy": "#9467bd",
                "mae": "#8c564b",
                "mse": "#e377c2",
                "r2": "#7f7f7f",
            }


class AlertSystem:
    """
    Configurable alert system for monitoring training anomalies.
    
    This class provides intelligent alerting based on metric thresholds,
    trend analysis, and anomaly detection with configurable cooldown periods.
    """
    
    def __init__(self, config: DashboardConfig):
        """
        Initialize alert system.
        
        Args:
            config: Dashboard configuration containing alert settings
        """
        self.config = config
        self.enabled = config.enable_alerts
        self.alert_history = []
        self.last_alert_times = defaultdict(float)
        self.alert_callbacks = []
    
    def add_alert_callback(self, callback: Callable[[str, AlertLevel, Dict[str, Any]], None]) -> None:
        """
        Add a callback to be called when alerts are triggered.
        
        Args:
            callback: Function to call with (message, level, context) parameters
        """
        self.alert_callbacks.append(callback)
    
    def check_alerts(self, metrics: Dict[str, float], epoch: int) -> List[Dict[str, Any]]:
        """
        Check metrics against alert thresholds and trigger alerts if necessary.
        
        Args:
            metrics: Dictionary of current metric values
            epoch: Current epoch number
            
        Returns:
            List of triggered alerts
        """
        if not self.enabled:
            return []
        
        triggered_alerts = []
        current_time = time.time()
        
        for metric_name, value in metrics.items():
            # Skip None values
            if value is None:
                continue
                
            if metric_name in self.config.alert_thresholds:
                thresholds = self.config.alert_thresholds[metric_name]
                alert = self._check_metric_thresholds(metric_name, value, thresholds, epoch)
                
                if alert:
                    # Check cooldown period
                    alert_key = f"{metric_name}_{alert['level'].value}"
                    if current_time - self.last_alert_times[alert_key] > self.config.alert_cooldown:
                        triggered_alerts.append(alert)
                        self.last_alert_times[alert_key] = current_time
                        self._trigger_alert(alert)
        
        return triggered_alerts
    
    def _check_metric_thresholds(
        self, 
        metric_name: str, 
        value: float, 
        thresholds: Dict[str, float], 
        epoch: int
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a metric value exceeds any thresholds.
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
            thresholds: Dictionary of threshold levels
            epoch: Current epoch number
            
        Returns:
            Alert dictionary if threshold exceeded, None otherwise
        """
        # Check thresholds in order of severity
        for level_name in ["critical", "error", "warning"]:
            if level_name in thresholds:
                threshold = thresholds[level_name]
                
                # Determine if threshold is exceeded (depends on metric type)
                exceeded = self._is_threshold_exceeded(metric_name, value, threshold)
                
                if exceeded:
                    return {
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "level": AlertLevel(level_name),
                        "epoch": epoch,
                        "timestamp": time.time(),
                        "message": f"{metric_name} ({value:.6f}) exceeded {level_name} threshold ({threshold:.6f})"
                    }
        
        return None
    
    def _is_threshold_exceeded(self, metric_name: str, value: float, threshold: float) -> bool:
        """
        Determine if a threshold is exceeded based on metric type.
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
            threshold: Threshold value
            
        Returns:
            True if threshold is exceeded
        """
        # Metrics that should be minimized (lower is better)
        minimize_metrics = {"loss", "error", "mae", "mse", "rmse"}
        
        # Check if this is a minimization metric
        is_minimize = any(keyword in metric_name.lower() for keyword in minimize_metrics)
        
        if is_minimize:
            return value > threshold  # Alert if value is too high
        else:
            return value < threshold  # Alert if value is too low (for accuracy, etc.)
    
    def _trigger_alert(self, alert: Dict[str, Any]) -> None:
        """
        Trigger an alert by calling all registered callbacks.
        
        Args:
            alert: Alert dictionary with details
        """
        self.alert_history.append(alert)
        
        # Call all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert["message"], alert["level"], alert)
            except Exception as e:
                logger.error(f"Error calling alert callback: {e}")
        
        # Log alert
        level_name = alert["level"].value.upper()
        logger.log(
            getattr(logging, level_name, logging.INFO),
            f"ALERT [{level_name}]: {alert['message']}"
        )
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of alerts in the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with alert summary statistics
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [a for a in self.alert_history if a["timestamp"] > cutoff_time]
        
        summary = {
            "total_alerts": len(recent_alerts),
            "by_level": defaultdict(int),
            "by_metric": defaultdict(int),
            "most_recent": recent_alerts[-1] if recent_alerts else None,
        }
        
        for alert in recent_alerts:
            summary["by_level"][alert["level"].value] += 1
            summary["by_metric"][alert["metric"]] += 1
        
        return dict(summary)


class MetricsVisualizer:
    """
    Advanced plotting and visualization utilities for training metrics.
    
    This class provides publication-quality visualizations with multiple
    backends (matplotlib, plotly) and themes.
    """
    
    def __init__(self, config: DashboardConfig):
        """
        Initialize metrics visualizer.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        self._setup_theme()
    
    def _setup_theme(self) -> None:
        """Set up visualization theme."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if self.config.theme == VisualizationTheme.DARK:
            plt.style.use('dark_background')
        elif self.config.theme == VisualizationTheme.PUBLICATION:
            # Publication-ready style
            plt.rcParams.update({
                'font.size': 12,
                'axes.linewidth': 1.2,
                'grid.alpha': 0.3,
                'lines.linewidth': 2,
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
            })
        elif self.config.theme == VisualizationTheme.MINIMAL:
            # Minimal clean style
            plt.rcParams.update({
                'axes.spines.top': False,
                'axes.spines.right': False,
                'grid.alpha': 0.2,
                'axes.axisbelow': True,
            })
        
        # Set seaborn palette if available
        if sns is not None:
            if self.config.theme == VisualizationTheme.COLORBLIND:
                sns.set_palette("colorblind")
            else:
                sns.set_palette("husl")
    
    def create_training_overview(
        self, 
        metrics_data: Dict[str, List[float]], 
        epochs: List[int]
    ) -> Any:
        """
        Create comprehensive training overview plot.
        
        Args:
            metrics_data: Dictionary mapping metric names to lists of values
            epochs: List of epoch numbers
            
        Returns:
            Matplotlib figure with training overview
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for creating training overview")
        
        # Determine layout based on number of metrics
        primary_metrics = [m for m in self.config.primary_metrics if m in metrics_data]
        secondary_metrics = [m for m in self.config.secondary_metrics if m in metrics_data]
        
        n_primary = len(primary_metrics)
        n_secondary = len(secondary_metrics)
        
        # Create figure with subplots
        if n_secondary > 0:
            fig, axes = plt.subplots(2, max(n_primary, n_secondary), 
                                   figsize=self.config.figure_size, 
                                   dpi=self.config.dpi)
            if axes.ndim == 1:
                axes = axes.reshape(1, -1)
        else:
            fig, axes = plt.subplots(1, n_primary, 
                                   figsize=self.config.figure_size, 
                                   dpi=self.config.dpi)
            if n_primary == 1:
                axes = [axes]
            axes = [axes]  # Make it 2D for consistency
        
        # Plot primary metrics
        for i, metric in enumerate(primary_metrics):
            if i < axes.shape[1]:
                ax = axes[0, i]
                self._plot_metric(ax, epochs, metrics_data[metric], metric, primary=True)
        
        # Plot secondary metrics
        if n_secondary > 0:
            for i, metric in enumerate(secondary_metrics):
                if i < axes.shape[1] and len(axes) > 1:
                    ax = axes[1, i]
                    self._plot_metric(ax, epochs, metrics_data[metric], metric, primary=False)
        
        # Hide unused subplots
        for i in range(max(n_primary, n_secondary), axes.shape[1]):
            if len(axes) > 1:
                axes[1, i].set_visible(False)
            if i >= n_primary:
                axes[0, i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def _plot_metric(
        self, 
        ax: plt.Axes, 
        epochs: List[int], 
        values: List[float], 
        metric_name: str, 
        primary: bool = True
    ) -> None:
        """
        Plot a single metric with all enhancements.
        
        Args:
            ax: Matplotlib axes to plot on
            epochs: List of epoch numbers
            values: List of metric values
            metric_name: Name of the metric
            primary: Whether this is a primary metric
        """
        if len(epochs) != len(values):
            logger.warning(f"Epoch and value lengths don't match for {metric_name}")
            return
        
        # Get color for this metric
        color = self.config.metric_colors.get(metric_name, None)
        
        # Main line plot
        ax.plot(epochs, values, color=color, linewidth=2.5 if primary else 1.5, 
                label=metric_name, alpha=0.8)
        
        # Add moving average if requested
        if self.config.show_moving_averages and len(values) > self.config.moving_average_window:
            ma_values = self._compute_moving_average(values, self.config.moving_average_window)
            ma_epochs = epochs[self.config.moving_average_window-1:]
            ax.plot(ma_epochs, ma_values, color=color, linewidth=1, 
                   linestyle='--', alpha=0.6, label=f'{metric_name} (MA)')
        
        # Add trend line if requested
        if self.config.show_trend_lines and len(values) > 10:
            trend_line = self._compute_trend_line(epochs, values)
            ax.plot(epochs, trend_line, color=color, linewidth=1, 
                   linestyle=':', alpha=0.5, label=f'{metric_name} (trend)')
        
        # Add confidence intervals if requested
        if self.config.show_confidence_intervals and len(values) > 20:
            self._add_confidence_interval(ax, epochs, values, color)
        
        # Formatting
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis limits to focus on relevant range
        if values:
            y_min, y_max = min(values), max(values)
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    def _compute_moving_average(self, values: List[float], window: int) -> List[float]:
        """Compute moving average with specified window size."""
        return [np.mean(values[max(0, i-window+1):i+1]) for i in range(window-1, len(values))]
    
    def _compute_trend_line(self, epochs: List[int], values: List[float]) -> List[float]:
        """Compute linear trend line using least squares."""
        epochs_array = np.array(epochs)
        values_array = np.array(values)
        
        # Remove any NaN or infinite values
        mask = np.isfinite(values_array)
        if not np.any(mask):
            return values
        
        epochs_clean = epochs_array[mask]
        values_clean = values_array[mask]
        
        # Linear regression
        A = np.vstack([epochs_clean, np.ones(len(epochs_clean))]).T
        slope, intercept = np.linalg.lstsq(A, values_clean, rcond=None)[0]
        
        return slope * epochs_array + intercept
    
    def _add_confidence_interval(
        self, 
        ax: plt.Axes, 
        epochs: List[int], 
        values: List[float], 
        color: str
    ) -> None:
        """Add confidence interval band to the plot."""
        # Compute rolling standard deviation
        window = min(20, len(values) // 4)
        rolling_std = []
        
        for i in range(len(values)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(values), i + window // 2 + 1)
            rolling_std.append(np.std(values[start_idx:end_idx]))
        
        # Create confidence bands
        values_array = np.array(values)
        std_array = np.array(rolling_std)
        
        upper_bound = values_array + 1.96 * std_array  # 95% confidence
        lower_bound = values_array - 1.96 * std_array
        
        ax.fill_between(epochs, lower_bound, upper_bound, 
                       color=color, alpha=0.2, linewidth=0)
    
    def create_plotly_dashboard(
        self, 
        metrics_data: Dict[str, List[float]], 
        epochs: List[int]
    ) -> str:
        """
        Create interactive Plotly dashboard.
        
        Args:
            metrics_data: Dictionary mapping metric names to lists of values
            epochs: List of epoch numbers
            
        Returns:
            Path to saved HTML dashboard file
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for creating interactive dashboard")
        
        # Create subplots
        n_metrics = len(metrics_data)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig = sp.make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=list(metrics_data.keys()),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Add traces for each metric
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Main line
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    mode='lines',
                    name=metric_name,
                    line=dict(
                        color=self.config.metric_colors.get(metric_name),
                        width=2
                    ),
                    hovertemplate=f'{metric_name}: %{{y:.6f}}<br>Epoch: %{{x}}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add moving average if requested
            if self.config.show_moving_averages and len(values) > self.config.moving_average_window:
                ma_values = self._compute_moving_average(values, self.config.moving_average_window)
                ma_epochs = epochs[self.config.moving_average_window-1:]
                
                fig.add_trace(
                    go.Scatter(
                        x=ma_epochs,
                        y=ma_values,
                        mode='lines',
                        name=f'{metric_name} (MA)',
                        line=dict(
                            color=self.config.metric_colors.get(metric_name),
                            width=1,
                            dash='dash'
                        ),
                        opacity=0.7,
                        hovertemplate=f'{metric_name} MA: %{{y:.6f}}<br>Epoch: %{{x}}<extra></extra>'
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title="Training Metrics Dashboard",
            showlegend=True,
            height=400 * n_rows,
            template="plotly_white" if self.config.theme != VisualizationTheme.DARK else "plotly_dark"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Epoch")
        for i, metric_name in enumerate(metrics_data.keys()):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.update_yaxes(title_text=metric_name.replace('_', ' ').title(), row=row, col=col)
        
        # Save to HTML file
        output_path = Path(self.config.export_dir) / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(output_path))
        
        return str(output_path)


class ValidationMonitor(Callback):
    """
    Comprehensive validation monitoring callback with advanced visualization
    and alerting capabilities.
    
    This callback provides real-time monitoring of training progress with
    interactive visualizations, trend analysis, and intelligent alerting.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None, **kwargs):
        """
        Initialize validation monitor.
        
        Args:
            config: DashboardConfig instance or None for default config
            **kwargs: Additional configuration parameters
        """
        # Initialize configuration
        if config is None:
            config = DashboardConfig(**kwargs)
        else:
            # Override config with any provided kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        
        # Initialize components
        self.visualizer = MetricsVisualizer(config)
        self.alert_system = AlertSystem(config)
        
        # Data storage
        self.metrics_history = defaultdict(list)
        self.epochs = []
        self.timestamps = []
        
        # Monitoring state
        self.monitoring_active = False
        self.last_update_time = 0
        
        # Integration setup
        self.tensorboard_writer = None
        if config.log_to_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = Path("runs") / f"validation_monitor_{time.strftime('%Y%m%d_%H%M%S')}"
            self.tensorboard_writer = SummaryWriter(log_dir)
        
        # Set up alert callbacks
        self.alert_system.add_alert_callback(self._log_alert)
        if config.log_to_wandb and WANDB_AVAILABLE:
            self.alert_system.add_alert_callback(self._wandb_alert)
        
        logger.info("ValidationMonitor initialized with comprehensive monitoring capabilities")
    
    def on_train_begin(self, trainer: Any) -> None:
        """Initialize monitoring at the beginning of training."""
        self.monitoring_active = True
        self.metrics_history.clear()
        self.epochs.clear()
        self.timestamps.clear()
        self.last_update_time = time.time()
        
        logger.info("Validation monitoring started")
    
    def on_epoch_end(self, trainer: Any, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Update monitoring data and visualizations after each epoch.
        
        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            logs: Dictionary of metrics from the epoch
        """
        if not self.monitoring_active or not logs:
            return
        
        # Store epoch data
        self.epochs.append(epoch)
        self.timestamps.append(time.time())
        
        # Update metrics history
        for metric_name, value in logs.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.metrics_history[metric_name].append(float(value))
        
        # Check for alerts
        alerts = self.alert_system.check_alerts(logs, epoch)
        
        # Update visualizations if it's time
        current_time = time.time()
        if current_time - self.last_update_time > self.config.update_interval:
            self._update_visualizations(epoch)
            self.last_update_time = current_time
        
        # Log to monitoring systems
        self._log_to_monitoring_systems(logs, epoch)
    
    def _update_visualizations(self, epoch: int) -> None:
        """Update and save visualizations."""
        try:
            # Create training overview plot
            if MATPLOTLIB_AVAILABLE and self.metrics_history:
                fig = self.visualizer.create_training_overview(dict(self.metrics_history), self.epochs)
                
                # Save plot if auto-export is enabled
                if self.config.auto_export:
                    for fmt in self.config.export_formats:
                        if fmt in ['png', 'pdf', 'svg']:
                            output_path = Path(self.config.export_dir) / f"training_overview_epoch_{epoch:04d}.{fmt}"
                            fig.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
                
                plt.close(fig)  # Free memory
            
            # Create interactive dashboard
            if PLOTLY_AVAILABLE and self.config.create_html_dashboard and self.metrics_history:
                dashboard_path = self.visualizer.create_plotly_dashboard(dict(self.metrics_history), self.epochs)
                logger.debug(f"Updated dashboard: {dashboard_path}")
                
        except Exception as e:
            logger.error(f"Error updating visualizations: {e}")
    
    def _log_to_monitoring_systems(self, logs: Dict[str, Any], epoch: int) -> None:
        """Log metrics to configured monitoring systems."""
        # TensorBoard logging
        if self.tensorboard_writer:
            for metric_name, value in logs.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self.tensorboard_writer.add_scalar(f"monitor/{metric_name}", value, epoch)
            
            # Log alert summary
            alert_summary = self.alert_system.get_alert_summary(hours=1)
            self.tensorboard_writer.add_scalar("monitor/alerts_last_hour", alert_summary["total_alerts"], epoch)
        
        # Weights & Biases logging
        if self.config.log_to_wandb and WANDB_AVAILABLE and wandb.run:
            wandb_logs = {}
            
            # Add monitoring-specific metrics
            for metric_name, values in self.metrics_history.items():
                if values:
                    wandb_logs[f"monitor/{metric_name}_trend"] = self._get_trend_direction(values)
                    if len(values) >= 10:
                        recent_values = values[-10:]
                        wandb_logs[f"monitor/{metric_name}_volatility"] = np.std(recent_values)
            
            # Add alert information
            alert_summary = self.alert_system.get_alert_summary(hours=1)
            wandb_logs["monitor/alerts_last_hour"] = alert_summary["total_alerts"]
            
            wandb.log(wandb_logs, step=epoch)
    
    def _get_trend_direction(self, values: List[float]) -> float:
        """
        Get trend direction as a single number (-1 to 1).
        
        Args:
            values: List of metric values
            
        Returns:
            Trend direction: -1 (decreasing), 0 (stable), 1 (increasing)
        """
        if len(values) < 5:
            return 0.0
        
        recent_values = values[-min(10, len(values)):]
        x = np.arange(len(recent_values))
        
        # Linear regression to get trend
        correlation = np.corrcoef(x, recent_values)[0, 1]
        return 0.0 if np.isnan(correlation) else np.clip(correlation, -1.0, 1.0)
    
    def _log_alert(self, message: str, level: AlertLevel, context: Dict[str, Any]) -> None:
        """Log alert to standard logging system."""
        level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }
        
        logger.log(level_map[level], f"MONITOR ALERT: {message}")
    
    def _wandb_alert(self, message: str, level: AlertLevel, context: Dict[str, Any]) -> None:
        """Log alert to Weights & Biases."""
        if WANDB_AVAILABLE and wandb.run:
            wandb.alert(
                title=f"Training Alert ({level.value.upper()})",
                text=message,
                level=getattr(wandb.AlertLevel, level.value.upper(), wandb.AlertLevel.INFO)
            )
    
    def on_train_end(self, trainer: Any) -> None:
        """Finalize monitoring and generate reports."""
        if not self.monitoring_active:
            return
        
        # Generate final report
        self._generate_final_report()
        
        # Close resources
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        self.monitoring_active = False
        logger.info("Validation monitoring completed")
    
    def _generate_final_report(self) -> None:
        """Generate comprehensive final training report."""
        try:
            # Create final visualizations
            if MATPLOTLIB_AVAILABLE and self.metrics_history:
                fig = self.visualizer.create_training_overview(dict(self.metrics_history), self.epochs)
                
                # Save final plots in all requested formats
                for fmt in self.config.export_formats:
                    if fmt in ['png', 'pdf', 'svg']:
                        output_path = Path(self.config.export_dir) / f"final_training_report.{fmt}"
                        fig.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
                        logger.info(f"Saved final training report: {output_path}")
                
                plt.close(fig)
            
            # Create final interactive dashboard
            if PLOTLY_AVAILABLE and self.config.create_html_dashboard and self.metrics_history:
                final_dashboard = self.visualizer.create_plotly_dashboard(dict(self.metrics_history), self.epochs)
                final_path = Path(self.config.export_dir) / "final_dashboard.html"
                
                # Copy to final location
                import shutil
                shutil.copy2(final_dashboard, final_path)
                logger.info(f"Saved final interactive dashboard: {final_path}")
            
            # Generate text report
            self._save_text_report()
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
    
    def _save_text_report(self) -> None:
        """Save comprehensive text report with statistics."""
        report_path = Path(self.config.export_dir) / "training_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Training Monitoring Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Training Duration: {len(self.epochs)} epochs\n")
            f.write(f"Start Time: {datetime.fromtimestamp(self.timestamps[0]).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End Time: {datetime.fromtimestamp(self.timestamps[-1]).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Metrics summary
            f.write("Metrics Summary:\n")
            f.write("-" * 20 + "\n")
            
            for metric_name, values in self.metrics_history.items():
                if values:
                    f.write(f"\n{metric_name}:\n")
                    f.write(f"  Final Value: {values[-1]:.6f}\n")
                    f.write(f"  Best Value: {min(values) if 'loss' in metric_name else max(values):.6f}\n")
                    f.write(f"  Mean: {np.mean(values):.6f}\n")
                    f.write(f"  Std Dev: {np.std(values):.6f}\n")
                    f.write(f"  Trend: {self._get_trend_description(values)}\n")
            
            # Alert summary
            alert_summary = self.alert_system.get_alert_summary(hours=24*7)  # Last week
            f.write(f"\nAlert Summary:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Alerts: {alert_summary['total_alerts']}\n")
            
            for level, count in alert_summary['by_level'].items():
                f.write(f"  {level.title()}: {count}\n")
        
        logger.info(f"Saved training report: {report_path}")
    
    def _get_trend_description(self, values: List[float]) -> str:
        """Get human-readable trend description."""
        trend_value = self._get_trend_direction(values)
        
        if trend_value > 0.3:
            return "Increasing"
        elif trend_value < -0.3:
            return "Decreasing"
        else:
            return "Stable"
    
    def get_current_statistics(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        stats = {
            "epochs_monitored": len(self.epochs),
            "metrics_tracked": len(self.metrics_history),
            "total_alerts": len(self.alert_system.alert_history),
            "monitoring_active": self.monitoring_active,
        }
        
        # Add per-metric statistics
        for metric_name, values in self.metrics_history.items():
            if values:
                stats[f"{metric_name}_current"] = values[-1]
                stats[f"{metric_name}_trend"] = self._get_trend_description(values)
        
        return stats


# Factory function for easy instantiation
def create_validation_monitor(
    primary_metrics: List[str] = None,
    secondary_metrics: List[str] = None,
    theme: str = "light",
    enable_alerts: bool = True,
    log_to_wandb: bool = None,
    log_to_tensorboard: bool = None,
    **kwargs
) -> ValidationMonitor:
    """
    Factory function to create a ValidationMonitor with common parameters.
    
    Args:
        primary_metrics: List of primary metrics to monitor prominently
        secondary_metrics: List of secondary metrics for detailed view
        theme: Visualization theme ('light', 'dark', 'colorblind', 'publication', 'minimal')
        enable_alerts: Whether to enable the alert system
        log_to_wandb: Whether to log to Weights & Biases (None for auto-detect)
        log_to_tensorboard: Whether to log to TensorBoard (None for auto-detect)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ValidationMonitor instance
    """
    config_params = {
        "theme": VisualizationTheme(theme.lower()),
        "enable_alerts": enable_alerts,
        **kwargs
    }
    
    if primary_metrics is not None:
        config_params["primary_metrics"] = primary_metrics
    
    if secondary_metrics is not None:
        config_params["secondary_metrics"] = secondary_metrics
    
    if log_to_wandb is not None:
        config_params["log_to_wandb"] = log_to_wandb
    
    if log_to_tensorboard is not None:
        config_params["log_to_tensorboard"] = log_to_tensorboard
    
    config = DashboardConfig(**config_params)
    return ValidationMonitor(config)


# Example usage and testing
if __name__ == "__main__":
    # Example configuration for molecular property prediction monitoring
    config = DashboardConfig(
        theme=VisualizationTheme.LIGHT,
        primary_metrics=["train_loss", "val_loss"],
        secondary_metrics=["learning_rate", "grad_norm", "val_mae", "val_r2"],
        show_trend_lines=True,
        show_moving_averages=True,
        enable_alerts=True,
        alert_thresholds={
            "val_loss": {"warning": 0.1, "error": 0.5, "critical": 1.0},
            "grad_norm": {"warning": 10.0, "error": 100.0, "critical": 1000.0},
        },
        auto_export=True,
        log_to_wandb=True,
        create_html_dashboard=True,
        verbose=True
    )
    
    monitor = ValidationMonitor(config)
    print(f"Created ValidationMonitor with config: {config}")
    print("Comprehensive validation monitoring system ready for integration.")