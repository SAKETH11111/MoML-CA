"""
scripts/gradient_monitor.py

Comprehensive gradient flow monitoring and analysis utilities.

This module provides tools to monitor gradient coverage, analyze parameter
utilization, and diagnose training issues in complex neural networks like
the Joint MGNN architecture.

Usage:
    python scripts/gradient_monitor.py --model_path model.pt --data_path data.pt
    
    # Or as a utility in other scripts:
    from scripts.gradient_monitor import DetailedGradientAnalyzer
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from torch_geometric.loader import DataLoader as GraphDataLoader

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from moml.models.mgnn import JointMGNN, DJMGNN, HMGNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetailedGradientAnalyzer:
    """
    Advanced gradient flow analyzer for complex neural network architectures.
    
    Provides detailed insights into parameter utilization, gradient magnitudes,
    dead parameters, and training dynamics.
    """
    
    def __init__(self, model: nn.Module, track_history: bool = True):
        self.model = model
        self.track_history = track_history
        self.parameter_registry = self._build_parameter_registry()
        self.gradient_history = [] if track_history else None
        
    def _build_parameter_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build a comprehensive registry of all model parameters."""
        registry = {}
        
        for name, param in self.model.named_parameters():
            module_name = '.'.join(name.split('.')[:-1]) if '.' in name else 'root'
            param_name = name.split('.')[-1]
            
            registry[name] = {
                'module_name': module_name,
                'param_name': param_name,
                'shape': list(param.shape),
                'numel': param.numel(),
                'requires_grad': param.requires_grad,
                'dtype': str(param.dtype),
                'device': str(param.device),
                'is_bias': 'bias' in param_name,
                'is_weight': 'weight' in param_name,
                'is_conv': 'conv' in module_name.lower(),
                'is_norm': any(norm in module_name.lower() for norm in ['norm', 'batch', 'layer']),
                'is_attention': 'attention' or 'attn' in module_name.lower(),
                'is_fusion': 'fusion' in module_name.lower(),
                'module_type': self._get_module_type(module_name)
            }
        
        return registry
    
    def _get_module_type(self, module_name: str) -> str:
        """Classify module type based on name."""
        name_lower = module_name.lower()
        
        if 'djmgnn' in name_lower:
            return 'DJMGNN'
        elif 'hmgnn' in name_lower:
            return 'HMGNN'
        elif 'fusion' in name_lower:
            return 'CrossModelFusion'
        elif 'task_heads' in name_lower:
            return 'MultiTaskHead'
        elif any(proj in name_lower for proj in ['proj', 'projection']):
            return 'Projection'
        elif 'norm' in name_lower:
            return 'Normalization'
        else:
            return 'Other'
    
    def analyze_gradient_flow(self, detailed: bool = True) -> Dict[str, Any]:
        """
        Comprehensive gradient flow analysis.
        
        Args:
            detailed: Whether to include detailed per-parameter analysis
            
        Returns:
            Dictionary containing gradient flow statistics
        """
        analysis = {
            'timestamp': torch.tensor(0.0).item(),  # Will be set by caller
            'total_parameters': len(self.parameter_registry),
            'parameters_with_grad': 0,
            'parameters_without_grad': 0,
            'gradient_coverage_percent': 0.0,
            'module_stats': defaultdict(lambda: {
                'total_params': 0,
                'params_with_grad': 0,
                'params_without_grad': 0,
                'avg_grad_norm': 0.0,
                'max_grad_norm': 0.0,
                'min_grad_norm': float('inf')
            }),
            'gradient_norms': {},
            'dead_parameters': [],
            'problematic_modules': [],
            'parameter_categories': defaultdict(lambda: {
                'total': 0,
                'with_grad': 0,
                'avg_grad_norm': 0.0
            })
        }
        
        total_grad_norm_squared = 0.0
        param_grad_norms = []
        
        for param_name, param_info in self.parameter_registry.items():
            param = dict(self.model.named_parameters())[param_name]
            module_type = param_info['module_type']
            
            # Update module stats
            analysis['module_stats'][module_type]['total_params'] += 1
            
            # Update parameter category stats
            category = 'bias' if param_info['is_bias'] else 'weight'
            analysis['parameter_categories'][category]['total'] += 1
            
            if param.grad is not None:
                # Parameter has gradient
                analysis['parameters_with_grad'] += 1
                analysis['module_stats'][module_type]['params_with_grad'] += 1
                analysis['parameter_categories'][category]['with_grad'] += 1
                
                # Calculate gradient norm
                grad_norm = param.grad.norm().item()
                param_grad_norms.append(grad_norm)
                analysis['gradient_norms'][param_name] = grad_norm
                total_grad_norm_squared += grad_norm ** 2
                
                # Update module gradient statistics
                module_stats = analysis['module_stats'][module_type]
                module_stats['max_grad_norm'] = max(module_stats['max_grad_norm'], grad_norm)
                module_stats['min_grad_norm'] = min(module_stats['min_grad_norm'], grad_norm)
                
            else:
                # Dead parameter
                analysis['parameters_without_grad'] += 1
                analysis['module_stats'][module_type]['params_without_grad'] += 1
                analysis['dead_parameters'].append({
                    'name': param_name,
                    'module_type': module_type,
                    'shape': param_info['shape'],
                    'numel': param_info['numel']
                })
        
        # Calculate coverage percentage
        if analysis['total_parameters'] > 0:
            analysis['gradient_coverage_percent'] = (
                analysis['parameters_with_grad'] / analysis['total_parameters'] * 100
            )
        
        # Calculate average gradient norms for modules
        for module_type, stats in analysis['module_stats'].items():
            if stats['params_with_grad'] > 0:
                # Collect gradient norms for this module
                module_grad_norms = [
                    analysis['gradient_norms'][param_name] 
                    for param_name, param_info in self.parameter_registry.items()
                    if param_info['module_type'] == module_type and param_name in analysis['gradient_norms']
                ]
                if module_grad_norms:
                    stats['avg_grad_norm'] = np.mean(module_grad_norms)
                    
                if stats['min_grad_norm'] == float('inf'):
                    stats['min_grad_norm'] = 0.0
        
        # Calculate average gradient norms for parameter categories
        for category, stats in analysis['parameter_categories'].items():
            if stats['with_grad'] > 0:
                category_grad_norms = [
                    analysis['gradient_norms'][param_name]
                    for param_name, param_info in self.parameter_registry.items()
                    if (param_info['is_bias'] if category == 'bias' else not param_info['is_bias'])
                    and param_name in analysis['gradient_norms']
                ]
                if category_grad_norms:
                    stats['avg_grad_norm'] = np.mean(category_grad_norms)
        
        # Global gradient statistics
        if param_grad_norms:
            analysis['global_grad_norm'] = np.sqrt(total_grad_norm_squared)
            analysis['avg_grad_norm'] = np.mean(param_grad_norms)
            analysis['std_grad_norm'] = np.std(param_grad_norms)
            analysis['min_grad_norm'] = np.min(param_grad_norms)
            analysis['max_grad_norm'] = np.max(param_grad_norms)
        else:
            analysis['global_grad_norm'] = 0.0
            analysis['avg_grad_norm'] = 0.0
            analysis['std_grad_norm'] = 0.0
            analysis['min_grad_norm'] = 0.0
            analysis['max_grad_norm'] = 0.0
        
        # Identify problematic modules (modules with no gradients)
        for module_type, stats in analysis['module_stats'].items():
            if stats['total_params'] > 0 and stats['params_with_grad'] == 0:
                analysis['problematic_modules'].append({
                    'module_type': module_type,
                    'total_params': stats['total_params']
                })
        
        # Store in history if tracking
        if self.track_history:
            analysis['timestamp'] = len(self.gradient_history)
            self.gradient_history.append(analysis)
        
        return analysis
    
    def diagnose_gradient_issues(self, analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Diagnose potential gradient flow issues and provide recommendations.
        
        Args:
            analysis: Gradient analysis results. If None, will run new analysis.
            
        Returns:
            Dictionary containing diagnosis and recommendations
        """
        if analysis is None:
            analysis = self.analyze_gradient_flow()
        
        diagnosis = {
            'overall_health': 'HEALTHY',
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'severity_score': 0  # 0-100, higher is worse
        }
        
        # Check gradient coverage
        coverage = analysis['gradient_coverage_percent']
        if coverage < 90:
            diagnosis['issues'].append(f"Low gradient coverage: {coverage:.1f}%")
            diagnosis['severity_score'] += 30
            diagnosis['recommendations'].append("Investigate dead parameters and add regularization")
        elif coverage < 95:
            diagnosis['warnings'].append(f"Moderate gradient coverage: {coverage:.1f}%")
            diagnosis['severity_score'] += 10
        
        # Check for vanishing gradients
        avg_grad = analysis.get('avg_grad_norm', 0)
        if avg_grad < 1e-6:
            diagnosis['issues'].append(f"Vanishing gradients detected: avg norm {avg_grad:.2e}")
            diagnosis['severity_score'] += 40
            diagnosis['recommendations'].append("Consider gradient clipping, learning rate adjustment, or architecture changes")
        elif avg_grad < 1e-4:
            diagnosis['warnings'].append(f"Small gradient magnitudes: avg norm {avg_grad:.2e}")
            diagnosis['severity_score'] += 15
        
        # Check for exploding gradients
        max_grad = analysis.get('max_grad_norm', 0)
        if max_grad > 100:
            diagnosis['issues'].append(f"Exploding gradients detected: max norm {max_grad:.2e}")
            diagnosis['severity_score'] += 35
            diagnosis['recommendations'].append("Apply gradient clipping immediately")
        elif max_grad > 10:
            diagnosis['warnings'].append(f"Large gradient magnitudes: max norm {max_grad:.2e}")
            diagnosis['severity_score'] += 10
        
        # Check problematic modules
        if analysis['problematic_modules']:
            diagnosis['issues'].append(f"Modules with no gradients: {[m['module_type'] for m in analysis['problematic_modules']]}")
            diagnosis['severity_score'] += 25
            diagnosis['recommendations'].append("Add regularization to unused modules or remove them")
        
        # Check gradient distribution
        std_grad = analysis.get('std_grad_norm', 0)
        if std_grad > avg_grad * 10:  # High variance
            diagnosis['warnings'].append(f"High gradient variance: std {std_grad:.2e}")
            diagnosis['severity_score'] += 5
            diagnosis['recommendations'].append("Consider gradient normalization or batch size adjustment")
        
        # Overall health assessment
        if diagnosis['severity_score'] > 50:
            diagnosis['overall_health'] = 'CRITICAL'
        elif diagnosis['severity_score'] > 25:
            diagnosis['overall_health'] = 'NEEDS_ATTENTION'
        elif diagnosis['severity_score'] > 10:
            diagnosis['overall_health'] = 'MINOR_ISSUES'
        
        return diagnosis
    
    def generate_report(self, output_path: str = 'gradient_analysis_report.json') -> str:
        """Generate a comprehensive gradient analysis report."""
        analysis = self.analyze_gradient_flow(detailed=True)
        diagnosis = self.diagnose_gradient_issues(analysis)
        
        report = {
            'model_summary': {
                'total_parameters': len(self.parameter_registry),
                'model_type': type(self.model).__name__
            },
            'gradient_analysis': analysis,
            'diagnosis': diagnosis,
            'parameter_registry': self.parameter_registry if len(self.parameter_registry) < 1000 else "Too large to include",
            'recommendations': self._generate_specific_recommendations(analysis, diagnosis)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_path
    
    def _generate_specific_recommendations(self, analysis: Dict[str, Any], diagnosis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on analysis results."""
        recommendations = []
        
        # Coverage-based recommendations
        coverage = analysis['gradient_coverage_percent']
        if coverage < 100:
            dead_count = len(analysis['dead_parameters'])
            recommendations.append(f"Address {dead_count} dead parameters by adding L2 regularization or removing unused components")
        
        # Module-specific recommendations
        for module_type, stats in analysis['module_stats'].items():
            if stats['params_without_grad'] > 0:
                recommendations.append(f"Module {module_type}: {stats['params_without_grad']}/{stats['total_params']} parameters have no gradients")
        
        # Gradient magnitude recommendations
        avg_grad = analysis.get('avg_grad_norm', 0)
        if avg_grad > 0:
            if avg_grad < 1e-5:
                recommendations.append("Consider increasing learning rate or using gradient scaling")
            elif avg_grad > 1:
                recommendations.append("Consider decreasing learning rate or adding gradient clipping")
        
        return recommendations
    
    def create_visualization(self, output_dir: str = 'gradient_analysis_plots') -> List[str]:
        """Create visualization plots for gradient analysis."""
        os.makedirs(output_dir, exist_ok=True)
        plot_files = []
        
        if not self.gradient_history:
            analysis = self.analyze_gradient_flow()
            self.gradient_history = [analysis]
        
        latest_analysis = self.gradient_history[-1]
        
        # Plot 1: Gradient Coverage by Module
        plt.figure(figsize=(12, 6))
        modules = list(latest_analysis['module_stats'].keys())
        coverages = [
            (stats['params_with_grad'] / stats['total_params'] * 100) if stats['total_params'] > 0 else 0
            for stats in latest_analysis['module_stats'].values()
        ]
        
        bars = plt.bar(modules, coverages)
        plt.title('Gradient Coverage by Module Type')
        plt.ylabel('Coverage Percentage')
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        
        # Color bars based on coverage
        for bar, coverage in zip(bars, coverages):
            if coverage >= 95:
                bar.set_color('green')
            elif coverage >= 90:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'gradient_coverage_by_module.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)
        
        # Plot 2: Gradient Magnitude Distribution
        if latest_analysis['gradient_norms']:
            plt.figure(figsize=(10, 6))
            grad_norms = list(latest_analysis['gradient_norms'].values())
            
            plt.hist(np.log10(np.array(grad_norms) + 1e-10), bins=50, alpha=0.7, edgecolor='black')
            plt.title('Gradient Magnitude Distribution (log scale)')
            plt.xlabel('log10(Gradient Norm)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            plot_file = os.path.join(output_dir, 'gradient_magnitude_distribution.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
        
        # Plot 3: Gradient Evolution Over Time (if history available)
        if len(self.gradient_history) > 1:
            plt.figure(figsize=(12, 8))
            
            # Extract data
            timestamps = [h.get('timestamp', i) for i, h in enumerate(self.gradient_history)]
            coverages = [h['gradient_coverage_percent'] for h in self.gradient_history]
            avg_grads = [h.get('avg_grad_norm', 0) for h in self.gradient_history]
            max_grads = [h.get('max_grad_norm', 0) for h in self.gradient_history]
            
            # Coverage subplot
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, coverages, 'g-', linewidth=2, label='Gradient Coverage')
            plt.ylabel('Coverage (%)')
            plt.title('Gradient Flow Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Gradient magnitude subplot
            plt.subplot(2, 1, 2)
            plt.plot(timestamps, avg_grads, 'b-', linewidth=2, label='Average Grad Norm')
            plt.plot(timestamps, max_grads, 'r-', linewidth=2, label='Max Grad Norm')
            plt.ylabel('Gradient Norm')
            plt.xlabel('Training Step/Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, 'gradient_evolution.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
        
        return plot_files


def monitor_training_gradients(
    model: nn.Module, 
    data_loader: GraphDataLoader, 
    num_batches: int = 10,
    output_dir: str = 'gradient_monitoring'
) -> Dict[str, Any]:
    """
    Monitor gradient flow during actual training steps.
    
    Args:
        model: The model to monitor
        data_loader: Data loader for training data
        num_batches: Number of batches to monitor
        output_dir: Directory to save monitoring results
        
    Returns:
        Dictionary containing monitoring results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = DetailedGradientAnalyzer(model, track_history=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    logger.info(f"Monitoring gradient flow for {num_batches} training steps...")
    
    model.train()
    monitoring_results = {
        'batch_analyses': [],
        'summary_stats': {},
        'issues_detected': []
    }
    
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
            
        # Forward pass
        optimizer.zero_grad()
        
        try:
            # Assuming the model has a standard interface
            outputs = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=getattr(batch, 'edge_attr', None),
                batch=batch.batch
            )
            
            # Simple loss for monitoring
            if isinstance(outputs, dict):
                loss = sum(v.sum() for v in outputs.values() if isinstance(v, torch.Tensor))
            else:
                loss = outputs.sum()
            
            # Backward pass
            loss.backward()
            
            # Analyze gradients
            analysis = analyzer.analyze_gradient_flow()
            diagnosis = analyzer.diagnose_gradient_issues(analysis)
            
            monitoring_results['batch_analyses'].append({
                'batch_idx': batch_idx,
                'loss': loss.item(),
                'gradient_coverage': analysis['gradient_coverage_percent'],
                'avg_grad_norm': analysis.get('avg_grad_norm', 0),
                'max_grad_norm': analysis.get('max_grad_norm', 0),
                'issues': diagnosis['issues'],
                'warnings': diagnosis['warnings']
            })
            
            logger.info(f"Batch {batch_idx}: Loss={loss.item():.6f}, Coverage={analysis['gradient_coverage_percent']:.1f}%")
            
            # Take optimizer step
            optimizer.step()
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            monitoring_results['issues_detected'].append(f"Batch {batch_idx}: {str(e)}")
    
    # Generate final report
    report_path = analyzer.generate_report(os.path.join(output_dir, 'gradient_monitoring_report.json'))
    plot_paths = analyzer.create_visualization(os.path.join(output_dir, 'plots'))
    
    monitoring_results['report_path'] = report_path
    monitoring_results['plot_paths'] = plot_paths
    monitoring_results['final_analysis'] = analyzer.gradient_history[-1] if analyzer.gradient_history else {}
    
    logger.info(f"Gradient monitoring complete. Results saved to {output_dir}")
    return monitoring_results


def main():
    parser = argparse.ArgumentParser(description="Gradient Flow Monitor and Analyzer")
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    parser.add_argument('--config_path', type=str, help='Path to model configuration')
    parser.add_argument('--data_path', type=str, help='Path to sample data')
    parser.add_argument('--output_dir', type=str, default='gradient_analysis', help='Output directory')
    parser.add_argument('--monitor_training', action='store_true', help='Monitor during training steps')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to monitor')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.monitor_training and args.data_path:
        # Monitor during training
        # This would require loading the model and data
        logger.info("Training monitoring mode - implement based on your specific setup")
        
    else:
        # Static analysis mode
        if args.model_path and os.path.exists(args.model_path):
            # Load and analyze existing model
            logger.info(f"Loading model from {args.model_path}")
            model = torch.load(args.model_path, map_location='cpu')
            
            analyzer = DetailedGradientAnalyzer(model)
            
            # Run analysis (will show all parameters have no gradients since no backward pass)
            analysis = analyzer.analyze_gradient_flow()
            diagnosis = analyzer.diagnose_gradient_issues(analysis)
            
            # Generate report
            report_path = analyzer.generate_report(os.path.join(args.output_dir, 'static_analysis_report.json'))
            
            logger.info(f"Static analysis complete. Report saved to {report_path}")
            logger.info(f"Model has {analysis['total_parameters']} parameters")
            logger.info(f"Note: Static analysis shows no gradients (expected - no backward pass performed)")
            
        else:
            logger.error("Please provide --model_path for static analysis or --data_path for training monitoring")


if __name__ == '__main__':
    main()