#!/usr/bin/env python3
"""
Test suite for the simple EarlyStopping class in train_alternating_optimized.py

This tests the basic early stopping mechanism implemented directly in the training script,
focusing on validation plateau detection and best score tracking.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import the EarlyStopping class from the training script
from scripts.train_alternating_optimized import EarlyStopping


class TestSimpleEarlyStopping(unittest.TestCase):
    """Test suite for the simple EarlyStopping implementation."""
    
    def test_initialization(self):
        """Test EarlyStopping initialization with default and custom parameters."""
        # Test default initialization
        early_stopping = EarlyStopping()
        self.assertEqual(early_stopping.patience, 7)
        self.assertEqual(early_stopping.min_delta, 0.001)
        self.assertEqual(early_stopping.best_loss, float('inf'))
        self.assertEqual(early_stopping.counter, 0)
        self.assertEqual(early_stopping.early_stop, False)
        self.assertEqual(early_stopping.best_step, 0)
        
        # Test custom initialization
        early_stopping_custom = EarlyStopping(patience=5, min_delta=0.01)
        self.assertEqual(early_stopping_custom.patience, 5)
        self.assertEqual(early_stopping_custom.min_delta, 0.01)
    
    def test_early_stopping_with_validation_plateau(self):
        """
        Test early stopping mechanism with validation plateau scenario.
        
        This test simulates the specific scenario requested:
        - Validation losses: [10, 8, 6, 6, 6, 6, 6, 6]
        - Patience = 5
        - Should trigger early stopping after 5 consecutive non-improvements
        """
        # Initialize with patience=5 as requested
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        
        # Validation loss sequence: first improves, then plateaus
        validation_losses = [10, 8, 6, 6, 6, 6, 6, 6]
        
        results = []
        for step, val_loss in enumerate(validation_losses):
            should_stop = early_stopping(val_loss, step)
            results.append(should_stop)
            
            # Break if early stopping is triggered
            if should_stop:
                break
        
        # Verify early stopping behavior
        # First few steps should not trigger stopping (improvements)
        self.assertFalse(results[0])  # Step 0: 10 -> new best
        self.assertFalse(results[1])  # Step 1: 8 -> improvement
        self.assertFalse(results[2])  # Step 2: 6 -> improvement
        
        # Next steps should not trigger stopping initially (patience not exceeded)
        self.assertFalse(results[3])  # Step 3: 6 -> no improvement, counter=1
        self.assertFalse(results[4])  # Step 4: 6 -> no improvement, counter=2
        self.assertFalse(results[5])  # Step 5: 6 -> no improvement, counter=3
        self.assertFalse(results[6])  # Step 6: 6 -> no improvement, counter=4
        
        # At step 7, patience should be exceeded (counter=5)
        self.assertTrue(results[7])   # Step 7: 6 -> no improvement, counter=5, trigger stop
        
        # Verify early_stop flag is set
        self.assertTrue(early_stopping.early_stop)
        
        # Verify best score is correctly recorded
        self.assertEqual(early_stopping.best_loss, 6.0)  # Minimum achieved
        self.assertEqual(early_stopping.best_step, 2)    # Step where minimum was achieved
    
    def test_best_score_tracking(self):
        """Test that best score is correctly tracked."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        # Sequence with improvements and plateaus
        test_sequence = [
            (5.0, 0),   # Initial best
            (3.0, 1),   # New best
            (2.5, 2),   # New best
            (2.6, 3),   # No improvement
            (2.4, 4),   # New best
            (2.5, 5),   # No improvement
        ]
        
        for val_loss, step in test_sequence:
            early_stopping(val_loss, step)
        
        # Best loss should be 2.4 from step 4
        self.assertEqual(early_stopping.best_loss, 2.4)
        self.assertEqual(early_stopping.best_step, 4)
    
    def test_min_delta_threshold(self):
        """Test that min_delta threshold is respected for improvements."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.1)
        
        # Small improvements below min_delta should not reset counter
        early_stopping(1.0, 0)   # Initial best
        self.assertEqual(early_stopping.counter, 0)
        
        early_stopping(0.95, 1)  # Improvement of 0.05 < min_delta (0.1)
        self.assertEqual(early_stopping.counter, 1)  # Should increment counter
        self.assertEqual(early_stopping.best_loss, 1.0)  # Should not update best
        
        early_stopping(0.85, 2)  # Improvement of 0.15 > min_delta (0.1)
        self.assertEqual(early_stopping.counter, 0)   # Should reset counter
        self.assertEqual(early_stopping.best_loss, 0.85)  # Should update best
    
    def test_patience_parameter(self):
        """Test different patience values."""
        # Test with patience=1
        early_stopping_impatient = EarlyStopping(patience=1, min_delta=0.001)
        early_stopping_impatient(1.0, 0)  # Set initial
        should_stop = early_stopping_impatient(1.0, 1)  # No improvement
        self.assertTrue(should_stop)  # Should stop immediately
        
        # Test with patience=10
        early_stopping_patient = EarlyStopping(patience=10, min_delta=0.001)
        early_stopping_patient(1.0, 0)  # Set initial
        
        # Should not stop for 9 non-improvements
        for i in range(1, 10):
            should_stop = early_stopping_patient(1.0, i)
            self.assertFalse(should_stop)
        
        # Should stop on the 10th non-improvement
        should_stop = early_stopping_patient(1.0, 10)
        self.assertTrue(should_stop)
    
    def test_counter_reset_on_improvement(self):
        """Test that counter resets when improvement is detected."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        early_stopping(1.0, 0)   # Initial
        self.assertEqual(early_stopping.counter, 0)
        
        early_stopping(1.0, 1)   # No improvement
        self.assertEqual(early_stopping.counter, 1)
        
        early_stopping(1.0, 2)   # No improvement
        self.assertEqual(early_stopping.counter, 2)
        
        early_stopping(0.9, 3)   # Improvement - should reset counter
        self.assertEqual(early_stopping.counter, 0)
        self.assertEqual(early_stopping.best_loss, 0.9)
        self.assertEqual(early_stopping.best_step, 3)
    
    def test_edge_cases(self):
        """Test edge cases like identical losses and extreme values."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.0)
        
        # Test with identical losses (should trigger early stopping)
        early_stopping(1.0, 0)
        self.assertFalse(early_stopping(1.0, 1))  # counter=1
        self.assertTrue(early_stopping(1.0, 2))   # counter=2, trigger stop
        
        # Test with very small losses
        early_stopping_small = EarlyStopping(patience=2, min_delta=1e-10)
        early_stopping_small(1e-6, 0)
        early_stopping_small(1e-7, 1)  # Small improvement
        self.assertEqual(early_stopping_small.counter, 0)  # Should reset
        
        # Test with zero loss
        early_stopping_zero = EarlyStopping(patience=2, min_delta=0.001)
        early_stopping_zero(0.1, 0)
        early_stopping_zero(0.0, 1)  # Perfect loss
        self.assertEqual(early_stopping_zero.best_loss, 0.0)
        self.assertEqual(early_stopping_zero.counter, 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)