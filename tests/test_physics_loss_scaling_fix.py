#!/usr/bin/env python3
"""
Test script to verify the physics_loss scaling removal is successful.
Ensures the compute_losses function is intact and the artificial scaling is gone.
"""

import sys
import ast
import re

def verify_physics_loss_fix():
    """Verify that the artificial physics_loss scaling has been removed."""
    print("🔍 Verifying Physics Loss Scaling Removal...")
    
    script_path = "/home/saketh/MoML-CA/scripts/train_alternating_optimized.py"
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
            
        # Check that artificial scaling is removed
        artificial_scaling_patterns = [
            r'losses\["physics_loss"\]\s*=\s*losses\["physics_loss"\]\s*\*\s*0\.01',
            r'physics_loss.*\*\s*0\.01.*Even more reduced impact'
        ]
        
        for pattern in artificial_scaling_patterns:
            if re.search(pattern, content):
                print(f"❌ Found artificial scaling pattern: {pattern}")
                return False
                
        print("✅ No artificial physics_loss scaling found")
        
        # Verify legitimate curriculum weighting is still present
        if "phase_weights['physics_loss']" in content:
            print("✅ Legitimate curriculum weighting preserved")
        else:
            print("❌ Curriculum weighting missing")
            return False
            
        # Verify sanitize_loss is still applied
        if 'losses["physics_loss"] = sanitize_loss(physics_loss_raw' in content:
            print("✅ Physics loss sanitization preserved")
        else:
            print("❌ Physics loss sanitization missing")
            return False
            
        # Test syntax by parsing
        try:
            ast.parse(content)
            print("✅ Python syntax is valid")
        except SyntaxError as e:
            print(f"❌ Syntax error: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Physics Loss Scaling Fix...")
    
    success = verify_physics_loss_fix()
    
    if success:
        print("\n🎉 SUCCESS: Physics loss scaling fix verified!")
        print("✨ The PIMEH head will now receive proper gradient signals")
        sys.exit(0)
    else:
        print("\n💥 FAILURE: Physics loss fix verification failed")
        sys.exit(1)