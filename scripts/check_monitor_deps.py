#!/usr/bin/env python3
"""
Check dependencies for the professional training monitor.
"""

import sys

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('rich', 'rich'),
        ('psutil', 'psutil')
    ]
    
    missing = []
    available = []
    
    for module_name, import_name in required_modules:
        try:
            __import__(import_name)
            available.append(module_name)
            print(f"✅ {module_name}")
        except ImportError:
            missing.append(module_name)
            print(f"❌ {module_name} - MISSING")
    
    print(f"\n📊 Summary:")
    print(f"✅ Available: {len(available)}")
    print(f"❌ Missing: {len(missing)}")
    
    if missing:
        print(f"\n🔧 To install missing dependencies:")
        print(f"pip install {' '.join(missing)}")
        return False
    else:
        print(f"\n🎯 All dependencies available! Ready to run the monitor.")
        return True

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)