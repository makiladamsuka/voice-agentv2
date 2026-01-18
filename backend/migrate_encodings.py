#!/usr/bin/env python3
"""
Migration script to fix numpy version incompatibility in face encodings pickle file.
Converts pickle file from numpy 2.x format to numpy 1.x format.
"""

import pickle
import sys
from pathlib import Path

def migrate_encodings():
    encodings_path = Path(__file__).parent / "known_faces" / "encodings.pkl"
    backup_path = encodings_path.with_suffix('.pkl.backup')
    
    if not encodings_path.exists():
        print("❌ No encodings.pkl file found")
        return False
    
    print(f"📦 Found encodings file: {encodings_path}")
    
    # Try loading with custom unpickler that handles numpy._core
    try:
        import numpy as np
        import importlib
        
        # Create a custom unpickler that redirects numpy._core to numpy.core
        class NumpyCompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle numpy 2.x -> 1.x compatibility
                if module.startswith('numpy._core'):
                    module = module.replace('numpy._core', 'numpy.core')
                elif module == 'numpy.core.multiarray':
                    module = 'numpy.core.multiarray'
                return super().find_class(module, name)
        
        # Load the data
        with open(encodings_path, 'rb') as f:
            unpickler = NumpyCompatUnpickler(f)
            data = unpickler.load()
        
        print(f"✅ Successfully loaded encodings: {len(data)} people")
        print(f"   Names: {list(data.keys())}")
        
        # Backup original
        import shutil
        shutil.copy2(encodings_path, backup_path)
        print(f"✅ Backed up original to: {backup_path}")
        
        # Save with current numpy version
        with open(encodings_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Re-saved encodings with current numpy version")
        print(f"   Current numpy version: {np.__version__}")
        
        # Verify it can be loaded normally now
        with open(encodings_path, 'rb') as f:
            verify_data = pickle.load(f)
        print(f"✅ Verification: File can be loaded normally")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  FACE ENCODINGS MIGRATION TOOL")
    print("=" * 60)
    print()
    
    success = migrate_encodings()
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("   You can now run the greeting agent without errors.")
    else:
        print("\n❌ Migration failed.")
        print("   You may need to re-enroll faces using: python enroll_face.py")
        sys.exit(1)
