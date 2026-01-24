#!/usr/bin/env python
"""Quick test for mz_width default."""

import pandas as pd
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

if __name__ == "__main__":
    from ms_mint_app.tools import get_targets_v2
    
    # Test: Target without mz_width
    test_data = pd.DataFrame({
        "peak_label": ["Glucose"],
        "rt": [120.5],
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        targets_df, _, _, _ = get_targets_v2([temp_file])
        row = targets_df.iloc[0]
        
        print("Testing mz_width default:")
        print(f"  Input: No mz_width specified")
        print(f"  Output: mz_width = {row['mz_width']}")
        print(f"  Expected: 10.0 ppm")
        
        if row['mz_width'] == 10.0:
            print("✅ PASSED: mz_width defaulted to 10.0 ppm")
        else:
            print(f"❌ FAILED: Expected 10.0, got {row['mz_width']}")
            sys.exit(1)
            
    finally:
        Path(temp_file).unlink(missing_ok=True)
    
    print("\n✓ All defaults working correctly:")
    print(f"  - mz_width: {row['mz_width']} ppm")
    print(f"  - polarity: {row['polarity']}")
    print(f"  - ms_type: {row['ms_type']}")
