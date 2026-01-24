#!/usr/bin/env python
"""Test polarity and ms_type handling improvements."""

import pandas as pd
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

if __name__ == "__main__":
    from ms_mint_app.tools import get_targets_v2
    
    print("=" * 70)
    print("Testing Polarity and MS Type Handling")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "Polarity not provided → NULL (optional)",
            "data": pd.DataFrame({
                "peak_label": ["Glucose"],
                "rt": [120.5],
            }),
            "expected": {
                "polarity": None,  # Should be NULL
                "ms_type": "ms1",
            }
        },
        {
            "name": "Polarity provided → Normalized",
            "data": pd.DataFrame({
                "peak_label": ["Glucose"],
                "rt": [120.5],
                "polarity": ["+"],
            }),
            "expected": {
                "polarity": "Positive",
                "ms_type": "ms1",
            }
        },
        {
            "name": "MS2: Has filterLine → ms_type=ms2",
            "data": pd.DataFrame({
                "peak_label": ["Fragment163"],
                "rt": [120.5],
                "filterLine": ["FTMS + p ESI Full ms2 163.06@hcd25.00"],
            }),
            "expected": {
                "ms_type": "ms2",
            }
        },
        {
            "name": "MS1: No filterLine → ms_type=ms1",
            "data": pd.DataFrame({
                "peak_label": ["Glucose"],
                "rt": [120.5],
            }),
            "expected": {
                "ms_type": "ms1",
            }
        },
        {
            "name": "Contradiction: Says ms2 but no filterLine → Corrected to ms1",
            "data": pd.DataFrame({
                "peak_label": ["BadTarget"],
                "rt": [120.5],
                "ms_type": ["ms2"],  # User says ms2
                # No filterLine!
            }),
            "expected": {
                "ms_type": "ms1",  # Should be corrected
            }
        },
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 70)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_case['data'].to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            targets_df, _, _, _ = get_targets_v2([temp_file])
            row = targets_df.iloc[0]
            
            success = True
            for key, expected_val in test_case['expected'].items():
                actual_val = row.get(key)
                
                # Handle NULL comparison
                if expected_val is None:
                    if pd.isna(actual_val):
                        print(f"  ✓ {key}: NULL (as expected)")
                    else:
                        print(f"  ❌ {key}: expected NULL, got '{actual_val}'")
                        success = False
                else:
                    if actual_val == expected_val:
                        print(f"  ✓ {key}: {actual_val}")
                    else:
                        print(f"  ❌ {key}: expected '{expected_val}', got '{actual_val}'")
                        success = False
            
            if success:
                print(f"✅ PASSED")
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    sys.exit(0 if failed == 0 else 1)
