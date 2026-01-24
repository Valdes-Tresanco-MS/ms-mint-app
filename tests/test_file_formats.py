#!/usr/bin/env python
"""Test multi-format file reading for targets/metadata."""

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 70)
    print("Testing Multi-Format File Reading")
    print("=" * 70)
    
    # Test data
    test_df = pd.DataFrame({
        'peak_label': ['Glucose', 'Serine', 'Alanine'],
        'mz_mean': [180.063, 106.050, 90.055],
        'rt': [120.5, 150.2, 95.0],
    })
    
    results = {}
    
    # Test 1: CSV
    print("\n1. Testing CSV (.csv)...")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_df.to_csv(f.name, index=False)
            temp_file = f.name
        
        from ms_mint_app.tools import read_tabular_file
        df = read_tabular_file(temp_file)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        print(f"   ✅ CSV works! Read {len(df)} rows")
        results['csv'] = 'OK'
        Path(temp_file).unlink()
    except Exception as e:
        print(f"   ❌ CSV failed: {e}")
        results['csv'] = str(e)
    
    # Test 2: TSV
    print("\n2. Testing TSV (.tsv)...")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            test_df.to_csv(f.name, index=False, sep='\t')
            temp_file = f.name
        
        df = read_tabular_file(temp_file)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        print(f"   ✅ TSV works! Read {len(df)} rows")
        results['tsv'] = 'OK'
        Path(temp_file).unlink()
    except Exception as e:
        print(f"   ❌ TSV failed: {e}")
        results['tsv'] = str(e)
    
    # Test 3: TXT (tab-delimited)
    print("\n3. Testing TXT with tabs (.txt)...")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_df.to_csv(f.name, index=False, sep='\t')
            temp_file = f.name
        
        df = read_tabular_file(temp_file)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        print(f"   ✅ TXT (tab) works! Read {len(df)} rows")
        results['txt_tab'] = 'OK'
        Path(temp_file).unlink()
    except Exception as e:
        print(f"   ❌ TXT (tab) failed: {e}")
        results['txt_tab'] = str(e)
    
    # Test 4: TXT (comma-delimited)
    print("\n4. Testing TXT with commas (.txt)...")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_df.to_csv(f.name, index=False)
            temp_file = f.name
        
        df = read_tabular_file(temp_file)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        print(f"   ✅ TXT (comma) works! Read {len(df)} rows")
        results['txt_comma'] = 'OK'
        Path(temp_file).unlink()
    except Exception as e:
        print(f"   ❌ TXT (comma) failed: {e}")
        results['txt_comma'] = str(e)
    
    # Test 5: Excel .xlsx (modern)
    print("\n5. Testing Excel .xlsx...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            test_df.to_excel(f.name, index=False, engine='openpyxl')
            temp_file = f.name
        
        df = read_tabular_file(temp_file)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        print(f"   ✅ Excel .xlsx works! Read {len(df)} rows")
        results['xlsx'] = 'OK'
        Path(temp_file).unlink()
    except ImportError as e:
        print(f"   ⚠️  Excel .xlsx MISSING DEPENDENCY: {e}")
        results['xlsx'] = f'MISSING: {e}'
    except Exception as e:
        print(f"   ❌ Excel .xlsx failed: {e}")
        results['xlsx'] = str(e)
    
    # Test 6: Excel .xls (legacy)
    print("\n6. Testing Excel .xls (legacy)...")
    try:
        with tempfile.NamedTemporaryFile(suffix='.xls', delete=False) as f:
            test_df.to_excel(f.name, index=False, engine='xlwt')
            temp_file = f.name
        
        df = read_tabular_file(temp_file)
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        print(f"   ✅ Excel .xls works! Read {len(df)} rows")
        results['xls'] = 'OK'
        Path(temp_file).unlink()
    except ImportError as e:
        print(f"   ⚠️  Excel .xls MISSING DEPENDENCY: {e}")
        results['xls'] = f'MISSING: {e}'
    except Exception as e:
        print(f"   ❌ Excel .xls failed: {e}")
        results['xls'] = str(e)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_ok = True
    for fmt, status in results.items():
        emoji = "✅" if status == 'OK' else "❌"
        print(f"{emoji} {fmt}: {status}")
        if status != 'OK':
            all_ok = False
    
    if all_ok:
        print("\n🎉 All formats work! No additional dependencies needed.")
    else:
        print("\n⚠️  Some formats need additional dependencies.")
        print("\nTo fix, run:")
        if 'MISSING' in results.get('xlsx', ''):
            print("  pip install openpyxl")
        if 'MISSING' in results.get('xls', ''):
            print("  pip install xlrd xlwt")
    
    sys.exit(0 if all_ok else 1)
