
import pytest
import numpy as np
import base64
import zlib
import struct
import tempfile
import os
from pathlib import Path
from ms_mint_app.tools import iter_mzml_fast, iter_mzxml_fast, _decode_binary_mzml, _decode_peaks_optimized

# Helpers to create valid XML content
def encode_array(arr, dtype, compression=None):
    if dtype == np.float32:
        raw = arr.astype(np.float32).tobytes()
    else:
        raw = arr.astype(np.float64).tobytes()
    
    if compression == 'zlib':
        raw = zlib.compress(raw)
    
    return base64.b64encode(raw).decode('ascii')

def create_mzml(filename, scans):
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">\n')
        f.write('  <run id="run1">\n')
        f.write('    <spectrumList count="{0}">\n'.format(len(scans)))
        
        for i, scan in enumerate(scans):
            f.write(f'      <spectrum index="{i}" id="scan={scan["num"]}" defaultArrayLength="{len(scan["mz"])}">\n')
            f.write('        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{0}"/>\n'.format(scan.get("msLevel", 1)))
            if scan.get("polarity") == "Positive":
                f.write('        <cvParam cvRef="MS" accession="MS:1000130" name="positive scan" value=""/>\n')
            elif scan.get("polarity") == "Negative":
                f.write('        <cvParam cvRef="MS" accession="MS:1000129" name="negative scan" value=""/>\n')
                
            f.write('        <scanList count="1">\n')
            f.write('          <cvParam cvRef="MS" accession="MS:1000795" name="no combination" value=""/>\n')
            f.write('          <scan>\n')
            f.write('            <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{0}" unitName="second"/>\n'.format(scan.get("rt", 0)))
            f.write('          </scan>\n')
            f.write('        </scanList>\n')
            
            f.write('        <binaryDataArrayList count="2">\n')
            
            # m/z array
            mz_enc = encode_array(scan["mz"], np.float64, 'zlib')
            f.write('          <binaryDataArray encodedLength="0">\n')
            f.write('            <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>\n')
            f.write('            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>\n')
            f.write('            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>\n')
            f.write(f'            <binary>{mz_enc}</binary>\n')
            f.write('          </binaryDataArray>\n')
            
            # intensity array
            int_enc = encode_array(scan["intensity"], np.float64, 'zlib')
            f.write('          <binaryDataArray encodedLength="0">\n')
            f.write('            <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>\n')
            f.write('            <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>\n')
            f.write('            <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>\n')
            f.write(f'            <binary>{int_enc}</binary>\n')
            f.write('          </binaryDataArray>\n')
            
            f.write('        </binaryDataArrayList>\n')
            f.write('      </spectrum>\n')
            
        f.write('    </spectrumList>\n')
        f.write('  </run>\n')
        f.write('</mzML>\n')

def create_mzxml(filename, scans):
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        f.write('<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2" version="3.2">\n')
        f.write('  <msRun scanCount="{0}">\n'.format(len(scans)))
        
        for scan in scans:
            pol = "+" if scan.get("polarity") == "Positive" else "-"
            f.write(f'    <scan num="{scan["num"]}" msLevel="{scan.get("msLevel", 1)}" retentionTime="PT{scan.get("rt", 0)}S" polarity="{pol}" peaksCount="{len(scan["mz"])}">\n')
            
            # Encode peaks in pairs (mz, intensity) floats usually 32-bit network order in mzXML by default but we can specify
            # Let's use 64-bit to match mzML test above for simplicity
            dt = np.dtype([("mz", ">f8"), ("intensity", ">f8")])
            arr = np.empty(len(scan["mz"]), dtype=dt)
            arr["mz"] = scan["mz"]
            arr["intensity"] = scan["intensity"]
            
            raw = arr.tobytes()
            enc = base64.b64encode(zlib.compress(raw)).decode('ascii')
            
            f.write(f'      <peaks compressionType="zlib" compressedLen="0" precision="64" byteOrder="network" pairOrder="m/z-int">{enc}</peaks>\n')
            f.write('    </scan>\n')
            
        f.write('  </msRun>\n')
        f.write('</mzXML>\n')

@pytest.fixture
def synthetic_data(tmp_path):
    mzml_file = tmp_path / "test.mzML"
    mzxml_file = tmp_path / "test.mzXML"
    
    scans = [
        {"num": 1, "msLevel": 1, "rt": 10.0, "polarity": "Positive", "mz": np.array([100.1, 200.2]), "intensity": np.array([1000.0, 500.0])},
        {"num": 2, "msLevel": 1, "rt": 20.0, "polarity": "Negative", "mz": np.array([150.5, 250.6]), "intensity": np.array([800.0, 400.0])},
    ]
    
    create_mzml(mzml_file, scans)
    create_mzxml(mzxml_file, scans)
    
    return mzml_file, mzxml_file, scans

def test_iter_mzml_fast(synthetic_data):
    mzml_path, _, expected_scans = synthetic_data
    parsed = list(iter_mzml_fast(str(mzml_path)))
    
    assert len(parsed) == len(expected_scans)
    
    for p, e in zip(parsed, expected_scans):
        assert p["num"] == e["num"]
        assert p["msLevel"] == e["msLevel"]
        assert abs(p["retentionTime"] - e["rt"]) < 1e-4
        assert p["polarity"] == e["polarity"]
        assert np.allclose(p["m/z array"], e["mz"])
        assert np.allclose(p["intensity array"], e["intensity"])

def test_iter_mzxml_fast(synthetic_data):
    _, mzxml_path, expected_scans = synthetic_data
    parsed = list(iter_mzxml_fast(str(mzxml_path)))
    
    assert len(parsed) == len(expected_scans)
    
    for p, e in zip(parsed, expected_scans):
        assert p["num"] == e["num"]
        assert p["msLevel"] == e["msLevel"]
        assert abs(p["retentionTime"] - e["rt"]) < 1e-4
        # mzXML polarity is slightly implementation dependent in extraction but we expect consistent results
        assert p["polarity"] == e["polarity"]
        assert np.allclose(p["m/z array"], e["mz"], atol=1e-4) # 32-bit vs 64-bit precision issues may arise
        assert np.allclose(p["intensity array"], e["intensity"], atol=1e-4)

def test_parsers_consistency(synthetic_data):
    mzml_path, mzxml_path, _ = synthetic_data
    
    parsed_mzml = list(iter_mzml_fast(str(mzml_path)))
    parsed_mzxml = list(iter_mzxml_fast(str(mzxml_path)))
    
    assert len(parsed_mzml) == len(parsed_mzxml)
    
    for m, x in zip(parsed_mzml, parsed_mzxml):
        assert m["num"] == x["num"]
        assert m["msLevel"] == x["msLevel"]
        assert abs(m["retentionTime"] - x["retentionTime"]) < 1e-3
        assert m["polarity"] == x["polarity"]
        assert np.allclose(m["m/z array"], x["m/z array"], atol=1e-4)
        assert np.allclose(m["intensity array"], x["intensity array"], atol=1e-4)

def test_real_files_parsing():
    """Test parsing of real files from test_MS1 directory."""
    # Use relative path dynamically based on this test file's location
    test_dir = Path(__file__).parent / "test_MS1" / "ms-files"
    mzml_path = test_dir / "test.mzML"
    mzxml_path = test_dir / "test.mzXML"
    
    if not mzml_path.exists() or not mzxml_path.exists():
        pytest.skip(f"Real test files not found in {test_dir}")
        
    print(f"\nTesting real file: {mzml_path}")
    mzml_scans = list(iter_mzml_fast(str(mzml_path)))
    print(f"  Parsed {len(mzml_scans)} scans from mzML")
    assert len(mzml_scans) > 0, "No scans parsed from real mzML file"
    
    print(f"Testing real file: {mzxml_path}")
    mzxml_scans = list(iter_mzxml_fast(str(mzxml_path)))
    print(f"  Parsed {len(mzxml_scans)} scans from mzXML")
    assert len(mzxml_scans) > 0, "No scans parsed from real mzXML file"
    
    # Basic check on first scan structure
    s = mzml_scans[0]
    assert "m/z array" in s
    assert "intensity array" in s
    assert isinstance(s["m/z array"], np.ndarray)
    
    # Consistency check between formats if they are supposed to be same data
    # (assuming test.mzML and test.mzXML are same run)
    if len(mzml_scans) == len(mzxml_scans):
         s_xml = mzxml_scans[0]
         assert abs(s["retentionTime"] - s_xml["retentionTime"]) < 1e-2

