import base64
import io
import logging
import math
import os
import random
import re
import tempfile
import zlib
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Iterator, Any, List

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from dash.exceptions import PreventUpdate
import pygixml
from scipy.ndimage import binary_opening

from .duckdb_manager import duckdb_connection
from .sample_metadata import GROUP_COLUMNS

logger = logging.getLogger(__name__)

# Supported file extensions for tabular data
SUPPORTED_TABULAR_EXTENSIONS = {'.csv', '.tsv', '.txt', '.xls', '.xlsx'}


def read_tabular_file(file_path: str | Path, dtype: dict = None, nrows: int = None) -> pd.DataFrame:
    """
    Read a tabular file into a DataFrame, supporting multiple formats:
    - CSV (.csv)
    - TSV (.tsv, .txt with tab separator)
    - Excel (.xls, .xlsx)
    
    Args:
        file_path: Path to the file
        dtype: Optional dict of column dtypes
        nrows: Optional number of rows to read (for preview)
    
    Returns:
        pd.DataFrame with file contents
    
    Raises:
        ValueError: If file extension is not supported
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    read_kwargs = {}
    if dtype:
        read_kwargs['dtype'] = dtype
    if nrows is not None:
        read_kwargs['nrows'] = nrows
    
    if ext == '.csv':
        return pd.read_csv(file_path, **read_kwargs)
    
    elif ext == '.tsv':
        return pd.read_csv(file_path, sep='\t', **read_kwargs)
    
    elif ext == '.txt':
        # Try to auto-detect delimiter (tab or comma)
        with open(file_path, 'r') as f:
            first_line = f.readline()
        sep = '\t' if '\t' in first_line else ','
        return pd.read_csv(file_path, sep=sep, **read_kwargs)
    
    elif ext in ('.xls', '.xlsx'):
        # Excel files - read first sheet by default
        # Note: nrows works differently for Excel - need to filter after
        excel_kwargs = {k: v for k, v in read_kwargs.items() if k != 'nrows'}
        df = pd.read_excel(file_path, **excel_kwargs)
        if nrows is not None:
            df = df.head(nrows)
        return df
    
    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: CSV (.csv), TSV (.tsv), TXT (.txt), Excel (.xls, .xlsx)"
        )


# Column mappings for external software formats
# Maps external column names (lowercase) to MINT column names
COLUMN_MAPPINGS = {
    # EL-MAVEN / Maven
    'compoundid': 'peak_label',
    'compound': 'peak_label',
    'compoundname': 'peak_label',
    'medmz': 'mz_mean',
    'parent': 'mz_mean',
    'medrt': 'rt',
    'expectedrt': 'rt',
    
    # MZmine
    'row m/z': 'mz_mean',
    'row retention time': 'rt',
    'compound name': 'peak_label',
    
    # Generic alternatives
    'name': 'peak_label',
    'target': 'peak_label',
    'target_name': 'peak_label',
    'precursor_mz': 'mz_mean',
    'precursormz': 'mz_mean',
    'retention_time': 'rt',
    'retentiontime': 'rt',
    'rtmin': 'rt_min',
    'rtmax': 'rt_max',
}

# Priority order for target columns that can come from multiple source columns.
# When multiple source columns are present, the first one found in this list wins.
# Keys are MINT target column names, values are lists of source column names in priority order.
PRIORITY_MAPPINGS = {
    'mz_mean': ['medmz', 'parent', 'row m/z', 'precursor_mz', 'precursormz'],
    'peak_label': ['compoundid', 'compound', 'compoundname', 'compound name', 'name', 'target', 'target_name'],
    'rt': ['medrt', 'expectedrt', 'row retention time', 'retention_time', 'retentiontime'],
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names from external software formats to MINT format.
    
    Supports automatic detection and mapping of columns from:
    - EL-MAVEN (compoundId, medMz, medRt) - RT is in minutes, converted to seconds
    - MZmine (row m/z, row retention time)
    - Generic alternatives (name, mz, rt, etc.)
    
    When multiple source columns could map to the same target (e.g., 'medmz' and 
    'parent' both mapping to 'mz_mean'), the PRIORITY_MAPPINGS determines which 
    source column takes precedence.
    
    Args:
        df: DataFrame with potentially non-standard column names
    
    Returns:
        DataFrame with normalized column names and units
    """
    new_columns = {}
    mapped_info = []
    needs_rt_conversion = False
    
    # Build a set of available columns (lowercase) for quick lookup
    available_cols_lower = {str(col).lower().strip(): col for col in df.columns}
    
    # First pass: determine which source columns to use for priority-based targets
    # This ensures we pick the highest-priority source when multiple are present
    priority_selected = {}  # mint_col -> source_col_lower
    for mint_col, priority_sources in PRIORITY_MAPPINGS.items():
        if mint_col in df.columns:
            # Target already exists in original form, skip
            continue
        for source_lower in priority_sources:
            if source_lower in available_cols_lower:
                priority_selected[mint_col] = source_lower
                break  # Use the first (highest priority) match
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        # Check if this column needs mapping
        if col_lower in COLUMN_MAPPINGS:
            mint_col = COLUMN_MAPPINGS[col_lower]
            
            # Skip if target column already exists in original DataFrame
            if mint_col in df.columns:
                continue
            
            # Skip if we already mapped another column to this target
            if mint_col in new_columns.values():
                continue
            
            # For priority-based columns, only use if this is the selected source
            if mint_col in priority_selected:
                if col_lower != priority_selected[mint_col]:
                    continue  # Not the priority source, skip
            
            new_columns[col] = mint_col
            mapped_info.append(f"'{col}' → '{mint_col}'")
            
            # EL-MAVEN medRt is in minutes - needs conversion to seconds
            if col_lower == 'medrt':
                needs_rt_conversion = True
    
    if mapped_info:
        # Detect source format based on mapped columns
        if needs_rt_conversion:
            logging.info(f"Detected EL-MAVEN format file")
        logging.info(f"Auto-mapped columns: {', '.join(mapped_info)}")
        df = df.rename(columns=new_columns)
    
    # Convert RT from minutes to seconds for EL-MAVEN files
    if needs_rt_conversion and 'rt' in df.columns:
        df['rt'] = df['rt'] * 60.0
        logging.info("Converted RT from minutes to seconds (EL-MAVEN uses minutes)")
        
        # Also convert rt_min and rt_max if they were mapped
        if 'rt_min' in df.columns:
            df['rt_min'] = df['rt_min'] * 60.0
        if 'rt_max' in df.columns:
            df['rt_max'] = df['rt_max'] * 60.0
    
    return df


_RT_SECONDS = re.compile(
    r"^P(?:T(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<m>\d+(?:\.\d+)?)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?)$",
    re.I
)


def today() -> str:
    """Return today's date in ISO format (YYYY-MM-DD). Used for filenames."""
    return date.today().isoformat()


def rt_to_seconds(val) -> float:
    """Converts retentionTime to seconds (if it comes as PT...); if it is already numeric, returns it as is."""
    if isinstance(val, (int, float)):
        return float(val)
    s = (val or "").strip()
    # If it is like "0.12345", treat as seconds
    try:
        return float(s)
    except ValueError:
        pass
    m = _RT_SECONDS.match(s)
    if not m:
        return 0.0
    h = float(m.group("h") or 0.0)
    mi = float(m.group("m") or 0.0)
    se = float(m.group("s") or 0.0)
    return h * 3600.0 + mi * 60.0 + se


def get_acquisition_datetime(file_path: str | Path) -> Optional[str]:
    """
    Extract acquisition datetime from mzML/mzXML file header.
    
    For mzML: reads startTimeStamp from <run> element
    For mzXML: reads startTime from <msRun> element or falls back to file mtime
    
    Returns ISO format datetime string or None if not found.
    """
    from datetime import datetime
    import pygixml
    
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    try:
        doc = pygixml.parse_file(str(file_path))
        root = doc.first_child()
        
        if suffix == ".mzml":
            # Look for <run startTimeStamp="...">
            runs = root.select_nodes("//*[local-name()='run']")
            for run_node in runs:
                run = run_node.node
                timestamp = run.attribute("startTimeStamp").value
                if timestamp:
                    # ISO 8601 format: 2024-01-15T10:30:00Z
                    return timestamp
        
        elif suffix == ".mzxml":
            # Look for <msRun startTime="..." or <msRun>...<startTime>
            ms_runs = root.select_nodes("//*[local-name()='msRun']")
            for run_node in ms_runs:
                run = run_node.node
                # Check attribute first
                start_time = run.attribute("startTime").value
                if start_time and not start_time.startswith('PT'):
                   return start_time
                # Check child element
                child = run.first_child()
                while child and not child.is_null():
                    if child.name.endswith("startTime"):
                        start_time = child.text(True, "")
                        if start_time and not start_time.startswith('PT'):
                            return start_time
                    child = child.next_sibling()
    except Exception as e:
        logger.debug(f"Could not extract acquisition datetime from {file_path.name}: {e}")
    
    # Fallback: use file modification time
    try:
        mtime = file_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        pass
    
    return None



def _decode_peaks_optimized(attrs: Dict[str, str], text: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    KEY OPTIMIZATION: Decodes mz and intensity in A SINGLE operation
    using structured arrays (like pyteomics).

    This is ~2-3x faster than decoding separately.
    """
    if not text:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Determine dtype based on precision
    dt = np.float32 if attrs.get("precision") == "32" else np.float64

    # KEY: Create structured dtype for both arrays (mz, intensity)
    # byteorder '>' = big-endian (network byte order)
    endian = ">" if attrs.get("byteOrder") in ("network", "big") else "<"
    dtype = np.dtype([("mz", dt), ("intensity", dt)]).newbyteorder(endian)

    # Decode base64
    raw = base64.b64decode(text)

    # Decompress if necessary
    if attrs.get("compressionType") == "zlib":
        raw = zlib.decompress(raw)

    # SINGLE conversion from bytes to arrays
    arr = np.frombuffer(raw, dtype=dtype)

    # Extract fields from structured array (no copy, only views)
    return arr["mz"], arr["intensity"]



# def iter_mzxml_fast(path: str | Path, *, decode_binary: bool = True) -> Iterator[Dict[str, Any]]:
#     """
#     Legacy lxml-based iterator for mzXML files.
#     Kept for reference.
#     """
#     from lxml import etree
#     path = Path(path)
#     context = etree.iterparse(path.as_posix(), events=("start", "end"), remove_comments=True, huge_tree=False)
#     _, root = next(context)
#     current: Dict[str, Any] = {}
#     
#     for ev, elem in context:
#         tag = elem.tag
#         if '}' in tag:
#             tag = tag.rsplit("}", 1)[-1]
#             
#         if ev == "start" and tag == "scan":
#             attribs = elem.attrib
#             pol = attribs.get("polarity")
#             polarity = "Positive" if pol == "+" else ("Negative" if pol == "-" else None)
#             
#             current = {
#                 "num": int(attribs.get("num", "0")),
#                 "msLevel": int(attribs.get("msLevel", "1")),
#                 "retentionTime": rt_to_seconds(attribs.get("retentionTime", "0")),
#                 "polarity": polarity,
#                 "filterLine": attribs.get("filterLine"),
#                 "precursorMz": None,
#                 "m/z array": np.array([], dtype=np.float32), 
#                 "intensity array": np.array([], dtype=np.float32), 
#             }
#             
#         elif ev == "end" and tag == "precursorMz":
#             try:
#                 current["precursorMz"] = float(elem.text)
#             except (ValueError, TypeError):
#                 pass
#                 
#         elif ev == "end" and tag == "peaks":
#             if decode_binary:
#                 mz, it = _decode_peaks_optimized(elem.attrib, (elem.text or "").strip() or None)
#                 current["m/z array"] = mz
#                 current["intensity array"] = it
#                 
#         elif ev == "end" and tag == "scan":
#             yield current
#             elem.clear()
#             root.clear()

def iter_mzxml_fast(path: str | Path, *, decode_binary: bool = True) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    # pygixml implementation
    doc = pygixml.parse_file(str(path))
    
    # Select all scan elements using root relative xpath
    # Use first_child() to get the root element
    try:
        scans = doc.first_child().select_nodes("//*[local-name()='scan']")
    except Exception:
        # Fallback or error handling if file is empty/invalid
        return

    # Use a shared list for reusing objects if optimizing further, but dict creation is fine
    
    for scan in scans:
        scan = scan.node
        current = {
            "num": int(scan.attribute("num").value or "0"),
            "msLevel": int(scan.attribute("msLevel").value or "0"),
            "retentionTime": rt_to_seconds(scan.attribute("retentionTime").value),
            "polarity": None,
            "filterLine": scan.attribute("filterLine").value,
            "precursorMz": None,
            "m/z array": np.array([], dtype=np.float32), 
            "intensity array": np.array([], dtype=np.float32)
        }
        
        pol = scan.attribute("polarity").value
        if pol == "+":
            current["polarity"] = "Positive"
        elif pol == "-":
            current["polarity"] = "Negative"
            
        # Iterate children
        child = scan.first_child()
        while child and not child.is_null():
            tag = child.name
            
            if tag.endswith("precursorMz"):
                try:
                    current["precursorMz"] = float(child.text(True, ""))
                except ValueError:
                    pass
            elif tag.endswith("peaks"):
                if decode_binary:
                    # Attributes for decoding
                    precision = child.attribute("precision").value
                    byteOrder = child.attribute("byteOrder").value
                    compressionType = child.attribute("compressionType").value
                    
                    attrs = {
                        "precision": precision,
                        "byteOrder": byteOrder, 
                        "compressionType": compressionType
                    }
                    text = child.text(True, "")
                    
                    mz, inten = _decode_peaks_optimized(attrs, text)
                    current["m/z array"] = mz
                    current["intensity array"] = inten
                else:
                    # For non-decoded, we'd need to reconstruct the dict expected by callers
                    # But the optimized decoder is default. 
                    # If decode_binary is False existing code returned "peaks": {"attrs": ..., "text": ...}
                    current["peaks"] = {
                        "attrs": {
                            "precision": child.attribute("precision").value,
                            "byteOrder": child.attribute("byteOrder").value,
                            "compressionType": child.attribute("compressionType").value
                        }, 
                        "text": child.text(True, "")
                    }

            # ELMAVEN-like extra logic handled in loop in original, but here we construct current then yield
            # The original logic had `elif ev == "end" and tag == "scan":` block for ELMAVEN
            # We can do it here after parsing children
            
            child = child.next_sibling
            
        # ELMAVEN logic
        if current.get("msLevel") == 2:
            pol_str = current.get("polarity") or ""
            prec = current.get("precursorMz")
            mz_arr = current.get("m/z array", [])
            mz0 = float(mz_arr[0]) if len(mz_arr) > 0 else None
            if prec is not None and mz0 is not None:
                current["filterLine_ELMAVEN"] = f"{pol_str} {prec:.3f} [{mz0:.3f}]"

        yield current
        # doc.reset() or element clearing not strictly needed with pygixml struct parsing as it loads dom
        # but pygixml is fast. If memory is issue we might need iterative parser but pygixml is DOM-like.
        # The benchmark showed it's fine.


# =============================================================================
# CV Param Accessions (controlled vocabulary for mzML parsing)
# =============================================================================
CV_32BIT_FLOAT = "MS:1000521"
CV_64BIT_FLOAT = "MS:1000523"
CV_ZLIB_COMPRESSION = "MS:1000574"
CV_NO_COMPRESSION = "MS:1000576"
CV_MZ_ARRAY = "MS:1000514"
CV_INTENSITY_ARRAY = "MS:1000515"
CV_MS_LEVEL = "MS:1000511"
CV_POSITIVE_SCAN = "MS:1000130"
CV_NEGATIVE_SCAN = "MS:1000129"
CV_SCAN_START_TIME = "MS:1000016"


def _decode_binary_mzml(
    binary_text: str,
    is_64bit: bool = True,
    is_compressed: bool = True
) -> np.ndarray:
    """
    Decode a mzML binary data array.

    Args:
        binary_text: Base64-encoded binary string
        is_64bit: True for 64-bit float, False for 32-bit
        is_compressed: True if zlib compressed

    Returns:
        numpy array of decoded values
    """
    if not binary_text or not binary_text.strip():
        return np.array([], dtype=np.float64 if is_64bit else np.float32)

    # Base64 decode
    raw = base64.b64decode(binary_text.strip())

    # Decompress if needed
    if is_compressed:
        raw = zlib.decompress(raw)

    # Convert to numpy array
    dtype = np.float64 if is_64bit else np.float32
    return np.frombuffer(raw, dtype=dtype)


def _encode_binary_mzml(
    data: np.ndarray,
    is_64bit: bool = True,
    compress: bool = True
) -> str:
    """
    Encode a numpy array to mzML binary format (base64 + optional zlib).

    Args:
        data: numpy array of m/z or intensity values
        is_64bit: True for 64-bit float, False for 32-bit
        compress: True to apply zlib compression

    Returns:
        Base64-encoded string
    """
    dtype = np.float64 if is_64bit else np.float32
    arr = np.asarray(data, dtype=dtype)
    raw = arr.tobytes()

    if compress:
        raw = zlib.compress(raw)

    return base64.b64encode(raw).decode('ascii')


def write_mzml_from_spectra(
    spectra: List[Dict[str, Any]],
    output_path: str | Path,
    polarity: str = "Positive"
) -> None:
    """
    Write spectra to an mzML file using pure lxml.

    This is a lightweight mzML writer that produces files compatible with
    Asari and other mzML readers. Uses zlib compression and 64-bit floats.

    Args:
        spectra: List of spectrum dicts with keys:
            - scan_id: int (scan number)
            - rt: float (retention time in seconds)
            - mz_array: np.ndarray (m/z values)
            - intensity_array: np.ndarray (intensity values)
        output_path: Path to output mzML file
        polarity: "Positive" or "Negative"

    Example:
        >>> spectra = [
        ...     {"scan_id": 1, "rt": 0.5, "mz_array": np.array([100.0, 200.0]),
        ...      "intensity_array": np.array([1000.0, 2000.0])}
        ... ]
        >>> write_mzml_from_spectra(spectra, "output.mzML", polarity="Positive")
    """
    from lxml import etree

    output_path = Path(output_path)

    # mzML namespace
    NSMAP = {
        None: "http://psi.hupo.org/ms/mzml",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    }

    # Root element
    root = etree.Element("mzML", nsmap=NSMAP)
    root.set("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation",
             "http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd")
    root.set("version", "1.1.0")

    # CV list (required by schema)
    cv_list = etree.SubElement(root, "cvList", count="2")
    etree.SubElement(cv_list, "cv", id="MS", fullName="Proteomics Standards Initiative Mass Spectrometry Ontology",
                     version="4.1.0", URI="https://www.ebi.ac.uk/ols/ontologies/ms")
    etree.SubElement(cv_list, "cv", id="UO", fullName="Unit Ontology",
                     version="releases/2020-03-10", URI="http://obo.cvs.sourceforge.net/obo/obo/ontology/phenotype/unit.obo")

    # File description (minimal)
    file_desc = etree.SubElement(root, "fileDescription")
    file_content = etree.SubElement(file_desc, "fileContent")
    etree.SubElement(file_content, "cvParam", cvRef="MS", accession="MS:1000579",
                     name="MS1 spectrum", value="")
    etree.SubElement(file_content, "cvParam", cvRef="MS", accession="MS:1000580",
                     name="MSn spectrum", value="")

    # Software list (minimal)
    soft_list = etree.SubElement(root, "softwareList", count="1")
    software = etree.SubElement(soft_list, "software", id="ms-mint-app", version="1.0")
    etree.SubElement(software, "cvParam", cvRef="MS", accession="MS:1000799",
                     name="custom unreleased software tool", value="ms-mint-app")

    # Instrument configuration (minimal, required by schema)
    inst_list = etree.SubElement(root, "instrumentConfigurationList", count="1")
    inst_config = etree.SubElement(inst_list, "instrumentConfiguration", id="IC1")
    etree.SubElement(inst_config, "cvParam", cvRef="MS", accession="MS:1000031",
                     name="instrument model", value="")

    # Data processing (minimal)
    dp_list = etree.SubElement(root, "dataProcessingList", count="1")
    dp = etree.SubElement(dp_list, "dataProcessing", id="dp1")
    pm = etree.SubElement(dp, "processingMethod", order="1", softwareRef="ms-mint-app")
    etree.SubElement(pm, "cvParam", cvRef="MS", accession="MS:1000544",
                     name="Conversion to mzML", value="")

    # Run element
    run = etree.SubElement(root, "run", id="run1", defaultInstrumentConfigurationRef="IC1")

    # Spectrum list
    spec_list = etree.SubElement(run, "spectrumList", count=str(len(spectra)),
                                  defaultDataProcessingRef="dp1")

    # Polarity CV param
    polarity_cv = CV_POSITIVE_SCAN if polarity.lower().startswith("pos") else CV_NEGATIVE_SCAN
    polarity_name = "positive scan" if polarity.lower().startswith("pos") else "negative scan"

    for idx, spec_data in enumerate(spectra):
        scan_id = spec_data.get("scan_id", idx + 1)
        rt = spec_data.get("rt", 0.0)
        mz_array = spec_data.get("mz_array", np.array([]))
        intensity_array = spec_data.get("intensity_array", np.array([]))

        # Ensure arrays are numpy arrays
        mz_array = np.asarray(mz_array, dtype=np.float64)
        intensity_array = np.asarray(intensity_array, dtype=np.float64)

        # Spectrum element
        spectrum = etree.SubElement(
            spec_list, "spectrum",
            index=str(idx),
            id=f"controllerType=0 controllerNumber=1 scan={scan_id}",
            defaultArrayLength=str(len(mz_array))
        )

        # CV params for spectrum
        etree.SubElement(spectrum, "cvParam", cvRef="MS", accession=CV_MS_LEVEL,
                         name="ms level", value="1")
        etree.SubElement(spectrum, "cvParam", cvRef="MS", accession=polarity_cv,
                         name=polarity_name, value="")
        etree.SubElement(spectrum, "cvParam", cvRef="MS", accession="MS:1000127",
                         name="centroid spectrum", value="")

        # Scan list
        scan_list = etree.SubElement(spectrum, "scanList", count="1")
        etree.SubElement(scan_list, "cvParam", cvRef="MS", accession="MS:1000795",
                         name="no combination", value="")
        scan = etree.SubElement(scan_list, "scan")
        etree.SubElement(scan, "cvParam", cvRef="MS", accession=CV_SCAN_START_TIME,
                         name="scan start time", value=str(rt),
                         unitCvRef="UO", unitAccession="UO:0000010", unitName="second")

        # Binary data array list
        binary_list = etree.SubElement(spectrum, "binaryDataArrayList", count="2")

        # m/z array
        mz_bda = etree.SubElement(binary_list, "binaryDataArray",
                                   encodedLength=str(len(_encode_binary_mzml(mz_array))))
        etree.SubElement(mz_bda, "cvParam", cvRef="MS", accession=CV_64BIT_FLOAT,
                         name="64-bit float", value="")
        etree.SubElement(mz_bda, "cvParam", cvRef="MS", accession=CV_ZLIB_COMPRESSION,
                         name="zlib compression", value="")
        etree.SubElement(mz_bda, "cvParam", cvRef="MS", accession=CV_MZ_ARRAY,
                         name="m/z array", value="", unitCvRef="MS",
                         unitAccession="MS:1000040", unitName="m/z")
        mz_binary = etree.SubElement(mz_bda, "binary")
        mz_binary.text = _encode_binary_mzml(mz_array)

        # Intensity array
        int_bda = etree.SubElement(binary_list, "binaryDataArray",
                                    encodedLength=str(len(_encode_binary_mzml(intensity_array))))
        etree.SubElement(int_bda, "cvParam", cvRef="MS", accession=CV_64BIT_FLOAT,
                         name="64-bit float", value="")
        etree.SubElement(int_bda, "cvParam", cvRef="MS", accession=CV_ZLIB_COMPRESSION,
                         name="zlib compression", value="")
        etree.SubElement(int_bda, "cvParam", cvRef="MS", accession=CV_INTENSITY_ARRAY,
                         name="intensity array", value="", unitCvRef="MS",
                         unitAccession="MS:1000131", unitName="number of detector counts")
        int_binary = etree.SubElement(int_bda, "binary")
        int_binary.text = _encode_binary_mzml(intensity_array)

    # Write to file
    tree = etree.ElementTree(root)
    tree.write(str(output_path), xml_declaration=True, encoding="UTF-8", pretty_print=True)



# def iter_mzml_fast(path: str | Path, *, decode_binary: bool = True) -> Iterator[Dict[str, Any]]:
#     """
#     Legacy lxml-based iterator for mzML files.
#     Kept for reference.
#     """
#     from lxml import etree
#     path = Path(path)
#     NS = "{http://psi.hupo.org/ms/mzml}"
#     context = etree.iterparse(path.as_posix(), events=("start", "end"), remove_comments=True)
#     _, root = next(context)
#
#     current: Dict[str, Any] = {}
#     binary_arrays: List[Dict[str, Any]] = []
#     current_binary: Dict[str, Any] = {}
#
#     for ev, elem in context:
#         tag = elem.tag
#         if tag.startswith(NS):
#             tag = tag[len(NS):]
#
#         if ev == "start" and tag == "spectrum":
#             attribs = elem.attrib
#             index = attribs.get("index", "0")
#             spec_id = attribs.get("id", "")
#             scan_match = re.search(r'scan=(\d+)', spec_id)
#             scan_num = int(scan_match.group(1)) if scan_match else int(index) + 1
#             current = {
#                 "num": scan_num,
#                 "msLevel": 1,
#                 "retentionTime": 0.0,
#                 "polarity": None,
#                 "filterLine": None,
#                 "m/z array": np.array([], dtype=np.float64), 
#                 "intensity array": np.array([], dtype=np.float64)
#             }
#             binary_arrays = []
#
#         elif ev == "end" and tag == "cvParam":
#             accession = elem.get("accession", "")
#             value = elem.get("value", "")
#             if accession == CV_MS_LEVEL:
#                 current["msLevel"] = int(value) if value else 1
#             elif accession == CV_POSITIVE_SCAN:
#                 current["polarity"] = "Positive"
#             elif accession == CV_NEGATIVE_SCAN:
#                 current["polarity"] = "Negative"
#             elif accession == CV_SCAN_START_TIME:
#                 unit_name = elem.get("unitName", "second")
#                 time_val = float(value) if value else 0.0
#                 if unit_name == "minute":
#                     time_val *= 60.0
#                 current["retentionTime"] = time_val
#             elif accession == CV_32BIT_FLOAT:
#                 current_binary["is_64bit"] = False
#             elif accession == CV_64BIT_FLOAT:
#                 current_binary["is_64bit"] = True
#             elif accession == CV_ZLIB_COMPRESSION:
#                 current_binary["is_compressed"] = True
#             elif accession == CV_NO_COMPRESSION:
#                 current_binary["is_compressed"] = False
#             elif accession == CV_MZ_ARRAY:
#                 current_binary["type"] = "mz"
#             elif accession == CV_INTENSITY_ARRAY:
#                 current_binary["type"] = "intensity"
#
#         elif ev == "start" and tag == "binaryDataArray":
#             current_binary = {
#                 "is_64bit": True,
#                 "is_compressed": False,
#                 "type": None,
#                 "data": None,
#             }
#
#         elif ev == "end" and tag == "binary":
#             current_binary["data"] = elem.text
#
#         elif ev == "end" and tag == "binaryDataArray":
#             if decode_binary and current_binary.get("data"):
#                 arr = _decode_binary_mzml(
#                     current_binary["data"],
#                     is_64bit=current_binary.get("is_64bit", True),
#                     is_compressed=current_binary.get("is_compressed", False),
#                 )
#                 current_binary["array"] = arr
#             binary_arrays.append(current_binary)
#             current_binary = {}
#
#         elif ev == "end" and tag == "spectrum":
#             mz_array = np.array([], dtype=np.float64)
#             intensity_array = np.array([], dtype=np.float64)
#             for ba in binary_arrays:
#                 if ba.get("type") == "mz" and "array" in ba:
#                     mz_array = ba["array"].astype(np.float64)
#                 elif ba.get("type") == "intensity" and "array" in ba:
#                     intensity_array = ba["array"].astype(np.float64)
#             current["m/z array"] = mz_array
#             current["intensity array"] = intensity_array
#             yield current
#             elem.clear()
#             root.clear()

def iter_mzml_fast(path: str | Path, *, decode_binary: bool = True) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    # pygixml implementation
    doc = pygixml.parse_file(str(path))
    
    # Select all spectrum elements
    # Using local-name() is safer to avoid namespace issues in XPath
    try:
        spectra = doc.first_child().select_nodes("//*[local-name()='spectrum']")
    except Exception:
        return
    
    for spectrum in spectra:
        spectrum = spectrum.node
        current = {
            "num": 0,
            "msLevel": 1,
            "retentionTime": 0.0,
            "polarity": None,
            "filterLine": None,
            "m/z array": np.array([], dtype=np.float64), 
            "intensity array": np.array([], dtype=np.float64)
        }
        
        # Attributes
        spec_id = spectrum.attribute("id").value or ""
        index = spectrum.attribute("index").value or "0"
        
        # Scan number
        scan_match = re.search(r'scan=(\d+)', spec_id)
        current["num"] = int(scan_match.group(1)) if scan_match else int(index) + 1
        
        # We need to traverse children: cvParam, scanList, binaryDataArrayList
        # pygixml traversal:
        child = spectrum.first_child()
        while child and not child.is_null():
            tag = child.name
            
            if tag.endswith("cvParam"):
                acc = child.attribute("accession").value
                val = child.attribute("value").value
                if acc == CV_MS_LEVEL:
                    current["msLevel"] = int(val) if val else 1
                elif acc == CV_POSITIVE_SCAN:
                    current["polarity"] = "Positive"
                elif acc == CV_NEGATIVE_SCAN:
                    current["polarity"] = "Negative"
            
            elif tag.endswith("scanList"):
                grandchild = child.first_child()
                while grandchild and not grandchild.is_null():
                    if grandchild.name.endswith("scan"):
                        ggchild = grandchild.first_child()
                        while ggchild and not ggchild.is_null():
                            if ggchild.name.endswith("cvParam"):
                                acc = ggchild.attribute("accession").value
                                if acc == CV_SCAN_START_TIME:
                                    unit_name = ggchild.attribute("unitName").value
                                    val = ggchild.attribute("value").value
                                    time_val = float(val) if val else 0.0
                                    if unit_name == "minute":
                                        time_val *= 60.0
                                    current["retentionTime"] = time_val
                            ggchild = ggchild.next_sibling
                    grandchild = grandchild.next_sibling
            
            elif tag.endswith("binaryDataArrayList"):
                bd_child = child.first_child()
                while bd_child and not bd_child.is_null():
                    if bd_child.name.endswith("binaryDataArray"):
                        b_is_64bit = True
                        b_is_compressed = False
                        b_type = None
                        b_data = None
                        
                        b_param = bd_child.first_child()
                        while b_param and not b_param.is_null():
                            b_tag = b_param.name
                            if b_tag.endswith("cvParam"):
                                acc = b_param.attribute("accession").value
                                if acc == CV_32BIT_FLOAT:
                                    b_is_64bit = False
                                elif acc == CV_64BIT_FLOAT:
                                    b_is_64bit = True
                                elif acc == CV_ZLIB_COMPRESSION:
                                    b_is_compressed = True
                                elif acc == CV_NO_COMPRESSION:
                                    b_is_compressed = False
                                elif acc == CV_MZ_ARRAY:
                                    b_type = "mz"
                                elif acc == CV_INTENSITY_ARRAY:
                                    b_type = "intensity"
                            elif b_tag.endswith("binary"):
                                b_data = b_param.text(True, "") 
                            
                            b_param = b_param.next_sibling
                        
                        if decode_binary and b_data:
                            arr = _decode_binary_mzml(
                                b_data,
                                is_64bit=b_is_64bit,
                                is_compressed=b_is_compressed,
                            )
                            if b_type == "mz":
                                current["m/z array"] = arr
                            elif b_type == "intensity":
                                current["intensity array"] = arr
                                
                    bd_child = bd_child.next_sibling

            child = child.next_sibling
            
        yield current

BATCH_SIZE_POINTS = 500_000


def _build_table_from_lists(lists_dict: Dict[str, List], ms_level: int) -> pa.Table:
    """Helper to convert the current lists into a pa.Table."""
    counts = lists_dict.get('counts') or []
    if not counts:
        return pa.Table.from_pydict({})

    counts_arr = np.asarray(counts, dtype=np.int64)
    labels = np.asarray(lists_dict['labels'], dtype=object)
    scan_ids = np.asarray(lists_dict['scan_ids'], dtype=np.int32)
    scan_times = np.asarray(lists_dict['scan_times'], dtype=np.float64)

    labels_repeated = np.repeat(labels, counts_arr)
    scan_ids_repeated = np.repeat(scan_ids, counts_arr)
    scan_times_repeated = np.repeat(scan_times, counts_arr)

    mz_arrays = [np.asarray(arr, dtype=np.float64) for arr in lists_dict['mzs']]
    inten_arrays = [np.asarray(arr, dtype=np.float64) for arr in lists_dict['intensities']]
    mz_concat = np.concatenate(mz_arrays) if mz_arrays else np.array([], dtype=np.float64)
    inten_concat = np.concatenate(inten_arrays) if inten_arrays else np.array([], dtype=np.float64)

    arrays_dict = {
        'ms_file_label': pa.array(labels_repeated, type=pa.string()),
        'scan_id': pa.array(scan_ids_repeated, type=pa.int32()),
        'mz': pa.array(mz_concat, type=pa.float64()),
        'intensity': pa.array(inten_concat, type=pa.float64()),
        'scan_time': pa.array(scan_times_repeated, type=pa.float64()),
    }

    if ms_level == 2:
        mz_precursors = np.asarray(lists_dict['mz_precursors'], dtype=np.float64)
        filter_lines = np.asarray(lists_dict['filterLines'], dtype=object)
        filter_lines_elm = np.asarray(lists_dict['filterLines_ELMAVEN'], dtype=object)

        mz_prec_values = np.repeat(mz_precursors, counts_arr)
        filter_line_values = np.repeat(filter_lines, counts_arr)
        filter_line_elm_values = np.repeat(filter_lines_elm, counts_arr)

        arrays_dict.update({
            'mz_precursor': pa.array(mz_prec_values, type=pa.float64()),
            'filterLine': pa.array(filter_line_values, type=pa.string()),
            'filterLine_ELMAVEN': pa.array(filter_line_elm_values, type=pa.string()),
        })

    return pa.Table.from_pydict(arrays_dict)


def _init_lists() -> Dict[str, List]:
    """Helper to init/reset lists."""
    return {
        'labels': [],
        'scan_ids': [],
        'scan_times': [],
        'counts': [],
        'mzs': [],
        'intensities': [],
        'mz_precursors': [],
        'filterLines': [],
        'filterLines_ELMAVEN': []
    }


def _clear_lists(lists_dict: Dict[str, List]) -> None:
    for values in lists_dict.values():
        values.clear()


def convert_mzxml_to_parquet_fast_batches(
        file_path: str,
        time_unit: str = "s",
        remove_original: bool = False,
        tmp_dir: Optional[str] = None,
        batch_size_points: int = BATCH_SIZE_POINTS,
):
    file_path = Path(file_path)
    time_factor = 60.0 if time_unit == "min" else 1.0
    file_stem = file_path.stem

    current_lists = _init_lists()
    parquet_writer: Optional[pq.ParquetWriter] = None
    wrote_data = False
    if not tmp_dir:
        tmp_dir = tempfile.mkdtemp()
    tmp_fn = Path(tmp_dir, f"{file_stem}.parquet")

    ms_level = None
    polarity = None
    first_scan = True

    total_points = 0  # Counter within current batch

    def flush_batch() -> None:
        nonlocal current_lists, parquet_writer, total_points, wrote_data
        if not current_lists['mzs']:
            return
        table = _build_table_from_lists(current_lists, ms_level or 1)
        # Skip empty tables to avoid creating parquet files with no columns
        if table.num_rows == 0 or table.num_columns == 0:
            _clear_lists(current_lists)
            total_points = 0
            return
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                tmp_fn,
                table.schema,
                compression="snappy",
            )
        parquet_writer.write_table(table)
        wrote_data = True
        _clear_lists(current_lists)
        total_points = 0

    try:
        for data in iter_mzxml_fast(file_path.as_posix(), decode_binary=True):
            mz_arr = data.get("m/z array")
            if mz_arr is None or len(mz_arr) == 0:
                continue

            inten_arr = data.get("intensity array")
            n_points = len(mz_arr)
            total_points += n_points

            if first_scan:
                ms_level = int(data.get("msLevel", 0))
                polarity = "Positive" if data.get("polarity") == "+" else "Negative"
                first_scan = False

            scan_id = int(data.get("num", 0))
            scan_time = float(data.get("retentionTime", 0.0)) * time_factor

            current_lists['labels'].append(file_stem)
            current_lists['scan_ids'].append(scan_id)
            current_lists['scan_times'].append(scan_time)
            current_lists['counts'].append(n_points)
            current_lists['mzs'].append(np.asarray(mz_arr, dtype=np.float64))
            current_lists['intensities'].append(np.asarray(inten_arr, dtype=np.float64))

            if ms_level == 2:
                mz_prec = None
                fline = data.get("filterLine")
                fline_elm = None
                try:
                    mz_prec = float(data["precursorMz"][0]["precursorMz"])
                    if mz_prec is not None and n_points > 0:
                        fline_elm = f"{polarity} {mz_prec:.3f} [{mz_arr[0]:.3f}]"
                except (KeyError, IndexError, TypeError):
                    pass
                current_lists['mz_precursors'].append(mz_prec if mz_prec is not None else np.nan)
                current_lists['filterLines'].append(fline)
                current_lists['filterLines_ELMAVEN'].append(fline_elm)

            # --- START BATCH LOGIC ---
            if total_points >= batch_size_points:
                flush_batch()
            # --- END BATCH LOGIC ---
    except Exception as e:
        raise ValueError(f"Invalid XML in {file_path}: {e}") from e
    else:
        flush_batch()
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if not wrote_data:
        logger.warning(f"No valid data found in {file_path}")
        return 0, file_path, file_stem, 1, "Unknown", None, None

    if ms_level is None:
        ms_level = 1
    if polarity is None:
        polarity = "Unknown"

    if remove_original:
        try:
            os.remove(file_path)
        except OSError:
            pass

    # Extract acquisition datetime from file header
    acq_datetime = get_acquisition_datetime(file_path)

    return file_path, file_stem, ms_level, polarity, tmp_fn.as_posix(), acq_datetime


def _scan_id_from_mzml(spectrum: Dict[str, Any]) -> int:
    scan_id = spectrum.get("id", "")
    if isinstance(scan_id, str) and "scan=" in scan_id:
        try:
            return int(scan_id.split("scan=")[-1])
        except ValueError:
            pass
    return int(spectrum.get("index", 0))


def iter_mzml_pyteomics(path: str | Path) -> Iterator[Dict[str, Any]]:
    from pyteomics import mzml

    with mzml.read(str(path)) as reader:
        for spectrum in reader:
            mz = spectrum.get("m/z array")
            intensity = spectrum.get("intensity array")
            if mz is None or intensity is None or len(mz) == 0:
                continue

            scan = spectrum.get("scanList", {}).get("scan", [{}])[0]
            rt = scan.get("scan start time")
            scan_time = float(rt) if rt is not None else 0.0
            unit = getattr(rt, "unit_info", None)
            if unit == "minute":
                scan_time *= 60.0

            polarity = None
            if "positive scan" in spectrum:
                polarity = "Positive"
            elif "negative scan" in spectrum:
                polarity = "Negative"

            ms_level = spectrum.get("ms level")

            precursor_mz = None
            if ms_level == 2:
                precursors = spectrum.get("precursorList", {}).get("precursor", [])
                if precursors:
                    sel_ions = precursors[0].get("selectedIonList", {}).get("selectedIon", [])
                    if sel_ions:
                        precursor_mz = sel_ions[0].get("selected ion m/z")

            yield {
                "num": _scan_id_from_mzml(spectrum),
                "msLevel": int(ms_level or 0),
                "retentionTime": scan_time,
                "polarity": polarity,
                "filterLine": spectrum.get("filter string"),
                "precursorMz": precursor_mz,
                "m/z array": np.asarray(mz, dtype=np.float64),
                "intensity array": np.asarray(intensity, dtype=np.float64),
            }


def convert_mzml_to_parquet_fast_batches(
        file_path: str,
        time_unit: str = "s",
        remove_original: bool = False,
        tmp_dir: Optional[str] = None,
        batch_size_points: int = BATCH_SIZE_POINTS,
):
    file_path = Path(file_path)
    time_factor = 60.0 if time_unit == "min" else 1.0
    file_stem = file_path.stem

    current_lists = _init_lists()
    parquet_writer: Optional[pq.ParquetWriter] = None
    wrote_data = False
    if not tmp_dir:
        tmp_dir = tempfile.mkdtemp()
    tmp_fn = Path(tmp_dir, f"{file_stem}.parquet")

    ms_level = None
    polarity = None
    first_scan = True

    total_points = 0  # Counter within current batch

    def flush_batch() -> None:
        nonlocal current_lists, parquet_writer, total_points, wrote_data
        if not current_lists['mzs']:
            return
        table = _build_table_from_lists(current_lists, ms_level or 1)
        # Skip empty tables to avoid creating parquet files with no columns
        if table.num_rows == 0 or table.num_columns == 0:
            _clear_lists(current_lists)
            total_points = 0
            return
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                tmp_fn,
                table.schema,
                compression="snappy",
            )
        parquet_writer.write_table(table)
        wrote_data = True
        _clear_lists(current_lists)
        total_points = 0

    try:
        for data in iter_mzml_fast(file_path.as_posix()):
            mz_arr = data.get("m/z array")
            if mz_arr is None or len(mz_arr) == 0:
                continue

            inten_arr = data.get("intensity array")
            n_points = len(mz_arr)
            total_points += n_points

            if first_scan:
                ms_level = int(data.get("msLevel", 0))
                polarity = data.get("polarity")
                first_scan = False

            scan_id = int(data.get("num", 0))
            scan_time = float(data.get("retentionTime", 0.0)) * time_factor

            current_lists['labels'].append(file_stem)
            current_lists['scan_ids'].append(scan_id)
            current_lists['scan_times'].append(scan_time)
            current_lists['counts'].append(n_points)
            current_lists['mzs'].append(np.asarray(mz_arr, dtype=np.float64))
            current_lists['intensities'].append(np.asarray(inten_arr, dtype=np.float64))

            if ms_level == 2:
                mz_prec = data.get("precursorMz")
                fline = data.get("filterLine")
                fline_elm = None
                if mz_prec is not None and n_points > 0:
                    fline_elm = f"{polarity} {float(mz_prec):.3f} [{mz_arr[0]:.3f}]"
                current_lists['mz_precursors'].append(mz_prec if mz_prec is not None else np.nan)
                current_lists['filterLines'].append(fline)
                current_lists['filterLines_ELMAVEN'].append(fline_elm)

            # --- START BATCH LOGIC ---
            if total_points >= batch_size_points:
                flush_batch()
            # --- END BATCH LOGIC ---
    except Exception as e:
        raise ValueError(f"Invalid mzML in {file_path}: {e}") from e
    else:
        flush_batch()
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if not wrote_data:
        logger.warning(f"No valid data found in {file_path}")
        return 0, file_path, file_stem, 1, "Unknown", None, None

    if ms_level is None:
        ms_level = 1
    if polarity is None:
        polarity = "Unknown"

    if remove_original:
        try:
            os.remove(file_path)
        except OSError:
            pass

    # Extract acquisition datetime from file header
    acq_datetime = get_acquisition_datetime(file_path)

    return file_path, file_stem, ms_level, polarity, tmp_fn.as_posix(), acq_datetime



def convert_ms_file_to_parquet_fast_batches(
        file_path: str,
        time_unit: str = "s",
        remove_original: bool = False,
        tmp_dir: Optional[str] = None,
):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".mzxml":
        return convert_mzxml_to_parquet_fast_batches(
            file_path=file_path,
            time_unit=time_unit,
            remove_original=remove_original,
            tmp_dir=tmp_dir,
        )
    if suffix == ".mzml":
        return convert_mzml_to_parquet_fast_batches(
            file_path=file_path,
            time_unit=time_unit,
            remove_original=remove_original,
            tmp_dir=tmp_dir,
        )
    raise ValueError(f"Unsupported MS file type: {suffix}")


def get_metadata(files_path):
    ref_cols = {
        "ms_file_label": 'string',
        "label": 'string',
        "color": 'string',
        "use_for_optimization": 'boolean',
        "use_for_processing": 'boolean',
        "use_for_analysis": 'boolean',
        "sample_type": 'string',
    }
    ref_cols.update({col: 'string' for col in GROUP_COLUMNS})
    required = {"ms_file_label"}

    failed_files = {}
    dfs = []
    for file_path in files_path:
        try:
            preview = read_tabular_file(file_path, nrows=0)
            dtype_map = {col: ref_cols[col] for col in preview.columns if col in ref_cols}
            df = read_tabular_file(file_path, dtype=dtype_map)
            if 'use_for_optimization' in df.columns:
                df['use_for_optimization'] = df['use_for_optimization'].fillna(True)
            if 'use_for_processing' in df.columns:
                df['use_for_processing'] = df['use_for_processing'].fillna(True)

            if missing_required := required - set(df.columns):
                raise ValueError(f"Missing required columns: {missing_required}")
            dfs.append(df)
        except Exception as e:
            failed_files[file_path] = str(e)

    metadata_df = pd.concat(dfs).drop_duplicates(subset="ms_file_label")

    ref_names = list(ref_cols.keys())
    if missing := set(ref_names) - set(metadata_df.columns):
        for col in missing:
            metadata_df[col] = np.nan
    return metadata_df[ref_names], failed_files


def get_targets_v2(files_path):
    ref_cols = {
        "peak_label": 'string',
        "mz_mean": float,
        "mz_width": float,
        "mz": float,
        "rt": float,
        "rt_min": float,
        "rt_max": float,
        "rt_unit": 'string',
        "intensity_threshold": float,
        "polarity": 'string',
        "filterLine": 'string',
        "ms_type": 'string',
        "category": 'string',
        "score": float,
        "peak_selection": 'boolean',
        "bookmark": 'boolean',
        "source": 'string',
        "notes": 'string',
        "rt_auto_adjusted": 'boolean',
    }
    # Only peak_label is strictly required - RT values are smartly derived
    # from various combinations of rt, rt_min, rt_max (at least one required)
    required_cols = {"peak_label"}

    failed_files = {}
    failed_targets = []
    valid_targets = []

    total_files = len(files_path)
    files_processed = 0
    files_failed = 0
    targets_processed = 0
    targets_failed = 0
    rt_adjusted_labels = []  # Track targets with RT outside span that were adjusted

    for file_path in files_path:
        file_name = Path(file_path).name

        try:
            df = read_tabular_file(file_path, dtype=ref_cols)
            
            # Normalize column names from external formats (EL-MAVEN, MZmine, etc.)
            df = normalize_column_names(df)

            if missing_required := required_cols - set(df.columns):
                raise ValueError(f"Missing required columns: {missing_required}")

            for idx, row in df.iterrows():
                targets_processed += 1

                try:
                    target = row.to_dict()
                    target['source'] = file_name

                    if pd.isna(target.get('peak_label')) or target.get('peak_label') == '':
                        raise ValueError("peak_label is empty or null")

                    # ================================================================
                    # SMART RT VALUE DERIVATION
                    # ================================================================
                    # Automatically derive missing RT values based on what's provided
                    # Supports various input formats from different tools
                    
                    has_rt = not pd.isna(target.get('rt'))
                    has_rt_min = not pd.isna(target.get('rt_min'))
                    has_rt_max = not pd.isna(target.get('rt_max'))
                    
                    DEFAULT_RT_WINDOW = 5.0  # seconds
                    
                    # Reject if no RT information provided at all
                    if not any([has_rt, has_rt_min, has_rt_max]):
                        raise ValueError(
                            f"Target '{target['peak_label']}' must have at least one of: rt, rt_min, or rt_max"
                        )
                    
                    # Scenario 0: All three values provided - just validate, no derivation needed
                    if has_rt and has_rt_min and has_rt_max:
                        logging.debug(
                            f"Target '{target['peak_label']}': All RT values provided "
                            f"(rt={target['rt']:.2f}, rt_min={target['rt_min']:.2f}, rt_max={target['rt_max']:.2f})"
                        )
                        # Continue to validation below
                    
                    # Scenario 1: Have RT bounds, derive center
                    elif has_rt_min and has_rt_max and not has_rt:
                        target['rt'] = (target['rt_min'] + target['rt_max']) / 2
                        logging.debug(
                            f"Target '{target['peak_label']}': Derived rt={target['rt']:.2f} "
                            f"from rt_min={target['rt_min']:.2f} and rt_max={target['rt_max']:.2f}"
                        )
                    
                    # Scenario 2: Have RT only, derive bounds
                    elif has_rt and not has_rt_min and not has_rt_max:
                        target['rt_min'] = target['rt'] - DEFAULT_RT_WINDOW
                        target['rt_max'] = target['rt'] + DEFAULT_RT_WINDOW
                        target['rt_auto_adjusted'] = True  # Mark for data-driven optimization
                        logging.debug(
                            f"Target '{target['peak_label']}': Derived rt_min={target['rt_min']:.2f} "
                            f"and rt_max={target['rt_max']:.2f} from rt={target['rt']:.2f} (±{DEFAULT_RT_WINDOW}s)"
                        )
                    
                    # Scenario 3a: Have RT and rt_min, derive rt_max (symmetric window)
                    elif has_rt and has_rt_min and not has_rt_max:
                        window = abs(target['rt'] - target['rt_min'])
                        target['rt_max'] = target['rt'] + window
                        target['rt_auto_adjusted'] = True  # Mark for data-driven optimization of rt_max
                        logging.debug(
                            f"Target '{target['peak_label']}': Derived rt_max={target['rt_max']:.2f} "
                            f"from rt={target['rt']:.2f} and rt_min={target['rt_min']:.2f} (window={window:.2f}s)"
                        )
                    
                    # Scenario 3b: Have RT and rt_max, derive rt_min (symmetric window)
                    elif has_rt and has_rt_max and not has_rt_min:
                        window = abs(target['rt_max'] - target['rt'])
                        target['rt_min'] = target['rt'] - window
                        target['rt_auto_adjusted'] = True  # Mark for data-driven optimization of rt_min
                        logging.debug(
                            f"Target '{target['peak_label']}': Derived rt_min={target['rt_min']:.2f} "
                            f"from rt={target['rt']:.2f} and rt_max={target['rt_max']:.2f} (window={window:.2f}s)"
                        )
                    
                    # Unsupported combination (e.g., only rt_min or only rt_max)
                    else:
                        # Build helpful error message showing what was provided
                        provided = []
                        if has_rt:
                            provided.append(f"rt={target.get('rt'):.2f}")
                        if has_rt_min:
                            provided.append(f"rt_min={target.get('rt_min'):.2f}")
                        if has_rt_max:
                            provided.append(f"rt_max={target.get('rt_max'):.2f}")
                        
                        raise ValueError(
                            f"Target '{target['peak_label']}': Cannot derive RT values from the provided data. "
                            f"You provided: {', '.join(provided)}. "
                            f"Valid combinations: (1) 'rt' only, (2) 'rt_min' AND 'rt_max', "
                            f"(3) 'rt' + 'rt_min', or (4) 'rt' + 'rt_max'. "
                            f"Providing only 'rt_min' or only 'rt_max' is not supported."
                        )
                    
                    # At this point all RT values should be populated
                    # Now validate and adjust if needed
                    
                    # Validate rt_min < rt_max
                    if target['rt_min'] >= target['rt_max']:
                        raise ValueError(
                            f"Target '{target['peak_label']}': rt_min ({target['rt_min']:.2f}) "
                            f"must be less than rt_max ({target['rt_max']:.2f})"
                        )
                    
                    # Validate non-negative values
                    if target['rt_min'] < 0:
                        raise ValueError(
                            f"Target '{target['peak_label']}': rt_min cannot be negative ({target['rt_min']:.2f})"
                        )
                    
                    # Warn if window is suspiciously large
                    window_size = target['rt_max'] - target['rt_min']
                    if window_size > 60.0:
                        logging.warning(
                            f"Target '{target['peak_label']}': Large RT window ({window_size:.1f}s). "
                            f"Please verify rt_min and rt_max values."
                        )

                    # check if RT-unit is seconds or minutes
                    if 'rt_unit' in target:
                        if target['rt_unit'] in ['minutes', 'min', 'm']:
                            target['rt'] = target['rt'] * 60
                            target['rt_max'] = target['rt_max'] * 60
                            target['rt_min'] = target['rt_min'] * 60
                        elif target['rt_unit'] in ['seconds', 'sec', 's']:
                            target['rt'] = target['rt']
                            target['rt_max'] = target['rt_max']
                            target['rt_min'] = target['rt_min']
                        else:
                            raise ValueError(f"Invalid RT-unit: {target['rt_unit']}")

                    target['rt_unit'] = 's'
                    
                    # Validate RT is within [rt_min, rt_max] span, otherwise set to midpoint
                    rt_val = target.get('rt')
                    rt_min_val = target.get('rt_min')
                    rt_max_val = target.get('rt_max')
                    if rt_val is not None and rt_min_val is not None and rt_max_val is not None:
                        if rt_val < rt_min_val or rt_val > rt_max_val:
                            old_rt = rt_val
                            target['rt'] = (rt_min_val + rt_max_val) / 2
                            target['rt_auto_adjusted'] = True  # Mark for later update to max intensity
                            rt_adjusted_labels.append(target['peak_label'])
                            logging.warning(
                                f"Target '{target['peak_label']}': RT {old_rt:.1f}s was outside span "
                                f"[{rt_min_val:.1f}, {rt_max_val:.1f}], adjusted to midpoint {target['rt']:.1f}s"
                            )


                    # ================================================================
                    # DEFAULT VALUES
                    # ================================================================
                    
                    # MZ Width: Default to 10 ppm if not provided
                    if pd.isna(target.get('mz_width')):
                        target['mz_width'] = 10.0  # ppm
                        logging.debug(f"Target '{target['peak_label']}': Using default mz_width=10.0 ppm")

                    # Polarity: Optional field (not used in processing, just metadata)
                    # Normalize if provided, otherwise leave as NULL
                    pol = target.get('polarity')
                    if not pd.isna(pol):
                        pol_str = str(pol)
                        target['polarity'] = (
                            pol_str
                            .replace('+', 'Positive')
                            .replace('positive', 'Positive')
                            .replace('-', 'Negative')
                            .replace('negative', 'Negative')
                        )
                    # If polarity is NaN, it stays NULL - that's fine
                    
                    # MS Type: Derive from filterLine (source of truth)
                    # MS2 targets MUST have filterLine, MS1 targets don't
                    has_filter = 'filterLine' in target and not pd.isna(target.get('filterLine')) and target['filterLine']
                    derived_ms_type = 'ms2' if has_filter else 'ms1'
                    
                    # Validate if user provided ms_type
                    user_ms_type = target.get('ms_type')
                    if not pd.isna(user_ms_type) and str(user_ms_type).lower() != derived_ms_type:
                        logging.warning(
                            f"Target '{target['peak_label']}': User-provided ms_type='{user_ms_type}' "
                            f"contradicts filterLine presence. Using '{derived_ms_type}' "
                            f"(filterLine {'present' if has_filter else 'absent'})."
                        )
                    
                    target['ms_type'] = derived_ms_type

                    valid_targets.append(target)

                except Exception as e:
                    targets_failed += 1
                    failed_targets.append({
                        'file': file_name,
                        'row': idx,
                        'peak_label': row.get('peak_label', 'UNKNOWN'),
                        'error': str(e)
                    })
                    logging.warning(f"Failed to process target at row {idx} in {file_name}: {str(e)}")

            files_processed += 1

        except Exception as e:
            files_failed += 1
            failed_files[file_path] = str(e)
            logging.error(f"Failed to process file {file_path}: {str(e)}")

            try:
                df_count = read_tabular_file(file_path)  # Read file to count rows
                file_target_count = len(df_count)
                targets_failed += file_target_count
                targets_processed += file_target_count
            except:
                logging.warning(f"Could not count targets in failed file {file_path}")

    # Check if there are valid targets
    if not valid_targets:
        error_msg = "No valid targets found in any file."
        error_msg += f"\n  Total files: {total_files}"
        error_msg += f"\n  Files processed: {files_processed}"
        error_msg += f"\n  Files failed: {files_failed}"
        error_msg += f"\n  Targets processed: {targets_processed}"
        error_msg += f"\n  Targets failed: {targets_failed}"
        raise ValueError(error_msg)

    # Build DataFrame with valid targets
    targets_df = pd.DataFrame(valid_targets)

    # Eliminar duplicados basados en peak_label, pero registrar los duplicados encontrados
    duplicate_labels = (
        targets_df[targets_df.duplicated(subset="peak_label", keep=False)]
        .get("peak_label", pd.Series([], dtype="string"))
        .dropna()
        .unique()
        .tolist()
    )
    if duplicate_labels:
        logging.warning(
            "Found duplicate target labels; keeping first occurrence: %s", duplicate_labels
        )
    targets_df = targets_df.drop_duplicates(subset="peak_label")

    # Ensure all reference columns exist
    ref_names = list(ref_cols.keys())
    for col in ref_names:
        if col not in targets_df.columns:
            targets_df[col] = np.nan

    # Apply the correct data types
    for col, dtype in ref_cols.items():
        if col in targets_df.columns:
            if dtype == 'string':
                targets_df[col] = targets_df[col].astype('string')
            elif dtype == 'boolean':
                if col != 'peak_selection':  # handle peak_selection defaults separately below
                    targets_df[col] = targets_df[col].astype('boolean').fillna(False)
            elif dtype == float:
                targets_df[col] = pd.to_numeric(targets_df[col], errors='coerce')

    # Fill default values
    # Default to selected when the column is missing/blank so uploaded targets are included by default.
    targets_df['peak_selection'] = targets_df['peak_selection'].fillna(True).astype(bool)
    targets_df['bookmark'] = targets_df['bookmark'].fillna(False).astype(bool)

    logging.info("Processing summary:")
    logging.info(f"  Total files: {total_files}")
    logging.info(f"  Files processed: {files_processed}")
    logging.info(f"  Files failed: {files_failed}")
    logging.info(f"  Targets processed: {targets_processed}")
    logging.info(f"  Targets failed: {targets_failed}")
    logging.info(f"  Unique valid targets: {len(targets_df)}")
    if rt_adjusted_labels:
        logging.info(f"  RT values adjusted (outside span): {len(rt_adjusted_labels)}")

    # Prepare stats dictionary
    stats = {
        'total_files': total_files,
        'files_processed': files_processed,
        'files_failed': files_failed,
        'targets_processed': targets_processed,
        'targets_failed': targets_failed,
        'unique_valid_targets': len(targets_df),
        'duplicate_peak_labels': len(duplicate_labels),
        'duplicate_peak_labels_list': duplicate_labels,
        'rt_adjusted_count': len(rt_adjusted_labels),
        'rt_adjusted_labels': rt_adjusted_labels,
    }

    return targets_df[ref_names], failed_files, failed_targets, stats


def _insert_ms_data(wdir, ms_type, batch_ms, batch_ms_data, n_cpus=None):
    failed_files = []

    if not batch_ms[ms_type]:
        return 0, failed_files

    pldf = pd.DataFrame(
        batch_ms[ms_type],
        columns=['ms_file_label', 'label', 'ms_type', 'polarity', 'file_type', 'acquisition_datetime'],
    )

    # Auto-detect sample type and color in bulk before insertion
    from .plugins.ms_files import detect_sample_color
    
    # helper wrapper to unpack tuple
    def _get_meta(label):
        c, t = detect_sample_color(label)
        return (c or '#BBBBBB', t or 'Sample')
        
    # Apply to DataFrame
    meta = pldf['ms_file_label'].apply(_get_meta)
    pldf['color'] = [m[0] for m in meta]
    pldf['sample_type'] = [m[1] for m in meta]
    
    parquet_files_to_delete = batch_ms_data[ms_type].copy()
    insert_success = False
    
    with duckdb_connection(wdir, n_cpus=n_cpus) as conn:
        if conn is None:
            raise PreventUpdate
        try:
            # Performance tuning:
            # 1. Increase WAL limit to reduce disk flushes
            conn.execute("SET wal_autocheckpoint='1GB'")
            
            # Insert with all metadata columns populated
            conn.execute(
                "INSERT INTO samples(ms_file_label, label, ms_type, polarity, file_type, color, sample_type, acquisition_datetime) "
                "SELECT ms_file_label, label, ms_type, polarity, file_type, color, sample_type, acquisition_datetime FROM pldf"
            )
            
            ms_data_table = f'{ms_type}_data'
            extra_columns = ['mz_precursor', 'filterLine', 'filterLine_ELMAVEN']
            select_columns = ['ms_file_label', 'scan_id', 'mz', 'intensity', 'scan_time']
            if ms_type == 'ms2':
                select_columns.extend(extra_columns)
            conn.execute(f"""
                         INSERT INTO {ms_data_table} ({', '.join(select_columns)})
                         SELECT {', '.join(select_columns)}
                         FROM read_parquet(?)
                         """,
                         [batch_ms_data[ms_type]])
            
            # CHECKPOINT after each batch to flush WAL and keep insert times consistent
            conn.execute("CHECKPOINT")
            insert_success = True
                    
        except Exception as e:
            logging.error(f"DB error: {e}")
            failed_files.extend([{file_path: str(e)} for file_path in batch_ms_data[ms_type]])
    
    # Delete parquet files AFTER connection closes (files are released)
    if insert_success:
        for parquet_file in parquet_files_to_delete:
            try:
                os.remove(parquet_file)
            except OSError:
                pass  # File may already be deleted or locked
                
    return len(batch_ms_data[ms_type]), failed_files


# IMPORTANT: We've defined these functions here temporarily, but it should be moved to the backend.
def process_ms_files(wdir, set_progress, selected_files, n_cpus):
    def _send_progress(percent, stage="Processing", detail=""):
        if not set_progress:
            return
        try:
            set_progress(percent, stage, detail)
        except TypeError:
            try:
                set_progress(percent, detail)
            except TypeError:
                try:
                    set_progress(percent)
                except Exception:
                    pass
            except Exception:
                pass

    file_list = sorted([file for folder in selected_files.values() for file in folder])
    n_total = len(file_list)
    failed_files = []
    duplicates_count = 0
    total_processed = 0 # Files committed to DB
    n_converted = 0     # Files processed by CPU (for smooth progress bar)
    last_batch_time = None
    import concurrent.futures
    import time

    start_time = time.perf_counter()
    logger.info(f"Starting MS file processing for {n_total} file(s).")

    # set progress to 1 to the user feedback
    _send_progress(1)

    # get the ms_file_label data as df to avoid multiple queries
    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        data = conn.execute("SELECT ms_file_label FROM samples").df()

    files_name = {Path(file_path).stem: Path(file_path) for file_path in file_list}

    mask = data["ms_file_label"].isin(files_name)  # dict as set of keys
    duplicates = data.loc[mask, "ms_file_label"]  # Series with only the duplicate labels
    if not duplicates.empty:
        duplicates_count = int(duplicates.shape[0])
        _send_progress(round(duplicates.shape[0] / n_total * 100, 1))
        logging.info("Found %d duplicates: %s", duplicates.shape[0], duplicates.tolist())
        logger.info(f"Skipped {duplicates.shape[0]} duplicate file(s): {', '.join(duplicates.tolist())}")

    if len(files_name) - len(duplicates) > 0:
        total_to_process = len(files_name) - len(duplicates)
        # Use workspace's data/temp folder for intermediate parquet files
        workspace_temp = Path(wdir) / "data" / "temp"
        workspace_temp.mkdir(parents=True, exist_ok=True)
        
        # Filter out duplicates and create list of files to process
        files_to_process = [
            file_path for file_name, file_path in files_name.items()
            if file_name not in duplicates.tolist()
        ]
        
        # Chunked processing: submit files in chunks to limit temp folder size
        chunk_size = 64  # Max files in-flight at once
        batch_ms = {'ms1': [], 'ms2': []}
        batch_ms_data = {'ms1': [], 'ms2': []}
        # Decouple DB batch size from CPU count for fewer transactions
        batch_size = 100
        total_batches = (total_to_process + batch_size - 1) // batch_size
        batch_num = 0
        batch_start = time.time()

        # Prepare ThreadPoolExecutor for DB writes (max_workers=1 to ensure serial writes)
        # pipeline depth of 1: We convert batch N while writing batch N-1.
        with tempfile.TemporaryDirectory(dir=workspace_temp) as tmpdir, \
             concurrent.futures.ThreadPoolExecutor(max_workers=1) as db_executor, \
             concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor: # Reuse process pool!

            db_future = None  # Holds the future of the *previous* batch insertion

            # Helper to handle DB result processing
            def process_db_result(future):
                nonlocal total_processed, failed_files
                try:
                    b_processed, b_failed = future.result()
                    total_processed += b_processed
                    failed_files.extend(b_failed)
                    return b_processed
                except Exception as e:
                    logger.error(f"Async DB Insert Failed: {e}")
                    return 0

            # Process files in chunks
            for chunk_start in range(0, len(files_to_process), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(files_to_process))
                chunk_files = files_to_process[chunk_start:chunk_end]
                
                futures_name = {
                    executor.submit(
                        convert_ms_file_to_parquet_fast_batches, file_path, tmp_dir=tmpdir
                    ): file_path
                    for file_path in chunk_files
                }
                
                for future in concurrent.futures.as_completed(futures_name.keys()):
                        try:
                            result = future.result()
                        except Exception as e:
                            failed_files.append({futures_name[future]: str(e)})
                            n_converted += 1
                            done_count = n_converted + duplicates_count
                            _send_progress(round(done_count / n_total * 100, 1), stage="Processing", detail="Processing files...")
                            logger.error(f"Failed: {Path(futures_name[future]).name} ({e})")
                            continue

                        _file_path, _ms_file_label, _ms_level, _polarity, _parquet_df, _acq_datetime = result
                        
                        if _parquet_df is None:
                            failed_files.append({_file_path: "No valid data found or conversion failed"})
                            n_converted += 1
                            done_count = n_converted + duplicates_count
                            _send_progress(round(done_count / n_total * 100, 1), stage="Processing", detail="Processing files...")
                            logger.warning(f"Failed (no data): {Path(_file_path).name}")
                            continue

                        # Success handling
                        n_converted += 1
                        
                        suffix = Path(_file_path).suffix.lower()
                        if suffix == ".mzxml":
                            file_type = "mzXML"
                        elif suffix == ".mzml":
                            file_type = "mzML"
                        else:
                            file_type = suffix.lstrip(".")
                        batch_ms[f'ms{_ms_level}'].append(
                            (_ms_file_label, _ms_file_label, f'ms{_ms_level}', _polarity, file_type, _acq_datetime)
                        )

                        batch_ms_data[f'ms{_ms_level}'].append(_parquet_df)
                        
                        # Update progress bar smoothly (file by file)
                        done_count = n_converted + duplicates_count
                        pct = round(done_count / n_total * 100, 1)
                        # We can show detailed text if we want, or keep it generic until batch finishes
                        # Showing batch info here might be wrong since batch_num updates later?
                        # Let's show generic "Processing file X/Y" or similar
                        detail_text = (
                            f"Batch {batch_num+1}/{total_batches} | "
                            f"Progress {done_count}/{n_total}"
                        )
                        if last_batch_time is not None:
                             detail_text += f" | Time/batch {last_batch_time:.2f}s"

                        _send_progress(pct, stage="Processing", detail=detail_text)

                        if len(batch_ms['ms1']) == batch_size:
                            # 1. Wait for previous DB write to finish (pipelining)
                            if db_future:
                                process_db_result(db_future)

                            # 2. Submit new batch (COPY data so main thread can reuse lists)
                            # Deep copy not needed for list of strings/tuples, shallow copy of list is fine
                            # But we need new lists for next batch in main thread
                            batch_ms_copy = {'ms1': batch_ms['ms1'], 'ms2': []} # MS2 is separate triggering
                            batch_data_copy = {'ms1': batch_ms_data['ms1'], 'ms2': []}

                            t_insert_start = time.time() # This timestamp is less meaningful now, denotes submit time
                            db_future = db_executor.submit(_insert_ms_data, wdir, 'ms1', batch_ms_copy, batch_data_copy, n_cpus=4)

                            batch_elapsed = time.time() - batch_start
                            last_batch_time = batch_elapsed # Update for next batch display

                            batch_num += 1
                            detail = (
                                f"Batch {batch_num}/{total_batches} | "
                                f"Progress {done_count}/{n_total} | "
                                f"Time/batch {batch_elapsed:.2f}s"
                            )
                            # Ensure we don't send > 100% or anything weird, though n_converted handles it
                            _send_progress(pct, stage="Processing", detail=detail)
                            logger.info(detail)

                            batch_ms['ms1'] = []
                            batch_ms_data['ms1'] = []
                            batch_start = time.time()

                            # Periodic CHECKPOINT every 20 batches (Sync check)
                            if batch_num % 20 == 0:
                                # Must wait for pending DB operations before checkpointing
                                if db_future:
                                    process_db_result(db_future)
                                    db_future = None
                                with duckdb_connection(wdir) as conn:
                                    if conn is not None:
                                        conn.execute("CHECKPOINT")
                                        logger.info(f"Checkpoint after batch {batch_num}")

                        elif len(batch_ms['ms2']) == batch_size:
                             # 1. Wait for previous DB write
                            if db_future:
                                process_db_result(db_future)

                            batch_ms_copy = {'ms1': [], 'ms2': batch_ms['ms2']}
                            batch_data_copy = {'ms1': [], 'ms2': batch_ms_data['ms2']}

                            db_future = db_executor.submit(_insert_ms_data, wdir, 'ms2', batch_ms_copy, batch_data_copy, n_cpus=4)

                            batch_elapsed = time.time() - batch_start

                            batch_num += 1
                            detail = (
                                f"Batch {batch_num}/{total_batches} | "
                                f"Conv Time: {batch_elapsed:0.2f}s | "
                                f"DB: Async"
                            )
                            done_count = total_processed + len(failed_files) + duplicates_count
                            _send_progress(round(done_count / n_total * 100, 1), detail)
                            logger.info(detail)

                            batch_ms['ms2'] = []
                            batch_ms_data['ms2'] = []
                            batch_start = time.time()

                            # Periodic CHECKPOINT every 20 batches
                            if batch_num % 20 == 0:
                                if db_future:
                                    process_db_result(db_future)
                                    db_future = None
                                with duckdb_connection(wdir) as conn:
                                    if conn is not None:
                                        conn.execute("CHECKPOINT")
                                        logger.info(f"Checkpoint after batch {batch_num}")

            # Flush remaining batches
            # 1. Wait for any pending async write
            _send_progress(99.0, stage="Finalizing", detail="Finishing background tasks...")
            t_wait_start = time.time()
            if db_future:
                process_db_result(db_future)
                db_future = None
            wait_time = time.time() - t_wait_start

            # 2. Process remaining items synchronously (or async and wait)
            if len(batch_ms['ms1']):
                _send_progress(99.5, stage="Finalizing", detail="Saving last batch (MS1)...")
                t_insert_start = time.time()
                b_processed, b_failed = _insert_ms_data(wdir, 'ms1', batch_ms, batch_ms_data, n_cpus=4)
                t_insert_end = time.time()
                
                batch_elapsed = time.time() - batch_start
                insert_time = t_insert_end - t_insert_start
                conv_time = batch_elapsed - insert_time - wait_time
                
                total_processed += b_processed
                failed_files.extend(b_failed)
                
                batch_num += 1
                detail = (
                    f"Batch {batch_num}/{total_batches} | "
                    f"Files {done_count}/{n_total}"
                )
                done_count = n_converted + duplicates_count
                _send_progress(round(done_count / n_total * 100, 1), stage="Processing", detail=detail)
                
                log_detail = (
                    f"Batch {batch_num}/{total_batches} | "
                    f"Conv: {conv_time:0.2f}s | "
                    f"Wait: {wait_time:0.2f}s | "
                    f"DB: {insert_time:0.2f}s (Sync Flush)"
                )
                logger.info(log_detail)

                batch_ms['ms1'] = []
                batch_ms_data['ms1'] = []
                
            if len(batch_ms['ms2']):
                _send_progress(99.5, stage="Finalizing", detail="Saving last batch (MS2)...")
                t_insert_start = time.time()
                b_processed, b_failed = _insert_ms_data(wdir, 'ms2', batch_ms, batch_ms_data, n_cpus=4)
                t_insert_end = time.time()
                
                batch_elapsed = time.time() - batch_start
                insert_time = t_insert_end - t_insert_start
                conv_time = batch_elapsed - insert_time - wait_time

                total_processed += b_processed
                failed_files.extend(b_failed)
                
                batch_num += 1
                detail = (
                    f"Batch {batch_num}/{total_batches} | "
                    f"Files {done_count}/{n_total}"
                )
                done_count = n_converted + duplicates_count
                _send_progress(round(done_count / n_total * 100, 1), stage="Processing", detail=detail)
                
                log_detail = (
                    f"Batch {batch_num}/{total_batches} | "
                    f"Conv: {conv_time:0.2f}s | "
                    f"Wait: {wait_time:0.2f}s | "
                    f"DB: {insert_time:0.2f}s (Sync Flush)"
                )
                logger.info(log_detail)
                
                batch_ms['ms2'] = []
                batch_ms_data['ms2'] = []

    total_elapsed = time.perf_counter() - start_time
    logger.info(f"Completed MS file processing. Success: {total_processed}, Failed: {len(failed_files)}. Total time: {total_elapsed:0.2f}s.")
    _send_progress(100, stage="Processing", detail=f"Done. Processed {total_processed} files.")
    
    # Clean up the workspace's temp folder
    workspace_temp = Path(wdir) / "data" / "temp"
    if workspace_temp.exists():
        try:
            import shutil
            shutil.rmtree(workspace_temp, ignore_errors=True)
            logger.info(f"Cleaned up temp folder: {workspace_temp}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp folder: {e}")
    
    return total_processed, failed_files, duplicates_count


def process_metadata(wdir, set_progress, selected_files):
    file_list = [file for folder in selected_files.values() for file in folder]
    n_total = len(file_list)
    set_progress(10)
    metadata_df, failed_files = get_metadata(file_list)

    if metadata_df.empty:
        set_progress(100)
        return 0, failed_files

    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        
        # Explicitly register the dataframe to ensure DuckDB can see it
        conn.register('metadata_df_source', metadata_df)
        
        columns_to_update = {
            "use_for_optimization": "BOOLEAN",
            "use_for_processing": "BOOLEAN",
            "use_for_analysis": "BOOLEAN",
            "color": "VARCHAR",
            "label": "VARCHAR",
            "sample_type": "VARCHAR",
            **{col: "VARCHAR" for col in GROUP_COLUMNS},
        }
        set_clauses = []
        for col, cast in columns_to_update.items():
            # Only update if the column exists in the dataframe and is not all-null?
            # Actually, using the dataframe's own columns is safer if we want to support partial updates better.
            # But get_metadata fills missing cols with NaN, so we stick to the CASE WHEN logic.
            # We assume NaN means "do not update/keep existing".
            set_clauses.append(f"""
                            {col} = CASE 
                                WHEN metadata_df_source.{col} IS NOT NULL 
                                THEN CAST(metadata_df_source.{col} AS {cast}) 
                                ELSE samples.{col} 
                            END
                        """)
        set_clause = ", ".join(set_clauses)
        
        # Use the registered view name
        stmt = f"""
            UPDATE samples 
            SET {set_clause}
            FROM metadata_df_source 
            WHERE samples.ms_file_label = metadata_df_source.ms_file_label
        """
        conn.execute(stmt)
        
        # Verify if any rows were updated? (DuckDB doesn't easily return affected rows here without another query, 
        # but the operation should have succeeded)
        
        # Import lazily to avoid circular dependency at module import time.
        from .plugins.ms_files import generate_colors
        generate_colors(wdir)

    set_progress(100)
    return len(metadata_df), failed_files


def process_targets(wdir, set_progress, selected_files):
    file_list = [file for folder in selected_files.values() for file in folder]
    n_total = len(file_list)
    set_progress(10)
    try:
        targets_df, failed_files, failed_targets, stats = get_targets_v2(file_list)
    except Exception as e:
        logging.error(f"Error processing targets: {e}")
        failed_files = {file: str(e) for file in file_list}
        return 0, failed_files, [], {}

    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        conn.execute(
            "INSERT OR REPLACE INTO targets(peak_label, mz_mean, mz_width, mz, rt, rt_min, rt_max, rt_unit, "
            "intensity_threshold, polarity, filterLine, ms_type, category, score, peak_selection, bookmark, source, notes, rt_auto_adjusted) "
            "SELECT peak_label, mz_mean, mz_width, mz, rt, rt_min, rt_max, rt_unit, intensity_threshold, polarity, "
            "filterLine, ms_type, category, score, peak_selection, bookmark, source, notes, rt_auto_adjusted "
            "FROM targets_df ORDER BY mz_mean, peak_label"
        )
        
        # Create initial backup CSV for recovery
        try:
            from pathlib import Path
            data_dir = Path(wdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            backup_path = data_dir / "targets_backup.csv"
            conn.execute(
                "COPY (SELECT * FROM targets) TO ? (HEADER, DELIMITER ',')",
                (str(backup_path),)
            )
            logging.info(f"Created initial targets backup: {backup_path}")
        except Exception as e:
            logging.warning(f"Failed to create initial targets backup: {e}")

    set_progress(100)
    return len(targets_df), failed_files, failed_targets, stats


# TODO: check if we need to use the intensity_threshold as baseline
def sparsify_chrom(
    scan,
    intensity,
    w=1,
    baseline=1.0,
    eps=0.0,
    min_peak_width=3,
    ):
    """
    Keep all points above the baseline (or baseline+eps)
    plus ±w neighboring points.
    """
    scan = np.asarray(scan)
    intensity = np.asarray(intensity)

    signal = intensity > (baseline + eps)
    if not np.any(signal):
        return scan[:0], intensity[:0]

    # Remove small islands (shorter than min_peak_width)
    structure = np.ones(min_peak_width, dtype=bool)
    cleaned_signal = binary_opening(signal, structure=structure)
    if not np.any(cleaned_signal):
        return scan[:0], intensity[:0]

    kernel = np.ones(2 * w + 1, dtype=np.int8)
    keep = np.convolve(cleaned_signal.astype(np.int8), kernel, mode="same") > 0

    return scan[keep], intensity[keep]

def proportional_min1_selection(df, group_col, list_col, total_select, seed=None):
    """
    Rule:
      - If total elements < total_select -> include all.
      - If not -> quota per group = max(1, floor(total_select * p_group)).
    Returns: (quotas, flat_selected_list)
    """
    rng = random.Random(seed) if seed is not None else random

    # Normalize list column into plain lists
    groups = []
    for lst in df[list_col]:
        if isinstance(lst, list):
            groups.append(lst)
        elif lst is None:
            groups.append([])
        else:
            try:
                groups.append(list(lst))
            except TypeError:
                groups.append([lst])

    # Total count
    sizes = [len(lst) for lst in groups]
    total = sum(sizes)

    # CASE 1: total less than total_select -> include ALL
    if total <= total_select:
        # Simply concatenate all elements
        full_list = []
        for lst in groups:
            full_list.extend(lst)
        return {g: s for g, s in zip(df[group_col], sizes)}, full_list

    # CASE 2: apply proportional quota
    quotas = {}
    for g, n in zip(df[group_col], sizes):
        p = n / total
        q = max(1, math.floor(total_select * p))
        quotas[g] = q

    # Sampling by group
    selected = []
    for (g, lst), q in zip(zip(df[group_col], groups), quotas.values()):
        seq = list(lst)
        k = min(q, len(seq))
        selected.extend(rng.sample(seq, k))

    return quotas, selected


def fig_to_src(fig, dpi=100):
    out_img = io.BytesIO()
    fig.savefig(out_img, format="jpeg", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


def fix_first_emtpy_line_after_upload_workaround(file_path):
    logging.warning(f'Check if first line is empty in {file_path}.')

    with open(file_path, 'r') as file:
        lines = file.readlines()

    if not lines:
        return

    # Check if the first line is an empty line (contains only newline character)
    if lines[0] == "\n":
        logging.warning(f'Empty first line detected in {file_path}. Removing it.')
        lines.pop(0)

        with open(file_path, 'w') as file:
            file.writelines(lines)
