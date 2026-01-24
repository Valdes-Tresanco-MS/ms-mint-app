
from ms_mint_app.tools import sparsify_chrom
import itertools
import logging
import math
import numpy as np

try:
    from scipy.signal import savgol_filter as _savgol_filter
except Exception:
    _savgol_filter = None

try:
    import lttbc as _lttbc
except Exception:
    _lttbc = None

logger = logging.getLogger(__name__)
LOG_TRACE_POINT_LIMIT = 50
_LTTBC_MISSING_WARNED = False


def _savgol_coeffs_numpy(window_length, polyorder, deriv=0, delta=1.0):
    half_window = window_length // 2
    pos = np.arange(-half_window, half_window + 1, dtype=float)
    a = np.vander(pos, polyorder + 1, increasing=True)
    b = np.zeros(polyorder + 1, dtype=float)
    b[deriv] = math.factorial(deriv) / (delta ** deriv)
    return np.linalg.lstsq(a.T, b, rcond=None)[0]


def _savgol_filter_numpy(intensity, window_length, polyorder):
    intensity = np.asarray(intensity, dtype=float)
    n_points = intensity.size
    if n_points == 0:
        return intensity

    half_window = window_length // 2
    if n_points < window_length:
        return intensity

    coeffs = _savgol_coeffs_numpy(window_length, polyorder)
    filtered = np.empty_like(intensity, dtype=float)

    # Interior points via convolution
    interior = np.convolve(intensity, coeffs[::-1], mode='valid')
    filtered[half_window:-half_window] = interior

    # Edge handling via polynomial interpolation (match SciPy's interp mode)
    x = np.arange(window_length, dtype=float)
    left_poly = np.polyfit(x, intensity[:window_length], polyorder)
    filtered[:half_window] = np.polyval(left_poly, x[:half_window])
    right_poly = np.polyfit(x, intensity[-window_length:], polyorder)
    filtered[-half_window:] = np.polyval(right_poly, x[-half_window:])

    return filtered


def apply_savgol_smoothing(intensity, window_length=7, polyorder=2):
    if window_length is None:
        window_length = 7
    if polyorder is None:
        polyorder = 2

    try:
        window_length = int(window_length)
        polyorder = int(polyorder)
    except (TypeError, ValueError):
        return intensity

    if window_length < 3:
        return intensity

    if window_length % 2 == 0:
        window_length -= 1

    intensity = np.asarray(intensity, dtype=float)
    n_points = intensity.size
    if n_points < 3:
        return intensity

    if window_length > n_points:
        window_length = n_points if n_points % 2 == 1 else n_points - 1

    if window_length < 3:
        return intensity

    if polyorder < 0:
        polyorder = 0
    if polyorder >= window_length:
        polyorder = window_length - 1

    if _savgol_filter is not None:
        smoothed = _savgol_filter(intensity, window_length, polyorder, mode='interp')
    else:
        smoothed = _savgol_filter_numpy(intensity, window_length, polyorder)

    return np.maximum(smoothed, 1.0)


def apply_lttb_downsampling(scan_time, intensity, n_out=100):
    global _LTTBC_MISSING_WARNED

    if n_out is None:
        n_out = 100

    try:
        n_out = int(n_out)
    except (TypeError, ValueError):
        return scan_time, intensity

    scan_time = np.asarray(scan_time, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    n_points = scan_time.size
    if n_points == 0 or n_out <= 0 or n_points <= n_out:
        return scan_time, intensity

    if _lttbc is None:
        if not _LTTBC_MISSING_WARNED:
            logger.warning("LTTB downsampling skipped: 'lttbc' is not available in this environment.")
            _LTTBC_MISSING_WARNED = True
        return scan_time, intensity

    downsampled_x, downsampled_y = _lttbc.downsample(scan_time, intensity, n_out)
    if len(downsampled_x) == 0:
        return scan_time, intensity

    return downsampled_x, downsampled_y


def calculate_rt_alignment(chrom_df, rt_min, rt_max):
    """
    Calculate RT shifts based on peak apex within the RT span.
    
    Args:
        chrom_df: DuckDB LazyFrame with chromatogram data
        rt_min: Minimum RT of the span
        rt_max: Maximum RT of the span
    
    Returns:
        dict: {ms_file_label: shift_value}
    """
    shifts = {}
    apex_rts = []
    ms_file_labels = []
    
    for row in chrom_df.iter_rows(named=True):
        scan_time = np.array(row['scan_time_sliced'])
        intensity = np.array(row['intensity_sliced'])
        
        # Find apex within RT span
        mask = (scan_time >= rt_min) & (scan_time <= rt_max)
        if mask.any():
            rt_in_range = scan_time[mask]
            int_in_range = intensity[mask]
            apex_idx = int_in_range.argmax()
            apex_rt = rt_in_range[apex_idx]
            apex_rts.append(apex_rt)
        else:
            apex_rts.append(None)
        
        ms_file_labels.append(row['ms_file_label'])
    
    # Calculate reference (median of valid apex RTs)
    valid_apex_rts = [rt for rt in apex_rts if rt is not None]
    if valid_apex_rts:
        reference_rt = np.median(valid_apex_rts)
        
        # Debug: log apex positions
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"RT Alignment: RT span = [{rt_min:.2f}, {rt_max:.2f}]")
        logger.info(f"RT Alignment: Found {len(valid_apex_rts)} peaks with apexes: {[f'{rt:.2f}' for rt in valid_apex_rts[:5]]}")
        logger.info(f"RT Alignment: Reference RT (median) = {reference_rt:.2f}")
        
        # Calculate shifts
        for i, ms_file_label in enumerate(ms_file_labels):
            if apex_rts[i] is not None:
                shift = reference_rt - apex_rts[i]
                shifts[ms_file_label] = shift
            else:
                shifts[ms_file_label] = 0.0
    
    return shifts


def calculate_shifts_per_sample_type(chrom_df, shifts_dict):
    """
    Group shifts by sample_type and calculate median.
    
    Args:
        chrom_df: DuckDB LazyFrame with chromatogram data (must have sample_type column)
        shifts_dict: Dictionary of {ms_file_label: shift_value}
    
    Returns:
        dict: {sample_type: median_shift}
    """
    sample_type_shifts = {}
    
    for row in chrom_df.iter_rows(named=True):
        sample_type = row['sample_type']
        shift = shifts_dict.get(row['ms_file_label'], 0.0)
        
        if sample_type not in sample_type_shifts:
            sample_type_shifts[sample_type] = []
        sample_type_shifts[sample_type].append(shift)
    
    # Calculate median shift per sample type
    return {st: float(np.median(shifts)) for st, shifts in sample_type_shifts.items()}


def generate_chromatogram_traces(
    chrom_df,
    use_megatrace=False,
    rt_alignment_shifts=None,
    ms_type=None,
    smoothing_params=None,
    downsample_params=None,
):
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')

    legend_groups = set()
    traces = []
    
    if chrom_df is None or len(chrom_df) == 0:
        return traces, x_min, x_max, y_min, y_max

    # MS2/SRM uses min_peak_width=1 and higher baseline to filter noise
    # MS1 uses defaults (min_peak_width=3, baseline=1.0)
    if ms_type == 'ms2':
        sparsify_kwargs = {'w': 1, 'baseline': 10.0, 'eps': 0.0, 'min_peak_width': 1}
    else:
        sparsify_kwargs = {'w': 1, 'baseline': 1.0, 'eps': 0.0}

    smoothing_enabled = bool(smoothing_params and smoothing_params.get('enabled'))
    smoothing_window = smoothing_params.get('window_length', 7) if smoothing_enabled else None
    smoothing_order = smoothing_params.get('polyorder', 2) if smoothing_enabled else None
    downsample_enabled = bool(downsample_params and downsample_params.get('enabled') and ms_type == 'ms1')
    downsample_n_out = downsample_params.get('n_out', 100) if downsample_enabled else None

    logger.info(
        "Trace pipeline: savgol=%s, lttb=%s (n_out=%s), sparsify",
        smoothing_enabled,
        downsample_enabled,
        downsample_n_out,
    )

    trace_logs = 0
    total_traces = 0

    if not use_megatrace:
        # ------------------------------------
        # ------------------------------------
        # NORMAL MODE: one trace per chromatogram
        # ------------------------------------
        # First, count samples per sample_type to sort by group size
        rows_list = list(chrom_df.iter_rows(named=True))
        sample_type_counts = {}
        for row in rows_list:
            stype = row['sample_type']
            sample_type_counts[stype] = sample_type_counts.get(stype, 0) + 1
        
        # Sort rows: larger sample_type groups first, so smaller groups are drawn last (on top)
        rows_sorted = sorted(rows_list, key=lambda r: sample_type_counts.get(r['sample_type'], 0), reverse=True)
        
        total_traces = len(rows_sorted)
        for i, row in enumerate(rows_sorted):

            scan_time = row['scan_time_sliced']
            intensity = row['intensity_sliced']
            n_start = len(scan_time)
            if smoothing_enabled:
                intensity = apply_savgol_smoothing(
                    intensity, window_length=smoothing_window, polyorder=smoothing_order
                )
            n_smooth = len(intensity)
            if downsample_enabled:
                scan_time, intensity = apply_lttb_downsampling(
                    scan_time, intensity, n_out=downsample_n_out
                )
            n_lttb = len(scan_time)

            scan_time_sparse, intensity_sparse = sparsify_chrom(
                scan_time, intensity, **sparsify_kwargs
            )
            n_sparse = len(scan_time_sparse)

            if trace_logs < LOG_TRACE_POINT_LIMIT:
                trace_label = row['label'] or row['ms_file_label']
                logger.info(
                    "Trace points %s: start=%d smooth=%d lttb=%d sparse=%d",
                    trace_label,
                    n_start,
                    n_smooth,
                    n_lttb,
                    n_sparse,
                )
                trace_logs += 1
            
            # Apply RT alignment shift if provided
            if rt_alignment_shifts and row['ms_file_label'] in rt_alignment_shifts:
                shift = rt_alignment_shifts[row['ms_file_label']]
                scan_time_sparse = [t + shift for t in scan_time_sparse]
            
            trace = {
                'type': 'scattergl',
                'mode': 'lines',
                'x': scan_time_sparse,
                'y': intensity_sparse,
                'line': {'color': row['color']},
                'name': row['label'] or row['ms_file_label'],
                'legendgroup': row['sample_type'],
                'hoverlabel': dict(namelength=-1)
            }

            if row['sample_type'] not in legend_groups:
                trace['legendgrouptitle'] = {'text': row['sample_type']}
                legend_groups.add(row['sample_type'])

            traces.append(trace)
            
            # Check for None values before min/max calculations
            if row['scan_time_min_in_range'] is not None:
                x_min = min(x_min, row['scan_time_min_in_range'])
            if row['scan_time_max_in_range'] is not None:
                x_max = max(x_max, row['scan_time_max_in_range'])
            if row['intensity_min_in_range'] is not None:
                y_min = min(y_min, row['intensity_min_in_range'])
            if row['intensity_max_in_range'] is not None:
                y_max = max(y_max, row['intensity_max_in_range'])

    else:
        # ------------------------------------
        # ------------------------------------
        # REDUCED MODE: one trace per sample_type
        # ------------------------------------
        grouped = {}
        color_counts = {}  # Track color counts per sample_type
        for row in chrom_df.iter_rows(named=True):
            total_traces += 1
            stype = row['sample_type']
            if stype not in grouped:
                grouped[stype] = {
                    'x': [],
                    'y': [],
                    'count': 0
                }
                color_counts[stype] = {}
            
            # Count colors for this sample type
            color = row['color']
            if color not in color_counts[stype]:
                color_counts[stype][color] = 0
            color_counts[stype][color] += 1

            # Sparsify individually before joining (optional, to save more)
            scan_time = row['scan_time_sliced']
            intensity = row['intensity_sliced']
            n_start = len(scan_time)
            if smoothing_enabled:
                intensity = apply_savgol_smoothing(
                    intensity, window_length=smoothing_window, polyorder=smoothing_order
                )
            n_smooth = len(intensity)
            if downsample_enabled:
                scan_time, intensity = apply_lttb_downsampling(
                    scan_time, intensity, n_out=downsample_n_out
                )
            n_lttb = len(scan_time)

            st, ints = sparsify_chrom(
                scan_time, intensity, **sparsify_kwargs
            )
            n_sparse = len(st)

            if trace_logs < LOG_TRACE_POINT_LIMIT:
                trace_label = row['label'] or row['ms_file_label']
                logger.info(
                    "Trace points %s: start=%d smooth=%d lttb=%d sparse=%d",
                    trace_label,
                    n_start,
                    n_smooth,
                    n_lttb,
                    n_sparse,
                )
                trace_logs += 1

            # Apply RT alignment shift if provided
            if rt_alignment_shifts and row['ms_file_label'] in rt_alignment_shifts:
                shift = rt_alignment_shifts[row['ms_file_label']]
                st = [t + shift for t in st]

            grouped[stype]['x'].append(st)
            grouped[stype]['x'].append([None])
            grouped[stype]['y'].append(ints)
            grouped[stype]['y'].append([None])
            grouped[stype]['count'] += 1

            if row['scan_time_min_in_range'] is not None:
                x_min = min(x_min, row['scan_time_min_in_range'])
            if row['scan_time_max_in_range'] is not None:
                x_max = max(x_max, row['scan_time_max_in_range'])
            if row['intensity_min_in_range'] is not None:
                y_min = min(y_min, row['intensity_min_in_range'])
            if row['intensity_max_in_range'] is not None:
                y_max = max(y_max, row['intensity_max_in_range'])

        # Build traces - sort by count descending so smaller groups are drawn last (on top)
        sorted_groups = sorted(grouped.items(), key=lambda x: x[1]['count'], reverse=True)
        for stype, data in sorted_groups:
            # Flatten arrays
            x_flat = list(itertools.chain(*data['x']))
            y_flat = list(itertools.chain(*data['y']))
            
            # Get the most common color for this sample type
            stype_colors = color_counts.get(stype, {})
            if stype_colors:
                most_common_color = max(stype_colors, key=stype_colors.get)
            else:
                most_common_color = None  # Fallback

            trace = {
                'type': 'scattergl',
                'mode': 'lines',
                'x': x_flat,
                'y': y_flat,
                'line': {'color': most_common_color},
                'name': f"{stype} ({data['count']})",
                'legendgroup': stype,
                'hoverlabel': dict(namelength=-1),
                'connectgaps': False,
                'hoverinfo': 'skip',
                'hovertemplate': None
            }
            traces.append(trace)
            
    if total_traces > LOG_TRACE_POINT_LIMIT:
        logger.info(
            "Trace points logging truncated: showing first %d of %d traces",
            LOG_TRACE_POINT_LIMIT,
            total_traces,
        )

    # Fix infinite values if data was empty or all Nones
    if x_min == float('inf'): x_min = 0
    if x_max == float('-inf'): x_max = 1
    if y_min == float('inf'): y_min = 0
    if y_max == float('-inf'): y_max = 1
            
    return traces, x_min, x_max, y_min, y_max
