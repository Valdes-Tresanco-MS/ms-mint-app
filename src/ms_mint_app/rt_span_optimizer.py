"""
RT Span Optimizer for MS-MINT App

This module provides adaptive RT span detection using chromatogram data
to automatically determine optimal rt_min and rt_max values.

Algorithm: Peak boundary detection at a threshold percentage of peak height,
which naturally handles asymmetric/tailed peaks without requiring parameter estimation.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


def optimize_rt_span(
    scan_time: np.ndarray,
    intensity: np.ndarray,
    expected_rt: float,
    min_width: float = 5.0,
    max_width: float = 120.0,
    threshold_pct: float = 0.10,
    sigma_smooth: float = 2.0,
    apex_search_window: float = 15.0,
) -> Tuple[float, float, float]:
    """
    Find optimal rt_min, rt_max from chromatogram data using adaptive peak detection.

    Algorithm:
    1. Smooth the signal with a Gaussian filter to reduce noise
    2. Find the peak apex NEAR the expected RT (within apex_search_window)
    3. Determine peak boundaries at threshold_pct of peak height (default 10%)
    4. Apply min/max width constraints

    Args:
        scan_time: Array of retention times (seconds)
        intensity: Array of intensity values
        expected_rt: Expected retention time (target RT) - this is TRUSTED
        min_width: Minimum allowed peak width in seconds (default 5s)
        max_width: Maximum allowed peak width in seconds (default 120s)
        threshold_pct: Fraction of peak height for boundary detection (default 0.10 = 10%)
        sigma_smooth: Gaussian smoothing sigma in data points (default 2.0)
        apex_search_window: How far from expected_rt to search for apex (default ±15s)

    Returns:
        Tuple of (rt_min, rt_max, apex_rt)
    """
    if len(scan_time) < 3 or len(intensity) < 3:
        # Not enough data points, return expected RT with min_width
        half_width = min_width / 2
        return expected_rt - half_width, expected_rt + half_width, expected_rt

    # Ensure arrays are numpy and sorted by time
    scan_time = np.asarray(scan_time, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    sort_idx = np.argsort(scan_time)
    scan_time = scan_time[sort_idx]
    intensity = intensity[sort_idx]

    # Apply Gaussian smoothing to reduce noise
    if len(intensity) > 5:
        intensity_smooth = gaussian_filter1d(intensity, sigma=sigma_smooth)
    else:
        intensity_smooth = intensity.copy()

    # Find the apex: maximum intensity within a NARROW window around expected_rt
    # This respects the user's RT hint instead of finding global max
    search_mask = np.abs(scan_time - expected_rt) <= apex_search_window
    if not np.any(search_mask):
        # No data in search window, use full data but warn
        logger.warning(f"No data within ±{apex_search_window}s of expected_rt={expected_rt:.1f}s")
        search_mask = np.ones(len(scan_time), dtype=bool)

    # Find local maximum near expected_rt
    search_intensities = intensity_smooth.copy()
    search_intensities[~search_mask] = -np.inf  # Exclude points outside window

    apex_idx = np.argmax(search_intensities)
    apex_rt = scan_time[apex_idx]
    apex_intensity = intensity_smooth[apex_idx]

    if apex_intensity <= 0:
        # No valid peak, return expected RT with min_width
        half_width = min_width / 2
        return expected_rt - half_width, expected_rt + half_width, expected_rt

    # Calculate threshold intensity
    # Use baseline as the minimum in the search window
    baseline = np.min(intensity_smooth[search_mask])
    peak_height = apex_intensity - baseline
    threshold_intensity = baseline + peak_height * threshold_pct

    # Find left boundary: walk left from apex until below threshold
    rt_min = scan_time[0]  # Default to start
    for i in range(apex_idx - 1, -1, -1):
        if intensity_smooth[i] < threshold_intensity:
            rt_min = scan_time[i]
            break

    # Find right boundary: walk right from apex until below threshold
    rt_max = scan_time[-1]  # Default to end
    for i in range(apex_idx + 1, len(scan_time)):
        if intensity_smooth[i] < threshold_intensity:
            rt_max = scan_time[i]
            break

    # Apply width constraints
    current_width = rt_max - rt_min
    if current_width < min_width:
        # Expand symmetrically
        expand = (min_width - current_width) / 2
        rt_min -= expand
        rt_max += expand
    elif current_width > max_width:
        # Contract symmetrically around apex
        rt_min = apex_rt - max_width / 2
        rt_max = apex_rt + max_width / 2

    # Ensure boundaries don't exceed data range
    rt_min = max(rt_min, scan_time[0])
    rt_max = min(rt_max, scan_time[-1])

    logger.debug(
        f"RT span optimized: apex={apex_rt:.1f}s, "
        f"span=[{rt_min:.1f}, {rt_max:.1f}]s, width={rt_max - rt_min:.1f}s"
    )

    return rt_min, rt_max, apex_rt


def combine_chromatograms(
    chromatograms: list,
    method: str = "max"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine multiple chromatograms into one representative signal.

    Args:
        chromatograms: List of dicts with 'scan_time' and 'intensity' arrays
        method: Combination method - 'max', 'mean', or 'median'

    Returns:
        Tuple of (combined_scan_time, combined_intensity)
    """
    if not chromatograms:
        return np.array([]), np.array([])

    if len(chromatograms) == 1:
        return (
            np.asarray(chromatograms[0]['scan_time']),
            np.asarray(chromatograms[0]['intensity'])
        )

    # Collect all unique time points
    all_times = set()
    for chrom in chromatograms:
        all_times.update(chrom['scan_time'])

    combined_time = np.array(sorted(all_times))

    # Interpolate each chromatogram to the common time grid
    interpolated = []
    for chrom in chromatograms:
        t = np.asarray(chrom['scan_time'])
        i = np.asarray(chrom['intensity'])
        if len(t) > 1:
            interp_i = np.interp(combined_time, t, i, left=0, right=0)
            interpolated.append(interp_i)

    if not interpolated:
        return np.array([]), np.array([])

    stacked = np.vstack(interpolated)

    if method == "max":
        combined_intensity = np.max(stacked, axis=0)
    elif method == "mean":
        combined_intensity = np.mean(stacked, axis=0)
    elif method == "median":
        combined_intensity = np.median(stacked, axis=0)
    else:
        combined_intensity = np.max(stacked, axis=0)

    return combined_time, combined_intensity


def optimize_rt_spans_batch(
    conn,
    threshold_pct: float = 0.10,
    min_width: float = 5.0,
    max_width: float = 120.0,
) -> int:
    """
    Optimize RT spans for all targets that were auto-adjusted.

    This function:
    1. Finds all targets with rt_auto_adjusted = TRUE
    2. For each target, combines chromatograms across all files
    3. Detects peak boundaries using adaptive method
    4. Updates rt_min, rt_max, and rt in the database

    Args:
        conn: Active DuckDB connection
        threshold_pct: Fraction of peak height for boundary detection
        min_width: Minimum allowed peak width in seconds
        max_width: Maximum allowed peak width in seconds

    Returns:
        Number of targets updated
    """
    # Get all targets that need optimization
    targets_to_optimize = conn.execute("""
        SELECT peak_label, rt
        FROM targets
        WHERE rt_auto_adjusted = TRUE
    """).fetchall()

    if not targets_to_optimize:
        logger.info("No targets require RT span optimization")
        return 0

    updated_count = 0

    for peak_label, expected_rt in targets_to_optimize:
        # Get all chromatograms for this target
        chrom_data = conn.execute("""
            SELECT scan_time, intensity
            FROM chromatograms
            WHERE peak_label = ?
        """, [peak_label]).fetchall()

        if not chrom_data:
            logger.warning(f"No chromatograms found for target '{peak_label}'")
            continue

        # Convert to list of dicts
        chromatograms = [
            {'scan_time': row[0], 'intensity': row[1]}
            for row in chrom_data
        ]

        # Combine chromatograms
        combined_time, combined_intensity = combine_chromatograms(
            chromatograms, method="max"
        )

        if len(combined_time) < 3:
            logger.warning(f"Insufficient data for target '{peak_label}'")
            continue

        # Optimize RT span
        try:
            rt_min, rt_max, apex_rt = optimize_rt_span(
                combined_time,
                combined_intensity,
                expected_rt or np.median(combined_time),
                min_width=min_width,
                max_width=max_width,
                threshold_pct=threshold_pct,
            )

            # Update database
            conn.execute("""
                UPDATE targets
                SET rt_min = ?,
                    rt_max = ?,
                    rt = ?,
                    rt_auto_adjusted = FALSE
                WHERE peak_label = ?
            """, [rt_min, rt_max, apex_rt, peak_label])

            updated_count += 1
            logger.debug(
                f"Optimized RT span for '{peak_label}': "
                f"rt={apex_rt:.1f}s, span=[{rt_min:.1f}, {rt_max:.1f}]s"
            )

        except Exception as e:
            logger.error(f"Failed to optimize RT span for '{peak_label}': {e}")

    logger.info(f"Optimized RT spans for {updated_count} targets")
    return updated_count
