"""
Tests for RT Span Optimizer module
"""

import pytest
import numpy as np

from ms_mint_app.rt_span_optimizer import (
    optimize_rt_span,
    combine_chromatograms,
    optimize_rt_spans_batch,
)
from ms_mint_app.duckdb_manager import duckdb_connection


class TestOptimizeRtSpan:
    """Tests for the optimize_rt_span function."""

    def test_simple_symmetric_peak(self):
        """Test with a simple symmetric Gaussian-like peak."""
        # Create a synthetic peak centered at RT=60
        scan_time = np.linspace(40, 80, 100)
        intensity = 1000 * np.exp(-0.5 * ((scan_time - 60) / 5) ** 2)
        
        rt_min, rt_max, apex_rt = optimize_rt_span(scan_time, intensity, expected_rt=60.0)
        
        # Apex should be near 60
        assert abs(apex_rt - 60.0) < 1.0
        # Span should be symmetric around apex
        assert rt_min < 60.0 < rt_max
        # Width should be reasonable (roughly 2-3 sigma on each side at 10% height)
        assert 10 < (rt_max - rt_min) < 40

    def test_tailed_peak(self):
        """Test with an asymmetric tailed peak."""
        scan_time = np.linspace(40, 100, 150)
        # Create tailed peak: fast rise, slow decay
        intensity = np.zeros_like(scan_time)
        for i, t in enumerate(scan_time):
            if t < 60:
                intensity[i] = 1000 * np.exp(-0.5 * ((t - 60) / 3) ** 2)
            else:
                intensity[i] = 1000 * np.exp(-0.5 * ((t - 60) / 10) ** 2)
        
        rt_min, rt_max, apex_rt = optimize_rt_span(scan_time, intensity, expected_rt=60.0)
        
        # Apex should be near 60
        assert abs(apex_rt - 60.0) < 2.0
        # Right side should be wider than left side (tailed)
        left_width = apex_rt - rt_min
        right_width = rt_max - apex_rt
        assert right_width > left_width

    def test_min_width_constraint(self):
        """Test that minimum width constraint is applied."""
        # Very narrow peak
        scan_time = np.linspace(55, 65, 50)
        intensity = 1000 * np.exp(-0.5 * ((scan_time - 60) / 0.5) ** 2)
        
        rt_min, rt_max, apex_rt = optimize_rt_span(
            scan_time, intensity, expected_rt=60.0, min_width=10.0
        )
        
        # Width should be at least min_width
        assert (rt_max - rt_min) >= 10.0

    def test_max_width_constraint(self):
        """Test that maximum width constraint is applied."""
        # Very wide peak
        scan_time = np.linspace(0, 200, 200)
        intensity = 1000 * np.exp(-0.5 * ((scan_time - 100) / 50) ** 2)
        
        rt_min, rt_max, apex_rt = optimize_rt_span(
            scan_time, intensity, expected_rt=100.0, max_width=60.0
        )
        
        # Width should be at most max_width
        assert (rt_max - rt_min) <= 60.0

    def test_insufficient_data(self):
        """Test behavior with too few data points."""
        scan_time = np.array([60.0, 61.0])
        intensity = np.array([100.0, 50.0])
        
        rt_min, rt_max, apex_rt = optimize_rt_span(scan_time, intensity, expected_rt=60.0)
        
        # Should fall back to expected_rt with default width
        assert apex_rt == 60.0
        assert rt_min < 60.0 < rt_max

    def test_no_peak_in_window(self):
        """Test when there's no peak near expected RT."""
        scan_time = np.linspace(0, 50, 100)
        intensity = 1000 * np.exp(-0.5 * ((scan_time - 25) / 5) ** 2)
        
        # Expected RT is outside the data range
        rt_min, rt_max, apex_rt = optimize_rt_span(scan_time, intensity, expected_rt=100.0)
        
        # Should handle gracefully
        assert rt_min is not None
        assert rt_max is not None


class TestCombineChromatograms:
    """Tests for the combine_chromatograms function."""

    def test_single_chromatogram(self):
        """Test with a single chromatogram."""
        chromatograms = [
            {'scan_time': [1.0, 2.0, 3.0], 'intensity': [10.0, 100.0, 50.0]}
        ]
        
        time, intensity = combine_chromatograms(chromatograms)
        
        np.testing.assert_array_equal(time, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(intensity, [10.0, 100.0, 50.0])

    def test_multiple_chromatograms_max(self):
        """Test combining multiple chromatograms with max method."""
        chromatograms = [
            {'scan_time': [1.0, 2.0, 3.0], 'intensity': [10.0, 100.0, 50.0]},
            {'scan_time': [1.0, 2.0, 3.0], 'intensity': [20.0, 80.0, 60.0]},
        ]
        
        time, intensity = combine_chromatograms(chromatograms, method="max")
        
        np.testing.assert_array_equal(time, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(intensity, [20.0, 100.0, 60.0])

    def test_multiple_chromatograms_mean(self):
        """Test combining multiple chromatograms with mean method."""
        chromatograms = [
            {'scan_time': [1.0, 2.0, 3.0], 'intensity': [10.0, 100.0, 50.0]},
            {'scan_time': [1.0, 2.0, 3.0], 'intensity': [20.0, 80.0, 60.0]},
        ]
        
        time, intensity = combine_chromatograms(chromatograms, method="mean")
        
        np.testing.assert_array_equal(time, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(intensity, [15.0, 90.0, 55.0])

    def test_empty_list(self):
        """Test with empty chromatogram list."""
        time, intensity = combine_chromatograms([])
        
        assert len(time) == 0
        assert len(intensity) == 0

    def test_different_time_grids(self):
        """Test combining chromatograms with different time points."""
        chromatograms = [
            {'scan_time': [1.0, 2.0, 3.0], 'intensity': [10.0, 100.0, 50.0]},
            {'scan_time': [1.5, 2.5, 3.5], 'intensity': [20.0, 80.0, 60.0]},
        ]
        
        time, intensity = combine_chromatograms(chromatograms)
        
        # Should have all unique time points
        assert len(time) == 6
        assert 1.0 in time
        assert 1.5 in time


class TestOptimizeRtSpansBatch:
    """Integration tests for optimize_rt_spans_batch."""

    @pytest.fixture
    def temp_wdir(self, tmp_path):
        """Create a temporary workspace with test data."""
        wdir = tmp_path / "workspace"
        wdir.mkdir(parents=True)
        
        with duckdb_connection(wdir) as conn:
            # Insert target with rt_auto_adjusted = TRUE
            conn.execute("""
                INSERT INTO targets (peak_label, rt, rt_min, rt_max, mz_mean, ms_type, rt_auto_adjusted)
                VALUES ('TestTarget1', 60.0, 55.0, 65.0, 100.0, 'ms1', TRUE)
            """)
            
            # Insert sample
            conn.execute("""
                INSERT INTO samples (ms_file_label, sample_type, label, use_for_optimization, ms_type)
                VALUES ('File1', 'TypeA', 'Label1', TRUE, 'ms1')
            """)
            
            # Insert chromatogram with a clear peak
            # Gaussian-like peak centered at 60s
            scan_times = list(np.linspace(40, 80, 100))
            intensities = list(1000 * np.exp(-0.5 * ((np.array(scan_times) - 60) / 5) ** 2))
            
            conn.execute("""
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity)
                VALUES ('TestTarget1', 'File1', ?, ?)
            """, [scan_times, intensities])
        
        return str(wdir)

    def test_batch_optimization(self, temp_wdir):
        """Test that batch optimization updates rt_min, rt_max, and rt."""
        with duckdb_connection(temp_wdir) as conn:
            # Run optimization
            updated_count = optimize_rt_spans_batch(conn)
            
            assert updated_count == 1
            
            # Verify the target was updated
            result = conn.execute("""
                SELECT rt, rt_min, rt_max, rt_auto_adjusted
                FROM targets
                WHERE peak_label = 'TestTarget1'
            """).fetchone()
            
            rt, rt_min, rt_max, rt_auto_adjusted = result
            
            # RT should be near peak apex (60)
            assert abs(rt - 60.0) < 3.0
            
            # RT span should bracket the peak
            assert rt_min < rt < rt_max
            
            # Flag should be reset
            assert rt_auto_adjusted is False

    def test_no_targets_to_optimize(self, temp_wdir):
        """Test when no targets need optimization."""
        with duckdb_connection(temp_wdir) as conn:
            # Reset the flag
            conn.execute("UPDATE targets SET rt_auto_adjusted = FALSE")
            
            updated_count = optimize_rt_spans_batch(conn)
            
            assert updated_count == 0

    def test_missing_chromatogram(self, temp_wdir):
        """Test handling of targets without chromatograms."""
        with duckdb_connection(temp_wdir) as conn:
            # Add another target without chromatogram
            conn.execute("""
                INSERT INTO targets (peak_label, rt, mz_mean, ms_type, rt_auto_adjusted)
                VALUES ('TestTarget2', 100.0, 200.0, 'ms1', TRUE)
            """)
            
            # Should handle gracefully (only optimize the one with data)
            updated_count = optimize_rt_spans_batch(conn)
            
            assert updated_count == 1  # Only TestTarget1
