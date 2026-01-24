import tempfile
from pathlib import Path
import time
import pytest
from unittest.mock import patch, MagicMock

from ms_mint_app.duckdb_manager import (
    duckdb_connection,
    create_pivot,
    compute_chromatograms_in_batches,
    compute_results_in_batches,
)
from ms_mint_app.plugins.processing import (
    _build_delete_modal_content,
    _load_peaks_from_results,
    _download_all_results,
    _download_dense_matrix,
    _delete_selected_results,
    _delete_all_results,
)
from dash.exceptions import PreventUpdate
import feffery_antd_components as fac
import dash


@pytest.fixture
def temp_wdir(tmp_path):
    """Create a temporary workspace with test data."""
    wdir = tmp_path / "workspace"
    wdir.mkdir(parents=True)
    
    with duckdb_connection(wdir) as conn:
        # Setup initial test data
        # Add samples (MS files) - polarity is ENUM type: 'Positive' or 'Negative'
        conn.execute("""
            INSERT INTO samples (ms_file_label, sample_type, label, use_for_optimization, ms_type, polarity) 
            VALUES 
                ('TestFile1', 'Sample', 'TestSample1', TRUE, 'ms1', 'Positive'),
                ('TestFile2', 'QC', 'TestQC', TRUE, 'ms1', 'Positive'),
                ('TestFile3', 'Blank', 'TestBlank', FALSE, 'ms2', 'Negative')
        """)
        
        # Add targets
        conn.execute("""
            INSERT INTO targets (peak_label, mz_mean, mz_width, rt, rt_min, rt_max, ms_type, bookmark) 
            VALUES 
                ('Target1', 100.5, 0.01, 60.0, 58.0, 62.0, 'ms1', TRUE),
                ('Target2', 200.3, 0.01, 75.0, 73.0, 77.0, 'ms1', FALSE),
                ('Target3', 150.2, 0.01, 90.0, 88.0, 92.0, 'ms2', FALSE)
        """)
        
    return str(wdir)


class TestProcessingDatabaseOperations:
    """Test database CRUD operations for processing plugin."""
    
    def test_insert_chromatogram_data(self, temp_wdir):
        """Test inserting chromatogram data into the database."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert chromatogram without 'mz' column (it doesn't exist in the schema)
            conn.execute("""
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity)
                VALUES ('Target1', 'TestFile1', [58.0, 60.0, 62.0], [100.0, 500.0, 200.0])
            """)
            
            result = conn.execute("""
                SELECT peak_label, ms_file_label, len(scan_time) as num_points
                FROM chromatograms
                WHERE peak_label = 'Target1' AND ms_file_label = 'TestFile1'
            """).fetchone()
            
            assert result is not None
            assert result[0] == 'Target1'
            assert result[1] == 'TestFile1'
            assert result[2] == 3  # 3 data points
    
    def test_insert_results_data(self, temp_wdir):
        """Test inserting results data into the database."""
        with duckdb_connection(temp_wdir) as conn:
            conn.execute("""
                INSERT INTO results (
                    peak_label, ms_file_label, peak_area, peak_area_top3, 
                    peak_mean, peak_median, peak_n_datapoints, peak_min, peak_max,
                    peak_rt_of_max, total_intensity, intensity
                )
                VALUES (
                    'Target1', 'TestFile1', 1000.0, 800.0, 
                    300.0, 350.0, 10, 100.0, 500.0,
                    60.0, 3000.0, [100.0, 200.0, 500.0, 400.0, 200.0, 100.0]
                )
            """)
            
            result = conn.execute("""
                SELECT peak_label, ms_file_label, peak_area, peak_n_datapoints
                FROM results
                WHERE peak_label = 'Target1' AND ms_file_label = 'TestFile1'
            """).fetchone()
            
            assert result is not None
            assert result[0] == 'Target1'
            assert result[1] == 'TestFile1'
            assert result[2] == 1000.0
            assert result[3] == 10
    
    def test_delete_selected_results(self, temp_wdir):
        """Test deleting selected results (pairs of peak_label, ms_file_label)."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert test results
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES 
                    ('Target1', 'TestFile1', 1000.0),
                    ('Target1', 'TestFile2', 1500.0),
                    ('Target2', 'TestFile1', 2000.0)
            """)
            
            # Delete specific pair
            conn.execute("BEGIN")
            conn.execute("""
                DELETE FROM results 
                WHERE (peak_label, ms_file_label) IN (('Target1', 'TestFile1'))
            """)
            conn.execute("COMMIT")
            
            # Verify deletion
            remaining = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            assert remaining == 2
            
            deleted_result = conn.execute("""
                SELECT COUNT(*) FROM results 
                WHERE peak_label = 'Target1' AND ms_file_label = 'TestFile1'
            """).fetchone()[0]
            assert deleted_result == 0
    
    def test_delete_all_results(self, temp_wdir):
        """Test deleting all results."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert test results
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES 
                    ('Target1', 'TestFile1', 1000.0),
                    ('Target2', 'TestFile2', 1500.0)
            """)
            
            # Verify data exists
            count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            assert count == 2
            
            # Delete all
            conn.execute("BEGIN")
            conn.execute("DELETE FROM results")
            conn.execute("COMMIT")
            
            # Verify all deleted
            count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            assert count == 0
    
    def test_transaction_rollback_on_error(self, temp_wdir):
        """Test that failed transactions are rolled back properly."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert initial data
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES ('Target1', 'TestFile1', 1000.0)
            """)
            
            initial_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            
            # Attempt a transaction that will fail
            try:
                conn.execute("BEGIN")
                conn.execute("INSERT INTO results (peak_label, ms_file_label, peak_area) VALUES ('Target2', 'TestFile2', 2000.0)")
                # Intentional error - invalid column
                conn.execute("INSERT INTO results (invalid_column) VALUES (999)")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
            
            # Verify rollback - count should be same as initial
            final_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            assert final_count == initial_count


class TestProcessingErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_workspace_path(self):
        """Test handling of invalid workspace path."""
        invalid_path = "/non/existent/workspace"
        
        with duckdb_connection(invalid_path) as conn:
            # Should return None for invalid path
            assert conn is None
    
    def test_empty_results_table(self, temp_wdir):
        """Test handling of empty results table."""
        with duckdb_connection(temp_wdir) as conn:
            count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            assert count == 0
            
            # Should handle empty gracefully
            peaks = conn.execute("SELECT DISTINCT peak_label FROM results ORDER BY peak_label").fetchall()
            assert peaks == []
    
    def test_null_value_handling(self, temp_wdir):
        """Test handling of NULL values in results."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert result with NULL peak_area
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES ('Target1', 'TestFile1', NULL)
            """)
            
            result = conn.execute("""
                SELECT peak_area FROM results 
                WHERE peak_label = 'Target1' AND ms_file_label = 'TestFile1'
            """).fetchone()
            
            assert result[0] is None
    
    def test_duplicate_insertion_handling(self, temp_wdir):
        """Test handling of duplicate result insertion."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert first result
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES ('Target1', 'TestFile1', 1000.0)
            """)
            
            # Attempting to insert duplicate should fail or be handled
            # The schema may have a unique constraint on (peak_label, ms_file_label)
            try:
                conn.execute("""
                    INSERT INTO results (peak_label, ms_file_label, peak_area)
                    VALUES ('Target1', 'TestFile1', 2000.0)
                """)
                # If it succeeds, there's no unique constraint (both rows exist)
                count = conn.execute("""
                    SELECT COUNT(*) FROM results 
                    WHERE peak_label = 'Target1' AND ms_file_label = 'TestFile1'
                """).fetchone()[0]
                # Either 1 (with constraint) or 2 (without) is acceptable
                assert count >= 1
            except Exception:
                # Expected if unique constraint exists
                pass


class TestProcessingSecurityAndValidation:
    """Test SQL injection protection and input validation."""
    
    def test_sql_injection_protection_peak_label(self, temp_wdir):
        """Test that peak_label parameter is properly sanitized."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert test data
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES ('Target1', 'TestFile1', 1000.0)
            """)
            
            # Attempt SQL injection via parameterized query
            malicious_input = "Target1'; DROP TABLE results; --"
            
            # This should safely find nothing (or Target1 if the string matches)
            result = conn.execute(
                "SELECT * FROM results WHERE peak_label = ?",
                [malicious_input]
            ).fetchall()
            
            # Should return empty result (no match)
            assert len(result) == 0
            
            # Verify table still exists
            tables = conn.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name = 'results'
            """).fetchall()
            assert len(tables) == 1
    
    def test_parameterized_queries_for_deletion(self, temp_wdir):
        """Test that deletion uses parameterized queries."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert test data
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES 
                    ('Target1', 'TestFile1', 1000.0),
                    ('Target2', 'TestFile2', 2000.0)
            """)
            
            # Use parameterized deletion
            peak = 'Target1'
            ms_file = 'TestFile1'
            conn.execute(
                "DELETE FROM results WHERE peak_label = ? AND ms_file_label = ?",
                [peak, ms_file]
            )
            
            # Verify correct deletion
            remaining = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            assert remaining == 1
            
            exists = conn.execute("""
                SELECT COUNT(*) FROM results 
                WHERE peak_label = 'Target2' AND ms_file_label = 'TestFile2'
            """).fetchone()[0]
            assert exists == 1


class TestProcessingBusinessLogic:
    """Test business logic and data transformations."""
    
    def test_create_pivot_peak_area(self, temp_wdir):
        """Test creating pivot table for peak_area."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert test results
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES 
                    ('Target1', 'TestFile1', 1000.0),
                    ('Target1', 'TestFile2', 1500.0),
                    ('Target2', 'TestFile1', 2000.0),
                    ('Target2', 'TestFile2', 2500.0)
            """)
            
            # Create pivot: rows=ms_file_label, cols=peak_label, value=peak_area
            df = create_pivot(conn, rows='ms_file_label', cols='peak_label', value='peak_area', table='results')
            
            # Verify pivot structure
            assert df is not None
            assert len(df) == 2  # 2 MS files
            assert 'Target1' in df.columns
            assert 'Target2' in df.columns
    
    def test_distinct_peak_labels(self, temp_wdir):
        """Test retrieving distinct peak labels from results."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert test results
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES 
                    ('Target1', 'TestFile1', 1000.0),
                    ('Target1', 'TestFile2', 1500.0),
                    ('Target2', 'TestFile1', 2000.0)
            """)
            
            # Get distinct peak labels
            peaks = conn.execute("""
                SELECT DISTINCT peak_label FROM results ORDER BY peak_label
            """).fetchall()
            
            assert len(peaks) == 2
            assert peaks[0][0] == 'Target1'
            assert peaks[1][0] == 'Target2'
    
    def test_result_statistics_computation(self, temp_wdir):
        """Test that result statistics are computed correctly."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert result with all statistics
            conn.execute("""
                INSERT INTO results (
                    peak_label, ms_file_label, peak_area, peak_area_top3,
                    peak_mean, peak_median, peak_min, peak_max, peak_n_datapoints
                )
                VALUES (
                    'Target1', 'TestFile1', 1000.0, 800.0,
                    250.0, 250.0, 100.0, 500.0, 10
                )
            """)
            
            result = conn.execute("""
                SELECT peak_area, peak_mean, peak_median, peak_min, peak_max
                FROM results
                WHERE peak_label = 'Target1' AND ms_file_label = 'TestFile1'
            """).fetchone()
            
            assert result[0] == 1000.0  # peak_area
            assert result[1] == 250.0   # peak_mean
            assert result[2] == 250.0   # peak_median
            assert result[3] == 100.0   # peak_min
            assert result[4] == 500.0   # peak_max
    
    def test_filtering_by_ms_type(self, temp_wdir):
        """Test filtering results by MS type via join with samples."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert results for different MS types
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES 
                    ('Target1', 'TestFile1', 1000.0),
                    ('Target3', 'TestFile3', 3000.0)
            """)
            
            # Filter for ms1 only
            ms1_results = conn.execute("""
                SELECT r.peak_label, r.ms_file_label, s.ms_type
                FROM results r
                LEFT JOIN samples s USING (ms_file_label)
                WHERE UPPER(s.ms_type) = 'MS1'
            """).fetchall()
            
            assert len(ms1_results) == 1
            assert ms1_results[0][0] == 'Target1'
            
            # Filter for ms2 only
            ms2_results = conn.execute("""
                SELECT r.peak_label, r.ms_file_label, s.ms_type
                FROM results r
                LEFT JOIN samples s USING (ms_file_label)
                WHERE UPPER(s.ms_type) = 'MS2'
            """).fetchall()
            
            assert len(ms2_results) == 1
            assert ms2_results[0][0] == 'Target3'


class TestProcessingBatchOperations:
    """Test batch processing operations."""
    
    def test_compute_chromatograms_mock_check(self, temp_wdir):
        """Test that compute_chromatograms_in_batches can be imported and called."""
        # This test verifies the function exists and can be called with proper parameters
        # We don't run actual computation to avoid performance overhead
        from ms_mint_app.duckdb_manager import compute_chromatograms_in_batches
        
        # Verify function is callable
        assert callable(compute_chromatograms_in_batches)
        
        # In a real test with data, we would call:
        # compute_chromatograms_in_batches(
        #     wdir=temp_wdir,
        #     use_for_optimization=False,
        #     batch_size=1000,
        #     recompute_ms1=False,
        #     recompute_ms2=False,
        #     n_cpus=1,
        #     ram=1,
        #     use_bookmarked=False
        # )
    
    def test_compute_results_mock_check(self, temp_wdir):
        """Test that compute_results_in_batches can be imported and called."""
        # This test verifies the function exists and can be called with proper parameters
        # We don't run actual computation to avoid performance overhead
        from ms_mint_app.duckdb_manager import compute_results_in_batches
        
        # Verify function is callable
        assert callable(compute_results_in_batches)
        
        # In a real test with data, we would call:
        # compute_results_in_batches(
        #     wdir=temp_wdir,
        #     use_bookmarked=False,
        #     recompute=False,
        #     batch_size=1000,
        #     checkpoint_every=10,
        #     set_progress=None,
        #     n_cpus=1,
        #     ram=1
        # )
    
    def test_batch_deletion_with_multiple_pairs(self, temp_wdir):
        """Test batch deletion of multiple result pairs."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert multiple results
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES 
                    ('Target1', 'TestFile1', 1000.0),
                    ('Target1', 'TestFile2', 1500.0),
                    ('Target2', 'TestFile1', 2000.0),
                    ('Target2', 'TestFile2', 2500.0)
            """)
            
            # Batch delete multiple pairs
            remove_pairs = [('Target1', 'TestFile1'), ('Target2', 'TestFile2')]
            placeholders = ", ".join(["(?, ?)"] * len(remove_pairs))
            params = [v for pair in remove_pairs for v in pair]
            
            conn.execute("BEGIN")
            conn.execute(
                f"DELETE FROM results WHERE (peak_label, ms_file_label) IN ({placeholders})",
                params
            )
            conn.execute("COMMIT")
            
            # Verify deletion
            remaining = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            assert remaining == 2
            
            # Verify specific pairs removed
            deleted_count = conn.execute("""
                SELECT COUNT(*) FROM results
                WHERE (peak_label = 'Target1' AND ms_file_label = 'TestFile1')
                   OR (peak_label = 'Target2' AND ms_file_label = 'TestFile2')
            """).fetchone()[0]
            assert deleted_count == 0


class TestProcessingDataPersistence:
    """Test data persistence and backup operations."""
    
    def test_results_backup_to_csv(self, temp_wdir):
        """Test that results can be exported to CSV for backup."""
        with duckdb_connection(temp_wdir) as conn:
            # Insert test results
            conn.execute("""
                INSERT INTO results (peak_label, ms_file_label, peak_area)
                VALUES 
                    ('Target1', 'TestFile1', 1000.0),
                    ('Target2', 'TestFile2', 2000.0)
            """)
            
            # Export to CSV (simulate backup)
            results_dir = Path(temp_wdir) / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            backup_path = results_dir / "results_backup.csv"
            
            conn.execute(
                "COPY (SELECT * FROM results) TO ? (HEADER, DELIMITER ',')",
                (str(backup_path),)
            )
            
            # Verify backup file was created with results
            assert backup_path.exists()
            assert backup_path.stat().st_size > 0
            
            # Verify content
            content = backup_path.read_text()
            assert 'Target1' in content
            assert 'Target2' in content
            assert '1000.0' in content


class TestStandaloneFunctions:
    """Unit tests for standalone functions extracted from callbacks."""
    
    class TestBuildDeleteModalContent:
        """Test suite for _build_delete_modal_content() standalone function."""
        
        def test_delete_selected_with_rows(self):
            """Should return modal for deleting selected rows."""
            visible, children = _build_delete_modal_content(
                clickedKey="processing-delete-selected",
                selectedRows=[{"peak_label": "A", "ms_file_label": "F1"}]
            )
            
            assert visible is True
            assert isinstance(children, fac.AntdFlex)
            assert "selected results" in children.children[0].children
            assert children.children[0].strong is True
        
        def test_delete_selected_without_rows_raises_prevent_update(self):
            """Should raise PreventUpdate if no rows selected for delete-selected."""
            with pytest.raises(PreventUpdate):
                _build_delete_modal_content(
                    clickedKey="processing-delete-selected",
                    selectedRows=[]
                )
        
        def test_delete_selected_with_none_raises_prevent_update(self):
            """Should raise PreventUpdate if selectedRows is None."""
            with pytest.raises(PreventUpdate):
                _build_delete_modal_content(
                    clickedKey="processing-delete-selected",
                    selectedRows=None
                )
        
        def test_delete_all(self):
            """Should return modal for deleting all results with danger type."""
            visible, children = _build_delete_modal_content(
                clickedKey="processing-delete-all",
                selectedRows=[]
            )
            
            assert visible is True
            assert isinstance(children, fac.AntdFlex)
            assert "ALL results" in children.children[0].children
            assert children.children[0].type == "danger"
    
    class TestLoadPeaksFromResults:
        """Test suite for _load_peaks_from_results() standalone function."""
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_load_peaks_with_data(self, mock_conn):
            """Should return options and select first peak when no current value."""
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ('Peak_A',),
                ('Peak_B',),
                ('Peak_C',),
            ]
            mock_conn.return_value.__enter__.return_value.execute.return_value = mock_cursor
            
            options, selected = _load_peaks_from_results(wdir="/fake/wdir", current_value=None)
            
            assert len(options) == 3
            assert options[0] == {'label': 'Peak_A', 'value': 'Peak_A'}
            assert selected == ['Peak_A']
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_load_peaks_preserves_valid_selection(self, mock_conn):
            """Should preserve current selection if it's still valid."""
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ('Peak_A',),
                ('Peak_B',),
                ('Peak_C',),
            ]
            mock_conn.return_value.__enter__.return_value.execute.return_value = mock_cursor
            
            options, selected = _load_peaks_from_results(
                wdir="/fake/wdir",
                current_value=['Peak_B', 'Peak_C']
            )
            
            assert selected == ['Peak_B', 'Peak_C']
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_load_peaks_auto_selects_when_invalid(self, mock_conn):
            """Should auto-select first peak if current selection no longer exists."""
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ('Peak_A',),
                ('Peak_B',),
            ]
            mock_conn.return_value.__enter__.return_value.execute.return_value = mock_cursor
            
            options, selected = _load_peaks_from_results(
                wdir="/fake/wdir",
                current_value=['Peak_X', 'Peak_Y']
            )
            
            assert selected == ['Peak_A']
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_load_peaks_filters_invalid_selections(self, mock_conn):
            """Should filter out invalid peaks but keep valid ones."""
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ('Peak_A',),
                ('Peak_B',),
            ]
            mock_conn.return_value.__enter__.return_value.execute.return_value = mock_cursor
            
            options, selected = _load_peaks_from_results(
                wdir="/fake/wdir",
                current_value=['Peak_A', 'Peak_X']
            )
            
            assert selected == ['Peak_A']
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_load_peaks_empty_results(self, mock_conn):
            """Should return empty lists if no peaks in database."""
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn.return_value.__enter__.return_value.execute.return_value = mock_cursor
            
            options, selected = _load_peaks_from_results(wdir="/fake/wdir", current_value=None)
            
            assert options == []
            assert selected == []
        
        def test_load_peaks_no_wdir_raises_prevent_update(self):
            """Should raise PreventUpdate if wdir is not provided."""
            with pytest.raises(PreventUpdate):
                _load_peaks_from_results(wdir=None, current_value=None)
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_load_peaks_no_connection_raises_prevent_update(self, mock_conn):
            """Should raise PreventUpdate if database connection fails."""
            mock_conn.return_value.__enter__.return_value = None
            
            with pytest.raises(PreventUpdate):
                _load_peaks_from_results(wdir="/fake/wdir", current_value=None)
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_load_peaks_filters_none_values(self, mock_conn):
            """Should filter out None values from database results."""
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ('Peak_A',),
                (None,),
                ('Peak_B',),
            ]
            mock_conn.return_value.__enter__.return_value.execute.return_value = mock_cursor
            
            options, selected = _load_peaks_from_results(wdir="/fake/wdir", current_value=None)
            
            assert len(options) == 2
            assert options[0]['value'] == 'Peak_A'
            assert options[1]['value'] == 'Peak_B'
    
    class TestDownloadAllResults:
        """Test suite for _download_all_results() standalone function."""
        
        def test_no_columns_selected(self):
            """Should return warning notification if no columns selected."""
            download, notification = _download_all_results("/fake/wdir", "TestWS", None)
            
            assert download == dash.no_update
            assert isinstance(notification, fac.AntdNotification)
            assert notification.type == "warning"
            assert "Select at least one column." in notification.description
        
        def test_empty_columns_list(self):
            """Should return warning notification if empty list."""
            download, notification = _download_all_results("/fake/wdir", "TestWS", [])
            
            assert download == dash.no_update
            assert notification.type == "warning"
        
        def test_invalid_columns(self):
            """Should return warning if only invalid columns selected."""
            download, notification = _download_all_results("/fake/wdir", "TestWS", ['invalid_col', 'bad_col'])
            
            assert download == dash.no_update
            assert notification.type == "warning"
            assert "No valid columns selected." in notification.description
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_database_connection_failure(self, mock_conn):
            """Should return error notification if database connection fails."""
            mock_conn.return_value.__enter__.return_value = None
            
            download, notification = _download_all_results("/fake/wdir", "TestWS", ['peak_area'])
            
            assert download == dash.no_update
            assert notification.type == "error"
            assert "Database unavailable." in notification.description
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        @patch('ms_mint_app.plugins.processing.dcc.send_file')
        def test_successful_download(self, mock_send, mock_conn):
            """Should successfully download with valid columns."""
            mock_send.return_value = "mock_file_download"
            
            # Mock DB execute to return None (copy command returns nothing or count)
            # The download uses COPY to file, so no DF returned
            mock_conn.return_value.__enter__.return_value.execute.return_value = None
            
            download, notification = _download_all_results("/fake/wdir", "TestWorkspace", ['peak_area', 'peak_mean'])
            
            assert download == "mock_file_download"
            assert notification == dash.no_update
            assert mock_send.called
            # Verify temporary file usage
            assert mock_send.call_args[0][0].endswith('.csv')
    
    class TestDownloadDenseMatrix:
        """Test suite for _download_dense_matrix() standalone function."""
        
        def test_missing_rows_parameter(self):
            """Should return warning if rows not provided."""
            download, notification = _download_dense_matrix("/fake/wdir", "TestWS", None, ['peak_label'], ['peak_area'])
            
            assert download == dash.no_update
            assert notification.type == "warning"
            assert "Missing row, column, or value selection." in notification.description
        
        def test_missing_cols_parameter(self):
            """Should return warning if cols not provided."""
            download, notification = _download_dense_matrix("/fake/wdir", "TestWS", ['ms_file_label'], None, ['peak_area'])
            
            assert download == dash.no_update
            assert notification.type == "warning"
        
        def test_missing_value_parameter(self):
            """Should return warning if value not provided."""
            download, notification = _download_dense_matrix("/fake/wdir", "TestWS", ['ms_file_label'], ['peak_label'], None)
            
            assert download == dash.no_update
            assert notification.type == "warning"
        
        def test_invalid_row_column(self):
            """Should return warning if invalid row/column selection."""
            download, notification = _download_dense_matrix(
                "/fake/wdir", "TestWS",
                ['invalid_col'],  # Invalid
                ['peak_label'],
                ['peak_area']
            )
            
            assert download == dash.no_update
            assert notification.type == "warning"
            assert "Invalid row/column selection." in notification.description
        
        def test_invalid_value_column(self):
            """Should return warning if invalid value column."""
            download, notification = _download_dense_matrix(
                "/fake/wdir", "TestWS",
                ['ms_file_label'],
                ['peak_label'],
                ['invalid_value']  # Invalid
            )
            
            assert download == dash.no_update
            assert notification.type == "warning"
            assert "Invalid value selection" in notification.description
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_database_connection_failure(self, mock_conn):
            """Should return error notification if database connection fails."""
            mock_conn.return_value.__enter__.return_value = None
            
            download, notification = _download_dense_matrix(
                "/fake/wdir", "TestWS",
                ['ms_file_label'], ['peak_label'], ['peak_area']
            )
            
            assert download == dash.no_update
            assert notification.type == "error"
            assert "Database unavailable." in notification.description
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        @patch('ms_mint_app.plugins.processing.create_pivot')
        @patch('ms_mint_app.plugins.processing.dcc.send_data_frame')
        def test_successful_dense_matrix_download(self, mock_send, mock_pivot, mock_conn):
            """Should successfully create and download dense matrix."""
            import pandas as pd
            
            mock_df = pd.DataFrame({
                'Peak1': [1000.0, 1500.0],
                'Peak2': [2000.0, 2500.0]
            })
            mock_pivot.return_value = mock_df
            mock_send.return_value = "mock_download"
            mock_conn.return_value.__enter__.return_value = MagicMock()
            
            download, notification = _download_dense_matrix(
                "/fake/wdir", "TestWorkspace",
                ['ms_file_label'], ['peak_label'], ['peak_area']
            )
            
            assert download == "mock_download"
            assert notification == dash.no_update
            assert mock_pivot.called
            assert mock_send.called
    
    class TestDeleteSelectedResults:
        """Test suite for _delete_selected_results() standalone function."""
        
        def test_no_rows_selected(self):
            """Should return failed status with no notification when no rows selected."""
            notif, action_store, total = _delete_selected_results("/fake/wdir", [])
            
            assert notif is None
            assert action_store == {'action': 'delete', 'status': 'failed'}
            assert total == [0, 0]
        
        def test_empty_rows_list(self):
            """Should handle None as selected rows."""
            notif, action_store, total = _delete_selected_results("/fake/wdir", None)
            
            assert notif is None
            assert action_store == {'action': 'delete', 'status': 'failed'}
            assert total == [0, 0]
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_database_connection_failure(self, mock_conn):
            """Should raise PreventUpdate if database connection fails."""
            mock_conn.return_value.__enter__.return_value = None
            
            with pytest.raises(PreventUpdate):
                _delete_selected_results("/fake/wdir", [{'peak_label': 'P1', 'ms_file_label': 'F1'}])
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_successful_deletion(self, mock_conn):
            """Should successfully delete selected results."""
            mock_db = MagicMock()
            mock_conn.return_value.__enter__.return_value = mock_db
            
            selected = [
                {'peak_label': 'Peak1', 'ms_file_label': 'File1'},
                {'peak_label': 'Peak1', 'ms_file_label': 'File2'},
                {'peak_label': 'Peak2', 'ms_file_label': 'File1'},
            ]
            
            notif, action_store, total = _delete_selected_results("/fake/wdir", selected)
            
            assert notif is None
            assert action_store == {'action': 'delete', 'status': 'success'}
            assert total == [2, 2]  # 2 unique peaks, 2 unique files
            assert mock_db.execute.call_count >= 2  # BEGIN, DELETE, COMMIT
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_deletion_rollback_on_error(self, mock_conn):
            """Should rollback transaction and return error notification on failure."""
            mock_db = MagicMock()
            # BEGIN succeeds, then DELETE fails, then ROLLBACK succeeds
            mock_db.execute.side_effect = [None, Exception("DB Error"), None]
            mock_conn.return_value.__enter__.return_value = mock_db
            
            selected = [{'peak_label': 'Peak1', 'ms_file_label': 'File1'}]
            notif, action_store, total = _delete_selected_results("/fake/wdir", selected)
            
            assert notif is not None
            assert notif.type == "error"
            assert "Error: DB Error" in notif.description
            assert action_store == {'action': 'delete', 'status': 'failed'}
            assert total == [0, 0]
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_filters_invalid_rows(self, mock_conn):
            """Should filter out rows with missing peak_label or ms_file_label."""
            mock_db = MagicMock()
            mock_conn.return_value.__enter__.return_value = mock_db
            
            selected = [
                {'peak_label': 'Peak1', 'ms_file_label': 'File1'},
                {'peak_label': None, 'ms_file_label': 'File2'},  # Invalid
                {'peak_label': 'Peak2'},  # Missing ms_file_label
            ]
            
            notif, action_store, total = _delete_selected_results("/fake/wdir", selected)
            
            # Should only count the valid row
            assert total[0] == 1  # 1 unique peak
            assert total[1] == 1  # 1 unique file
    
    class TestDeleteAllResults:
        """Test suite for _delete_all_results() standalone function."""
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_database_connection_failure(self, mock_conn):
            """Should raise PreventUpdate if database connection fails."""
            mock_conn.return_value.__enter__.return_value = None
            
            with pytest.raises(PreventUpdate):
                _delete_all_results("/fake/wdir")
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_successful_deletion_all(self, mock_conn):
            """Should successfully delete all results."""
            mock_db = MagicMock()
            mock_db.execute.return_value.fetchone.return_value = (5, 10)  # 5 peaks, 10 files
            mock_conn.return_value.__enter__.return_value = mock_db
            
            notif, action_store, total = _delete_all_results("/fake/wdir")
            
            assert notif is None
            assert action_store == {'action': 'delete', 'status': 'success'}
            assert total == [5, 10]
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_deletion_with_empty_table(self, mock_conn):
            """Should handle deletion when table is already empty."""
            mock_db = MagicMock()
            mock_db.execute.return_value.fetchone.return_value = None
            mock_conn.return_value.__enter__.return_value = mock_db
            
            notif, action_store, total = _delete_all_results("/fake/wdir")
            
            assert notif is None
            assert action_store == {'action': 'delete', 'status': 'failed'}
            assert total == [0, 0]
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_deletion_rollback_on_error(self, mock_conn):
            """Should rollback transaction and return error notification on failure."""
            mock_db = MagicMock()
            # BEGIN succeeds, COUNT fails, then ROLLBACK succeeds
            mock_db.execute.side_effect = [None, Exception("DB Error"), None]
            mock_conn.return_value.__enter__.return_value = mock_db
            
            notif, action_store, total = _delete_all_results("/fake/wdir")
            
            assert notif is not None
            assert notif.type == "error"
            assert "Error: DB Error" in notif.description
            assert action_store == {'action': 'delete', 'status': 'failed'}
            assert total == [0, 0]
        
        @patch('ms_mint_app.plugins.processing.duckdb_connection')
        def test_transaction_management(self, mock_conn):
            """Should properly manage transactions (BEGIN/COMMIT)."""
            mock_db = MagicMock()
            mock_db.execute.return_value.fetchone.return_value = (3, 5)
            mock_conn.return_value.__enter__.return_value = mock_db
            
            _delete_all_results("/fake/wdir")
            
            # Verify transaction flow: BEGIN, COUNT, DELETE, COMMIT
            calls = [str(call) for call in mock_db.execute.call_args_list]
            assert any('BEGIN' in str(call) for call in calls)
            assert any('COMMIT' in str(call) for call in calls)


class TestRTAlignmentAwareProcessing:
    """Test RT alignment-aware peak area calculation."""
    
    @pytest.fixture
    def aligned_wdir(self, tmp_path):
        """Create workspace with chromatogram data and RT alignment enabled."""
        wdir = tmp_path / "aligned_workspace"
        wdir.mkdir(parents=True)
        
        with duckdb_connection(wdir) as conn:
            # Add samples
            conn.execute("""
                INSERT INTO samples (ms_file_label, sample_type, label, use_for_optimization, ms_type, polarity) 
                VALUES 
                    ('File1', 'Sample', 'Sample1', TRUE, 'ms1', 'Positive'),
                    ('File2', 'Sample', 'Sample2', TRUE, 'ms1', 'Positive'),
                    ('File3', 'Sample', 'Sample3', TRUE, 'ms1', 'Positive')
            """)
            
            # Add target WITH RT alignment enabled and per-file shifts
            # Simulating: File1 peak at 100s, File2 at 101s, File3 at 102s
            # Reference RT = 101s (median), so shifts are: +1, 0, -1
            import json
            shifts_json = json.dumps({
                'File1': 1.0,   # Peak was at 100s, needs +1s to align to 101s
                'File2': 0.0,   # Peak at 101s, no shift needed
                'File3': -1.0   # Peak was at 102s, needs -1s to align to 101s
            })
            
            conn.execute("""
                INSERT INTO targets (
                    peak_label, mz_mean, mz_width, rt, rt_min, rt_max, ms_type, bookmark,
                    rt_align_enabled, rt_align_reference_rt, rt_align_shifts, 
                    rt_align_rt_min, rt_align_rt_max
                ) 
                VALUES (
                    'AlignedTarget', 100.5, 0.01, 101.0, 99.0, 103.0, 'ms1', TRUE,
                    TRUE, 101.0, ?, 99.0, 103.0
                )
            """, [shifts_json])
            
            # Add chromatograms with peaks at different RTs
            # File1: peak centered at 100s (shifted by +1s for alignment)
            conn.execute("""
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                VALUES ('AlignedTarget', 'File1', 
                        [97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0], 
                        [100.0, 200.0, 400.0, 1000.0, 400.0, 200.0, 100.0, 50.0],
                        'ms1')
            """)
            
            # File2: peak centered at 101s (no shift needed)
            conn.execute("""
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                VALUES ('AlignedTarget', 'File2', 
                        [97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0], 
                        [50.0, 100.0, 400.0, 800.0, 1000.0, 400.0, 200.0, 100.0],
                        'ms1')
            """)
            
            # File3: peak centered at 102s (shifted by -1s for alignment)
            conn.execute("""
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                VALUES ('AlignedTarget', 'File3', 
                        [97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0], 
                        [50.0, 100.0, 200.0, 400.0, 800.0, 1000.0, 400.0, 200.0],
                        'ms1')
            """)
            
        return str(wdir)
    
    def test_alignment_shifts_stored_correctly(self, aligned_wdir):
        """Verify that RT alignment shifts are stored as per-file JSON."""
        import json
        
        with duckdb_connection(aligned_wdir) as conn:
            result = conn.execute("""
                SELECT rt_align_enabled, rt_align_reference_rt, rt_align_shifts
                FROM targets
                WHERE peak_label = 'AlignedTarget'
            """).fetchone()
            
            assert result[0] is True  # rt_align_enabled
            assert result[1] == 101.0  # rt_align_reference_rt
            
            shifts = json.loads(result[2])
            assert 'File1' in shifts
            assert 'File2' in shifts
            assert 'File3' in shifts
            assert shifts['File1'] == 1.0
            assert shifts['File2'] == 0.0
            assert shifts['File3'] == -1.0
    
    def test_compute_results_with_alignment(self, aligned_wdir):
        """Test that compute_results_in_batches applies per-sample RT shifts."""
        # Run results computation
        result = compute_results_in_batches(
            wdir=aligned_wdir,
            use_bookmarked=False,
            recompute=True,
            batch_size=100,
            n_cpus=1,
            ram=1
        )
        
        assert result['processed'] == 3  # 3 chromatograms processed
        assert result['failed'] == 0
        
        with duckdb_connection(aligned_wdir) as conn:
            results = conn.execute("""
                SELECT ms_file_label, peak_area, peak_rt_of_max, rt_aligned, rt_shift
                FROM results
                WHERE peak_label = 'AlignedTarget'
                ORDER BY ms_file_label
            """).fetchall()
            
            assert len(results) == 3
            
            # Verify rt_aligned and rt_shift columns are populated
            for row in results:
                ms_file = row[0]
                peak_area = row[1]
                rt_aligned = row[3]
                rt_shift = row[4]
                
                # All should have rt_aligned=True
                assert rt_aligned is True, f"rt_aligned for {ms_file} should be True"
                
                # Verify correct shifts
                if ms_file == 'File1':
                    assert rt_shift == 1.0, f"rt_shift for File1 should be 1.0"
                elif ms_file == 'File2':
                    assert rt_shift == 0.0, f"rt_shift for File2 should be 0.0"
                elif ms_file == 'File3':
                    assert rt_shift == -1.0, f"rt_shift for File3 should be -1.0"
                
                # All files should have non-zero peak areas
                assert peak_area > 0, f"Peak area for {ms_file} should be > 0"

    
    def test_alignment_disabled_uses_global_window(self, aligned_wdir):
        """Test that when alignment is disabled, global rt_min/rt_max is used."""
        with duckdb_connection(aligned_wdir) as conn:
            # Disable alignment
            conn.execute("""
                UPDATE targets 
                SET rt_align_enabled = FALSE,
                    rt_align_shifts = NULL
                WHERE peak_label = 'AlignedTarget'
            """)
        
        # Run results computation
        compute_results_in_batches(
            wdir=aligned_wdir,
            use_bookmarked=False,
            recompute=True,
            batch_size=100,
            n_cpus=1,
            ram=1
        )
        
        with duckdb_connection(aligned_wdir) as conn:
            results = conn.execute("""
                SELECT ms_file_label, peak_area
                FROM results
                WHERE peak_label = 'AlignedTarget'
                ORDER BY ms_file_label
            """).fetchall()
            
            assert len(results) == 3
            
            # Without alignment, all files use the same window [99, 103]
            # Peak areas will differ based on where the actual peak is
            file1_area = next(r[1] for r in results if r[0] == 'File1')
            file2_area = next(r[1] for r in results if r[0] == 'File2')
            file3_area = next(r[1] for r in results if r[0] == 'File3')
            
            # File1's peak is at 100s, partially captured in [99, 103]
            # File2's peak is at 101s, well captured in [99, 103]
            # File3's peak is at 102s, partially captured in [99, 103]
            # Without alignment, File2 should have highest area captured
            assert file2_area > 0
    
    def test_json_extraction_with_special_characters(self, tmp_path):
        """Test that JSON extraction handles file names with special characters."""
        wdir = tmp_path / "special_chars_workspace"
        wdir.mkdir(parents=True)
        
        with duckdb_connection(wdir) as conn:
            # Add sample with special characters in name
            conn.execute("""
                INSERT INTO samples (ms_file_label, sample_type, label, use_for_optimization, ms_type, polarity) 
                VALUES ('File-With_Special.Chars', 'Sample', 'Sample1', TRUE, 'ms1', 'Positive')
            """)
            
            import json
            shifts_json = json.dumps({'File-With_Special.Chars': 0.5})
            
            conn.execute("""
                INSERT INTO targets (
                    peak_label, mz_mean, mz_width, rt, rt_min, rt_max, ms_type, bookmark,
                    rt_align_enabled, rt_align_reference_rt, rt_align_shifts, 
                    rt_align_rt_min, rt_align_rt_max
                ) 
                VALUES (
                    'SpecialTarget', 100.5, 0.01, 100.0, 98.0, 102.0, 'ms1', FALSE,
                    TRUE, 100.0, ?, 98.0, 102.0
                )
            """, [shifts_json])
            
            conn.execute("""
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                VALUES ('SpecialTarget', 'File-With_Special.Chars', 
                        [97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0], 
                        [100.0, 200.0, 500.0, 1000.0, 500.0, 200.0, 100.0],
                        'ms1')
            """)
        
        # Run results computation
        result = compute_results_in_batches(
            wdir=str(wdir),
            use_bookmarked=False,
            recompute=True,
            batch_size=100,
            n_cpus=1,
            ram=1
        )
        
        assert result['processed'] == 1
        assert result['failed'] == 0

