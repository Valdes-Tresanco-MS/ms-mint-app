import pytest
import math
import duckdb
import polars as pl
from pathlib import Path
from unittest.mock import patch, MagicMock
from dash.exceptions import PreventUpdate

from ms_mint_app.plugins.targets import (
    _targets_table, _target_delete, _save_target_table_on_edit,
    _save_switch_changes, _run_asari_analysis
)
from ms_mint_app.duckdb_manager import duckdb_connection

@pytest.fixture
def temp_wdir(tmp_path):
    wdir = tmp_path / "workspace"
    wdir.mkdir(parents=True)
    # Register workspace and create tables via duckdb_connection
    with duckdb_connection(wdir) as conn:
        conn.execute("INSERT INTO targets (peak_label, mz_mean, rt, ms_type, category) VALUES ('Target1', 100.0, 60.0, 'ms1', 'Cat1')")
        conn.execute("INSERT INTO targets (peak_label, mz_mean, rt, ms_type, category) VALUES ('Target2', 200.0, 120.0, 'ms1', 'Cat2')")
        conn.execute("INSERT INTO samples (ms_file_label, polarity, use_for_processing) VALUES ('File1', 'Positive', TRUE)")
    return str(wdir)

class TestTargetsTable:
    def test_targets_table_basic(self, temp_wdir):
        pagination = {'pageSize': 15, 'current': 1}
        filter_ = {}
        sorter = []
        filterOptions = {'category': {'filterCustomItems': []}}
        
        with patch('dash.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'targets-table.pagination'}]
            result = _targets_table({'page': 'Targets'}, pagination, filter_, sorter, filterOptions, temp_wdir)
        
        data, selected_keys, pagination_out, filter_out = result
        assert len(data) == 2
        # Check if Target1 or Target2 is present (order might vary depending on default sort if not specified)
        labels = [row['peak_label'] for row in data]
        assert 'Target1' in labels
        assert 'Target2' in labels
        assert pagination_out['total'] == 2

    def test_targets_table_filtering(self, temp_wdir):
        pagination = {'pageSize': 15, 'current': 1}
        filter_ = {'peak_label': ['Target1']}
        sorter = []
        filterOptions = {'peak_label': {'filterMode': 'keyword'}}
        
        with patch('dash.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'targets-table.filter'}]
            result = _targets_table({'page': 'Targets'}, pagination, filter_, sorter, filterOptions, temp_wdir)
        
        data, _, pagination_out, _ = result
        assert len(data) == 1
        assert data[0]['peak_label'] == 'Target1'

    def test_targets_table_sorting(self, temp_wdir):
        pagination = {'pageSize': 15, 'current': 1}
        filter_ = {}
        sorter = {'columns': ['mz_mean'], 'orders': ['descend']}
        filterOptions = {}
        
        with patch('dash.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'targets-table.sorter'}]
            result = _targets_table({'page': 'Targets'}, pagination, filter_, sorter, filterOptions, temp_wdir)
        
        data, _, _, _ = result
        assert data[0]['peak_label'] == 'Target2' # 200.0 > 100.0

    def test_targets_table_prevent_update_wrong_page(self, temp_wdir):
        with pytest.raises(PreventUpdate):
            _targets_table({'page': 'Other'}, {}, {}, {}, {}, temp_wdir)

class TestTargetsCRUD:
    def test_target_delete_selected(self, temp_wdir):
        selected_rows = [{'peak_label': 'Target1'}]
        result = _target_delete(1, selected_rows, 'delete-selected', temp_wdir)
        
        notification, action_store = result
        assert action_store['status'] == 'success'
        
        with duckdb_connection(temp_wdir) as conn:
            count = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
            assert count == 1
            label = conn.execute("SELECT peak_label FROM targets").fetchone()[0]
            assert label == 'Target2'

    def test_target_delete_all(self, temp_wdir):
        result = _target_delete(1, [], 'delete-all', temp_wdir)
        
        notification, action_store = result
        assert action_store['status'] == 'success'
        
        with duckdb_connection(temp_wdir) as conn:
            count = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
            assert count == 0

    def test_save_target_table_on_edit(self, temp_wdir):
        row_edited = {'peak_label': 'Target1', 'rt': 65.5}
        result = _save_target_table_on_edit(row_edited, 'rt', temp_wdir)
        
        notification, action_store = result
        assert action_store['status'] == 'success'
        
        with duckdb_connection(temp_wdir) as conn:
            rt = conn.execute("SELECT rt FROM targets WHERE peak_label = 'Target1'").fetchone()[0]
            assert rt == 65.5

    def test_save_switch_changes(self, temp_wdir):
        row = {'peak_label': 'Target1'}
        _save_switch_changes('peak_selection', True, row, temp_wdir)
        
        with duckdb_connection(temp_wdir) as conn:
            selected = conn.execute("SELECT peak_selection FROM targets WHERE peak_label = 'Target1'").fetchone()[0]
            assert selected is True

class TestTargetsAsari:
    @patch('ms_mint_app.plugins.targets_asari.run_asari_workflow')
    def test_run_asari_analysis_success(self, mock_run, temp_wdir):
        mock_run.return_value = {'success': True, 'message': 'Asari finished'}
        
        result = _run_asari_analysis(1, temp_wdir, 1, 5, 'pos', 20, 100000, 6, 0.9, 1, 90)
        
        notification, visible, alert, action_store, workspace_status = result
        assert visible is False
        assert 'timestamp' in action_store
        assert mock_run.called

    def test_run_asari_analysis_validation_failure(self, temp_wdir):
        # Invalid CPU cores
        result = _run_asari_analysis(1, temp_wdir, 0, 5, 'pos', 20, 100000, 6, 0.9, 1, 90)
        # result[2] is the AntdAlert
        assert result[2].message == "Invalid CPU cores selected."
        
        # Invalid SNR
        result = _run_asari_analysis(1, temp_wdir, 1, 5, 'pos', 0, 100000, 6, 0.9, 1, 90)
        assert result[2].message == "Invalid Signal/Noise Ratio."
