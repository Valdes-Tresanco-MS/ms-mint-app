import pytest
import os
import pandas as pd
import polars as pl
from pathlib import Path
from unittest.mock import MagicMock, patch
import dash
from dash.exceptions import PreventUpdate

from ms_mint_app.plugins.ms_files import (
    MsFilesPlugin,
    _ms_files_table,
    _confirm_and_delete,
    _save_table_on_edit,
    _save_switch_changes,
    _save_color_to_db,
    _genere_color_map,
    generate_colors
)

@pytest.fixture
def temp_wdir(tmp_path):
    # mint_root is tmp_path
    # workspaces_dir is tmp_path / "workspaces"
    # wdir is tmp_path / "workspaces" / "ws1"
    workspaces_dir = tmp_path / "workspaces"
    workspaces_dir.mkdir()
    wdir = workspaces_dir / "ws1"
    wdir.mkdir()
    
    from ms_mint_app.duckdb_manager import duckdb_connection
    with duckdb_connection(str(wdir), register_activity=False) as conn:
        conn.execute("""
            INSERT INTO samples (ms_file_label, label, color, use_for_optimization, use_for_processing, use_for_analysis, sample_type, group_1, group_2, polarity, ms_type, file_type)
            VALUES 
            ('file1', 'File 1', '#ff0000', True, True, True, 'Sample', 'G1', 'G2', 'Positive', 'ms1', 'mzML'),
            ('file2', 'File 2', '#00ff00', False, True, False, 'QC', 'G1', 'G2', 'Negative', 'ms1', 'mzML')
        """)
    return str(wdir)

class TestMSFilesTable:
    @patch('ms_mint_app.plugins.ms_files.dash.callback_context')
    def test_ms_files_table_basic(self, mock_ctx, temp_wdir):
        mock_ctx.triggered = []
        pagination = {'pageSize': 10, 'current': 1}
        filterOptions = {
            'sample_type': {}, 'polarity': {}, 'ms_type': {}, 'file_type': {},
            'ms_file_label': {}, 'label': {}, 'color': {}
        }
        
        res = _ms_files_table({'page': 'MS-Files'}, None, None, pagination, None, None, filterOptions, None, temp_wdir)
        
        data = res[0]
        assert len(data) == 2
        assert data[0]['ms_file_label'] == 'file1'
        assert data[0]['color']['content'] == '#ff0000'
        assert data[1]['ms_file_label'] == 'file2'

    @patch('ms_mint_app.plugins.ms_files.dash.callback_context')
    def test_ms_files_table_filtering(self, mock_ctx, temp_wdir):
        mock_ctx.triggered = []
        pagination = {'pageSize': 10, 'current': 1}
        filterOptions = {
            'sample_type': {}, 'polarity': {'filterMode': 'checkbox'}, 'ms_type': {}, 'file_type': {},
            'ms_file_label': {}, 'label': {}, 'color': {}
        }
        # Filter for Negative polarity
        filter_ = {'polarity': ['Negative']}
        res = _ms_files_table({'page': 'MS-Files'}, None, None, pagination, filter_, None, filterOptions, None, temp_wdir)
    
        data = res[0]
        assert len(data) == 1
        assert data[0]['ms_file_label'] == 'file2'

class TestMSFilesCRUD:
    def test_delete_selected(self, temp_wdir):
        selectedRows = [{'ms_file_label': 'file1'}]
        with patch('ms_mint_app.plugins.ms_files.activate_workspace_logging'):
            res = _confirm_and_delete(1, selectedRows, 'delete-selected', temp_wdir)
            
        notification = res[0]
        assert notification.type == 'success'
        assert "Deleted 1 files" in notification.description
        
        from ms_mint_app.duckdb_manager import duckdb_connection
        with duckdb_connection(temp_wdir) as conn:
            count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            assert count == 1
            # Ensure file1 is gone
            labels = conn.execute("SELECT ms_file_label FROM samples").df()['ms_file_label'].to_list()
            assert 'file1' not in labels

    def test_delete_all(self, temp_wdir):
        with patch('ms_mint_app.plugins.ms_files.activate_workspace_logging'):
            res = _confirm_and_delete(1, [], 'delete-all', temp_wdir)
            
        assert "Deleted 2 files" in res[0].description
        from ms_mint_app.duckdb_manager import duckdb_connection
        with duckdb_connection(temp_wdir) as conn:
            count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            assert count == 0

    def test_save_table_on_edit(self, temp_wdir):
        row_edited = {'ms_file_label': 'file1', 'label': 'New Label'}
        column_edited = 'label'
        
        res = _save_table_on_edit(row_edited, column_edited, temp_wdir)
        assert res[0].type == 'success'
        
        from ms_mint_app.duckdb_manager import duckdb_connection
        with duckdb_connection(temp_wdir) as conn:
            label = conn.execute("SELECT label FROM samples WHERE ms_file_label = 'file1'").fetchone()[0]
            assert label == 'New Label'

    def test_save_switch_changes(self, temp_wdir):
        recentlySwitchRow = {'ms_file_label': 'file1'}
        _save_switch_changes('use_for_optimization', False, recentlySwitchRow, temp_wdir)
        
        from ms_mint_app.duckdb_manager import duckdb_connection
        with duckdb_connection(temp_wdir) as conn:
            val = conn.execute("SELECT use_for_optimization FROM samples WHERE ms_file_label = 'file1'").fetchone()[0]
            assert val is False

class TestMSFilesColors:
    def test_set_color_manual(self, temp_wdir):
        recentlyButtonClickedRow = {'ms_file_label': 'file1', 'color': {'content': '#ff0000'}}
        # _save_color_to_db(okCounts, color, recentlyButtonClickedRow, wdir)
        res = _save_color_to_db(1, '#aabbcc', recentlyButtonClickedRow, temp_wdir)
        
        assert res.type == 'success'
        from ms_mint_app.duckdb_manager import duckdb_connection
        with duckdb_connection(temp_wdir) as conn:
            color = conn.execute("SELECT color FROM samples WHERE ms_file_label = 'file1'").fetchone()[0]
            assert color == '#aabbcc'

    def test_genere_color_map(self, temp_wdir):
        with patch('dash.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'ms-options.nClicks'}]
            res = _genere_color_map(1, 'generate-colors', temp_wdir)
            
        assert res[0].type == 'success'
        # Since we only have 2 files, generate_colors should have assigned colors.
        from ms_mint_app.duckdb_manager import duckdb_connection
        with duckdb_connection(temp_wdir) as conn:
            colors = conn.execute("SELECT color FROM samples").df()['color'].to_list()
            assert all(c.startswith('#') for c in colors)
