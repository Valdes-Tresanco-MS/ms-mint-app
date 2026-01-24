import pytest
import os
import platform
from pathlib import Path
from unittest.mock import MagicMock, patch
import dash
from dash.exceptions import PreventUpdate

from ms_mint_app.plugins.explorer import (
    FileExplorer,
    _navigate_folders,
    _update_selection,
    _background_processing
)

@pytest.fixture
def explorer():
    return FileExplorer()

@pytest.fixture
def temp_fs(tmp_path):
    # Create a dummy file system structure
    (tmp_path / "folder1").mkdir()
    (tmp_path / "folder2").mkdir()
    (tmp_path / "folder1" / "file1.mzML").write_text("content")
    (tmp_path / "folder1" / "file2.mzXML").write_text("content")
    (tmp_path / "folder2" / "file3.csv").write_text("content")
    (tmp_path / "folder2" / "subfolder").mkdir()
    (tmp_path / "folder2" / "subfolder" / "file4.csv").write_text("content")
    return tmp_path

class TestExplorerDataRetrieval:
    def test_get_table_data_basic(self, explorer, temp_fs):
        extensions = [".mzML", ".mzXML"]
        data = explorer.get_table_data(temp_fs / "folder1", extensions)
        
        # Should contain 2 files
        files = [d for d in data if not d['is_dir']]
        assert len(files) == 2
        names = [f['name']['content'] for f in files]
        assert "file1.mzML" in names
        assert "file2.mzXML" in names

    def test_get_table_data_with_folders(self, explorer, temp_fs):
        extensions = [".csv"]
        data = explorer.get_table_data(temp_fs, extensions)
        
        # Should contain 2 folders (folder1, folder2)
        folders = [d for d in data if d['is_dir']]
        assert len(folders) == 2
        
        # folder2 should have 1 csv file (file3.csv is direct child)
        folder2_data = next(d for d in data if d['name']['content'] == "folder2")
        assert folder2_data['file_count'] == 1 

    def test_get_table_data_permission_denied(self, explorer, temp_fs):
        with patch.object(Path, 'iterdir', side_effect=PermissionError):
            data = explorer.get_table_data(temp_fs, [".csv"])
            assert data == []

class TestExplorerSelectionLogic:
    def test_get_selected_tree_data(self, explorer):
        selected_files = ["/path/to/f1.mzML", "/path/to/f2.mzML", "/other/dir/f3.csv"]
        tree_data = explorer.get_selected_tree_data(selected_files)
        
        assert len(tree_data) == 2 # 2 folders: /path/to and /other/dir
        
        # Check first folder
        node1 = next(t for t in tree_data if "path/to" in t['key'])
        assert len(node1['children']) == 2
        assert node1['title'] == "to (2 files)"

    def test_update_selection_add_file(self, explorer):
        current_selection = []
        processing_type = {'extensions': ['.mzML']}
        recentlyCellClickRecord = {'key': '/path/to/file.mzML', 'is_dir': False}
        
        with patch('dash.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'file-table.nClicksCell'}]
            res = _update_selection(explorer, None, 1, 0, 0, [], recentlyCellClickRecord, current_selection, processing_type, [], [])
            
            selected_list = res[0]
            assert "/path/to/file.mzML" in selected_list

    def test_update_selection_add_folder(self, explorer, temp_fs):
        current_selection = []
        processing_type = {'extensions': ['.csv']}
        folder_path = str(temp_fs / "folder2")
        table_data = [{'key': folder_path, 'path': folder_path, 'is_dir': True}]
        selectedRowKeys = [folder_path]
        
        with patch('dash.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'file-table.selectedRowKeys'}]
            res = _update_selection(explorer, selectedRowKeys, 0, 0, 0, [], None, current_selection, processing_type, table_data, [])
            
            selected_list = res[0]
            # folder2 has file3.csv and subfolder/file4.csv. rglob should find both.
            assert len(selected_list) == 2
            assert any("file3.csv" in s for s in selected_list)
            assert any("file4.csv" in s for s in selected_list)

    def test_update_selection_clear(self, explorer):
        current_selection = ["/path/f1.csv"]
        with patch('dash.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'clear-selection-btn.nClicks'}]
            res = _update_selection(explorer, None, 0, 1, 0, [], None, current_selection, {}, [], [])
            assert res[0] == []

class TestExplorerNavigation:
    def test_navigate_folders_table_click(self, explorer, temp_fs):
        modal_visible = True
        recentlyCellClickRecord = {'key': str(temp_fs / "folder1"), 'is_dir': True}
        processing_type = {'extensions': ['.mzML']}
        
        with patch('dash.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'file-table.recentlyCellClickRecord'}]
            data = _navigate_folders(explorer, modal_visible, None, recentlyCellClickRecord, str(temp_fs), processing_type)
        
        breadcrumb = data[0]
        new_path = data[2]
        assert new_path == str(temp_fs / "folder1")
        assert breadcrumb[-1]['title'] == "folder1"

class TestExplorerProcessing:
    @patch('ms_mint_app.plugins.explorer.process_ms_files')
    @patch('ms_mint_app.plugins.explorer.activate_workspace_logging')
    def test_background_processing_ms_files(self, mock_logging, mock_process, explorer):
        mock_process.return_value = (5, [], 0) # total_processed, failed_files, duplicates_count
        
        processing_type = {'type': 'ms-files'}
        selected_files = ['/path/f1.mzML', '/path/f2.mzML']
        
        res = _background_processing(None, 1, processing_type, selected_files, 4, '/tmp/wdir')
        
        notification = res[0]
        assert notification.type == 'success'
        assert "Processed 5 items." in notification.description
        mock_process.assert_called_once()

    @patch('ms_mint_app.plugins.explorer.process_metadata')
    def test_background_processing_metadata(self, mock_process, explorer):
        mock_process.return_value = (2, [])
        processing_type = {'type': 'metadata'}
        selected_files = ['/path/meta.csv']
        
        res = _background_processing(None, 1, processing_type, selected_files, 4, None)
        assert res[0].type == 'success'
        assert "Processed 2 items." in res[0].description

    @patch('ms_mint_app.plugins.explorer.process_targets')
    def test_background_processing_targets_fail(self, mock_process, explorer):
        mock_process.return_value = (0, {'bad.csv': 'Error details'}, ['Row 1 failed'], {})
        processing_type = {'type': 'targets'}
        selected_files = ['/path/bad.csv']
        
        res = _background_processing(None, 1, processing_type, selected_files, 4, None)
        assert res[0].type == 'error'
        assert "Failed to process 1 file(s)." in res[0].description

    def test_background_processing_no_selection(self, explorer):
        with pytest.raises(PreventUpdate):
            _background_processing(None, 1, {'type': 'ms-files'}, [], 4, None)
