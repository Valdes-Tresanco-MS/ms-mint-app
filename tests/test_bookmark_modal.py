
import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock modules before import
sys.modules['feffery_antd_components'] = MagicMock()
sys.modules['duckdb'] = MagicMock()
mock_duckdb_manager = MagicMock()
sys.modules['ms_mint_app.duckdb_manager'] = mock_duckdb_manager
sys.modules['ms_mint_app.tools'] = MagicMock()

# Import the callback (will use mocks)
from ms_mint_app.plugins.target_optimization import _toggle_bookmark_logic

def test_toggle_bookmark_logic_false_to_true():
    # Setup
    mock_conn = MagicMock()
    mock_duckdb_manager.duckdb_connection.return_value.__enter__.return_value = mock_conn
    
    # Mock current state: Unbookmarked (False)
    # First execute is SELECT, second is UPDATE
    mock_conn.execute.return_value.fetchone.return_value = [False]
    
    # Call
    _toggle_bookmark_logic("TestTarget", "wd")
    
    # Verify DB update: UPDATE targets SET bookmark = ? WHERE peak_label = ?
    # We expect [True, "TestTarget"]
    # Check the calls to execute. 
    # execute call args list:
    # 1. SELECT ...
    # 2. UPDATE ...
    calls = mock_conn.execute.call_args_list
    assert len(calls) >= 2
    update_call = calls[-1]
    args, _ = update_call
    assert "UPDATE targets SET bookmark = ?" in args[0]
    assert args[1] == [True, "TestTarget"]

    # Verify AntdIcon was created with gold color
    sys.modules['feffery_antd_components'].AntdIcon.assert_called_with(icon="antd-star", style={"color": "gold"})

def test_toggle_bookmark_logic_true_to_false():
    # Setup
    mock_conn = MagicMock()
    mock_duckdb_manager.duckdb_connection.return_value.__enter__.return_value = mock_conn
    
    # Mock current state: Bookmarked (True)
    mock_conn.execute.return_value.fetchone.return_value = [True]
    
    # Call
    _toggle_bookmark_logic("TestTarget", "wd")
    
    # Verify DB update
    calls = mock_conn.execute.call_args_list
    update_call = calls[-1]
    args, _ = update_call
    assert args[1] == [False, "TestTarget"]

    # Verify AntdIcon was created with gray color
    sys.modules['feffery_antd_components'].AntdIcon.assert_called_with(icon="antd-star", style={"color": "gray"})
