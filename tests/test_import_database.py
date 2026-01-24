"""
Unit tests for the Import Database as Workspace feature.
Tests validate_mint_database() and import_database_as_workspace() functions.
"""

import os
import pytest
import duckdb
from pathlib import Path
from unittest.mock import patch

from ms_mint_app.duckdb_manager import (
    validate_mint_database,
    import_database_as_workspace,
    compact_database,
    _create_tables,
    _create_workspace_tables,
    REQUIRED_TABLES,
)


@pytest.fixture
def temp_mint_root(tmp_path):
    """Create a temporary MINT root directory with mint.db."""
    mint_root = tmp_path / "MINT" / "Local"
    mint_root.mkdir(parents=True)
    
    # Create mint.db with workspace tables
    con = duckdb.connect(str(mint_root / "mint.db"))
    _create_workspace_tables(con)
    con.close()
    
    return mint_root


@pytest.fixture
def valid_workspace_db(tmp_path):
    """Create a valid MINT workspace database for testing."""
    db_path = tmp_path / "valid_workspace.db"
    con = duckdb.connect(str(db_path))
    _create_tables(con)
    
    # Insert some test data
    con.execute("INSERT INTO samples (ms_file_label, label) VALUES ('test_file', 'Test')")
    con.execute("INSERT INTO targets (peak_label, mz_mean, rt, rt_min, rt_max) VALUES ('target1', 100.0, 5.0, 4.0, 6.0)")
    
    con.close()
    return db_path


@pytest.fixture
def invalid_db_missing_tables(tmp_path):
    """Create an invalid database missing required tables."""
    db_path = tmp_path / "invalid_missing.db"
    con = duckdb.connect(str(db_path))
    # Only create samples table
    con.execute("""
        CREATE TABLE samples (
            ms_file_label VARCHAR PRIMARY KEY,
            label VARCHAR
        )
    """)
    con.close()
    return db_path


@pytest.fixture
def invalid_db_missing_columns(tmp_path):
    """Create a database with tables but missing required columns."""
    db_path = tmp_path / "invalid_columns.db"
    con = duckdb.connect(str(db_path))
    
    # Create tables with incomplete schemas
    con.execute("CREATE TABLE samples (ms_file_label VARCHAR)")
    con.execute("CREATE TABLE targets (peak_label VARCHAR)")  # Missing mz_mean, rt, rt_min, rt_max
    con.execute("CREATE TABLE ms1_data (ms_file_label VARCHAR, scan_id INTEGER, mz DOUBLE, intensity DOUBLE, scan_time DOUBLE)")
    con.execute("CREATE TABLE chromatograms (peak_label VARCHAR, ms_file_label VARCHAR)")
    con.execute("CREATE TABLE results (peak_label VARCHAR, ms_file_label VARCHAR, peak_area DOUBLE)")
    
    con.close()
    return db_path


class TestValidateMintDatabase:
    """Tests for validate_mint_database()."""

    def test_valid_database(self, valid_workspace_db):
        """Test validation of a valid MINT database."""
        is_valid, error_msg, stats = validate_mint_database(str(valid_workspace_db))
        
        assert is_valid is True
        assert error_msg == ""
        assert 'samples' in stats
        assert stats['samples'] == 1
        assert 'targets' in stats
        assert stats['targets'] == 1

    def test_file_not_found(self, tmp_path):
        """Test validation with non-existent file."""
        is_valid, error_msg, stats = validate_mint_database(str(tmp_path / "nonexistent.db"))
        
        assert is_valid is False
        assert "not found" in error_msg.lower()

    def test_invalid_extension(self, tmp_path):
        """Test validation with invalid file extension."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a database")
        
        is_valid, error_msg, stats = validate_mint_database(str(txt_file))
        
        assert is_valid is False
        assert "extension" in error_msg.lower()

    def test_missing_required_tables(self, invalid_db_missing_tables):
        """Test validation with missing tables."""
        is_valid, error_msg, stats = validate_mint_database(str(invalid_db_missing_tables))
        
        assert is_valid is False
        assert "missing required tables" in error_msg.lower()

    def test_missing_required_columns(self, invalid_db_missing_columns):
        """Test validation with missing columns."""
        is_valid, error_msg, stats = validate_mint_database(str(invalid_db_missing_columns))
        
        assert is_valid is False
        assert "missing columns" in error_msg.lower()

    def test_directory_instead_of_file(self, tmp_path):
        """Test validation with directory path."""
        is_valid, error_msg, stats = validate_mint_database(str(tmp_path))
        
        assert is_valid is False
        assert "not a file" in error_msg.lower()


class TestImportDatabaseAsWorkspace:
    """Tests for import_database_as_workspace()."""

    def test_successful_import(self, temp_mint_root, valid_workspace_db):
        """Test successful database import."""
        success, error_msg, workspace_key = import_database_as_workspace(
            str(valid_workspace_db),
            "imported_ws",
            temp_mint_root
        )
        
        assert success is True
        assert error_msg == ""
        assert workspace_key != ""
        
        # Verify workspace folder was created
        ws_folder = temp_mint_root / "workspaces" / workspace_key
        assert ws_folder.exists()
        
        # Verify database was copied
        db_file = ws_folder / "workspace_mint.db"
        assert db_file.exists()
        
        # Verify workspace record was created
        con = duckdb.connect(str(temp_mint_root / "mint.db"))
        row = con.execute("SELECT name, active FROM workspaces WHERE key = ?", (workspace_key,)).fetchone()
        con.close()
        
        assert row is not None
        assert row[0] == "imported_ws"
        assert row[1] is True

    def test_duplicate_workspace_name(self, temp_mint_root, valid_workspace_db):
        """Test import with duplicate workspace name."""
        # First import should succeed
        success1, _, _ = import_database_as_workspace(
            str(valid_workspace_db),
            "test_ws",
            temp_mint_root
        )
        assert success1 is True
        
        # Second import with same name should fail
        success2, error_msg, _ = import_database_as_workspace(
            str(valid_workspace_db),
            "test_ws",
            temp_mint_root
        )
        
        assert success2 is False
        assert "already exists" in error_msg.lower()

    def test_invalid_source_database(self, temp_mint_root, invalid_db_missing_tables):
        """Test import with invalid source database."""
        success, error_msg, workspace_key = import_database_as_workspace(
            str(invalid_db_missing_tables),
            "ws_name",
            temp_mint_root
        )
        
        assert success is False
        assert "missing" in error_msg.lower()
        assert workspace_key == ""

    def test_nonexistent_source_file(self, temp_mint_root, tmp_path):
        """Test import with non-existent source file."""
        success, error_msg, _ = import_database_as_workspace(
            str(tmp_path / "nonexistent.db"),
            "ws_name",
            temp_mint_root
        )
        
        assert success is False
        assert "not found" in error_msg.lower()

    def test_import_deactivates_previous_workspace(self, temp_mint_root, valid_workspace_db):
        """Test that import deactivates previously active workspace."""
        # Create first workspace
        con = duckdb.connect(str(temp_mint_root / "mint.db"))
        con.execute(
            "INSERT INTO workspaces (name, active, created_at, last_activity) VALUES ('existing', true, NOW(), NOW())"
        )
        con.close()
        
        # Import new workspace
        success, _, new_key = import_database_as_workspace(
            str(valid_workspace_db),
            "new_ws",
            temp_mint_root
        )
        
        assert success is True
        
        # Verify old workspace is now inactive
        con = duckdb.connect(str(temp_mint_root / "mint.db"))
        old_active = con.execute("SELECT active FROM workspaces WHERE name = 'existing'").fetchone()[0]
        new_active = con.execute("SELECT active FROM workspaces WHERE key = ?", (new_key,)).fetchone()[0]
        con.close()
        
        assert old_active is False
        assert new_active is True


class TestCompactDatabase:
    """Tests for compact_database()."""

    @pytest.fixture
    def workspace_with_data(self, tmp_path):
        """Create a workspace directory with database containing data."""
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        
        db_file = workspace_path / "workspace_mint.db"
        con = duckdb.connect(str(db_file))
        _create_tables(con)
        
        # Insert substantial data to make file size meaningful
        for i in range(100):
            con.execute(
                "INSERT INTO samples (ms_file_label, label) VALUES (?, ?)",
                (f'file_{i}', f'Label {i}')
            )
            con.execute(
                """INSERT INTO targets (peak_label, mz_mean, rt, rt_min, rt_max) 
                   VALUES (?, ?, ?, ?, ?)""",
                (f'target_{i}', 100.0 + i, 5.0, 4.0, 6.0)
            )
        
        # Insert some ms1_data to increase file size
        for i in range(50):
            for j in range(100):
                con.execute(
                    """INSERT INTO ms1_data (ms_file_label, scan_id, mz, intensity, scan_time)
                       VALUES (?, ?, ?, ?, ?)""",
                    (f'file_{i}', j, 100.0 + j, 1000.0 * j, 0.1 * j)
                )
        
        con.execute("CHECKPOINT")
        con.close()
        
        return workspace_path

    def test_compact_database_reduces_size_after_delete(self, workspace_with_data):
        """Test that compacting reduces file size after data is deleted."""
        workspace_path = workspace_with_data
        db_file = workspace_path / "workspace_mint.db"
        
        # Get initial size after data insertion
        initial_size = db_file.stat().st_size
        
        # Delete the big data (ms1_data)
        con = duckdb.connect(str(db_file))
        con.execute("DELETE FROM ms1_data")
        con.execute("CHECKPOINT")
        con.close()
        
        # Size after delete (DuckDB doesn't reclaim space)
        size_after_delete = db_file.stat().st_size
        
        # Compact the database
        success, message = compact_database(workspace_path)
        
        assert success is True
        assert "Compacted" in message
        
        # Size after compaction should be smaller
        final_size = db_file.stat().st_size
        
        # The final size should be less than before (significant reduction expected)
        assert final_size < size_after_delete, \
            f"Expected size reduction: {size_after_delete} -> {final_size}"

    def test_compact_database_preserves_data(self, workspace_with_data):
        """Test that compacting preserves all remaining data."""
        workspace_path = workspace_with_data
        db_file = workspace_path / "workspace_mint.db"
        
        # Delete some data
        con = duckdb.connect(str(db_file))
        con.execute("DELETE FROM samples WHERE ms_file_label LIKE 'file_5%'")
        con.execute("DELETE FROM targets WHERE peak_label LIKE 'target_5%'")
        remaining_samples = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        remaining_targets = con.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        con.close()
        
        # Compact
        success, _ = compact_database(workspace_path)
        assert success is True
        
        # Verify data is preserved
        con = duckdb.connect(str(db_file))
        samples_after = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        targets_after = con.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        con.close()
        
        assert samples_after == remaining_samples
        assert targets_after == remaining_targets

    def test_compact_database_nonexistent_path(self, tmp_path):
        """Test compacting a non-existent database."""
        success, message = compact_database(tmp_path / "nonexistent")
        
        assert success is False
        assert "not found" in message.lower()

    def test_compact_database_handles_empty_tables(self, tmp_path):
        """Test compacting a database with empty tables."""
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir()
        
        db_file = workspace_path / "workspace_mint.db"
        con = duckdb.connect(str(db_file))
        _create_tables(con)
        con.close()
        
        # Should succeed even with empty tables
        success, _ = compact_database(workspace_path)
        
        assert success is True
        
        # Verify database is still valid
        con = duckdb.connect(str(db_file))
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        con.close()
        
        assert 'samples' in tables
        assert 'targets' in tables
