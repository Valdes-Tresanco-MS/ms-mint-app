from contextlib import contextmanager
from pathlib import Path
from threading import Thread

import duckdb
import time
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

try:
    import lttbc as _lttbc
except Exception:
    _lttbc = None

try:
    from scipy.signal import savgol_filter as _savgol_filter
except Exception:
    _savgol_filter = None
from .sample_metadata import GROUP_COLUMNS

FULL_RANGE_DOWNSAMPLE_POINTS = 1000
FULL_RANGE_DOWNSAMPLE_BATCH = 200
FULL_RANGE_SAVGOL_WINDOW = 10
FULL_RANGE_SAVGOL_ORDER = 2


def _apply_lttb_downsampling(scan_time, intensity, n_out=FULL_RANGE_DOWNSAMPLE_POINTS):
    if n_out is None:
        n_out = FULL_RANGE_DOWNSAMPLE_POINTS

    try:
        n_out = int(n_out)
    except (TypeError, ValueError):
        return scan_time, intensity

    if _lttbc is None:
        return scan_time, intensity

    if not scan_time or n_out <= 0 or len(scan_time) <= n_out:
        return scan_time, intensity

    downsampled_x, downsampled_y = _lttbc.downsample(scan_time, intensity, n_out)
    if len(downsampled_x) == 0:
        return scan_time, intensity

    return downsampled_x, downsampled_y


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

    interior = np.convolve(intensity, coeffs[::-1], mode='valid')
    filtered[half_window:-half_window] = interior

    x = np.arange(window_length, dtype=float)
    left_poly = np.polyfit(x, intensity[:window_length], polyorder)
    filtered[:half_window] = np.polyval(left_poly, x[:half_window])
    right_poly = np.polyfit(x, intensity[-window_length:], polyorder)
    filtered[-half_window:] = np.polyval(right_poly, x[-half_window:])

    return filtered


def _apply_savgol_smoothing(intensity,
                            window_length=FULL_RANGE_SAVGOL_WINDOW,
                            polyorder=FULL_RANGE_SAVGOL_ORDER):
    if window_length is None:
        window_length = FULL_RANGE_SAVGOL_WINDOW
    if polyorder is None:
        polyorder = FULL_RANGE_SAVGOL_ORDER

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


def get_physical_cores() -> int:
    """
    Get the number of physical CPU cores (not hyperthreads).
    Falls back to os.cpu_count() // 2 if psutil can't detect.
    """
    import os
    import psutil
    
    physical = psutil.cpu_count(logical=False)
    if physical is None:
        # Fallback: assume hyperthreading, so divide logical by 2
        physical = max(1, (os.cpu_count() or 4) // 2)
    return physical


def calculate_optimal_params(user_cpus: int = None, user_ram: int = None) -> tuple:
    """
    Calculate optimal CPU, RAM, and batch_size based on system resources.
    
    Algorithm (data-driven from experimental benchmarks):
    1. CPUs: min(logical // 2, physical_cores) - avoids hyperthreads
    2. RAM: 50% of available, balanced with 1GB per CPU minimum
    3. Batch: 500 × RAM_GB, capped at 8000
    
    If user provides explicit values, those are used instead of auto-detection.
    
    Args:
        user_cpus: Optional user-specified CPU count
        user_ram: Optional user-specified RAM in GB
        
    Returns:
        (cpus, ram_gb, batch_size) tuple
    """
    import os
    import psutil
    
    # Step 1: CPU calculation
    if user_cpus is not None:
        target_cpus = user_cpus
    else:
        logical_cores = os.cpu_count() or 4
        physical_cores = get_physical_cores()
        # Use half of logical, but never exceed physical (no hyperthreading benefit)
        target_cpus = min(logical_cores // 2, physical_cores)
        target_cpus = max(1, target_cpus)  # At least 1 CPU
    
    # Step 2: RAM calculation
    if user_ram is not None:
        usable_ram = user_ram
    else:
        available_ram = psutil.virtual_memory().available / (1024 ** 3)
        usable_ram = int(available_ram * 0.5)  # 50% of available
        usable_ram = max(4, usable_ram)  # Minimum 4GB
    
    # Step 3: Balance CPUs and RAM (1GB per CPU minimum)
    if usable_ram < target_cpus:
        # RAM is limiting factor - reduce CPUs to match
        cpus = usable_ram
        ram_gb = usable_ram
    else:
        # CPU is limiting factor
        cpus = target_cpus
        # Cap RAM at 2× CPUs (no benefit beyond that based on experiments)
        ram_gb = min(usable_ram, cpus * 2)
    
    # Step 4: Batch size optimization
    # Benchmarks (Jan 2026) showed 1000-3000 is the "sweet spot" for throughput
    # and stability. Larger batches (5000+) reduced speed by 3x and increased crash risk.
    # New formula: 200 * RAM_GB, capped at 3000.
    batch_size = min(200 * ram_gb, 3000)
    batch_size = max(1000, batch_size)  # Minimum 1000 for efficiency
    
    return cpus, ram_gb, batch_size


def calculate_optimal_batch_size(ram_gb: int = None, total_pairs: int = 0, n_cpus: int = None) -> int:
    """
    Calculate optimal batch size for chromatogram/results extraction.
    
    This is a simplified wrapper around calculate_optimal_params() for
    backward compatibility with existing code that only needs batch_size.
    
    Formula (revised Jan 2026 based on large dataset benchmarks):
    - batch = 200 × RAM_GB, capped at 3000
    - Experiments showed batch 1000 was 3x faster than batch 5000
    - Large batches (8000+) caused high memory pressure and crashes
    
    Args:
        ram_gb: RAM in GB. If None, auto-calculates 50% of available.
        total_pairs: Total pairs (used for progress reporting constraint)
        n_cpus: Number of CPUs (used for balancing with RAM)
        
    Returns:
        Optimal batch size
    """
    _, effective_ram, batch_size = calculate_optimal_params(
        user_cpus=n_cpus,
        user_ram=ram_gb
    )
    
    # Ensure at least 10 batches for progress reporting
    if total_pairs > 0:
        batch_size = min(batch_size, max(total_pairs // 10, 500))
    
    return batch_size


def get_effective_cpus(n_cpus: int, ram_gb: int) -> int:
    """
    Calculate effective CPUs, capped at RAM (1GB per CPU minimum).
    
    Args:
        n_cpus: Requested number of CPUs
        ram_gb: Available RAM in GB
        
    Returns:
        Effective number of CPUs to use
    """
    if not n_cpus or not ram_gb:
        return n_cpus or 4  # Default to 4 if not specified
    return min(n_cpus, ram_gb)


# Required tables and their core columns for validation
REQUIRED_TABLES = {
    'samples': ['ms_file_label'],
    'targets': ['peak_label', 'mz_mean', 'rt', 'rt_min', 'rt_max'],
    'ms1_data': ['ms_file_label', 'scan_id', 'mz', 'intensity', 'scan_time'],
    'chromatograms': ['peak_label', 'ms_file_label'],
    'results': ['peak_label', 'ms_file_label', 'peak_area'],
}


def validate_mint_database(db_path: str) -> tuple[bool, str, dict]:
    """
    Validate that a DuckDB file is a valid MINT database.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        (is_valid, error_message, stats_dict)
        stats_dict contains row counts for each table when valid
        
    Checks:
        - File exists and is readable
        - Contains required tables
        - Tables have expected core columns
    """
    import shutil
    
    db_path = Path(db_path)
    stats = {}
    
    # Check file exists
    if not db_path.exists():
        return False, f"File not found: {db_path}", stats
    
    if not db_path.is_file():
        return False, f"Not a file: {db_path}", stats
    
    # Check file extension
    if db_path.suffix.lower() not in ['.db', '.duckdb']:
        return False, f"Invalid file extension: {db_path.suffix}. Expected .db or .duckdb", stats
    
    # Try to open the database
    con = None
    try:
        con = duckdb.connect(database=str(db_path), read_only=True)
        
        # Get list of tables
        tables_result = con.execute("SHOW TABLES").fetchall()
        existing_tables = {row[0] for row in tables_result}
        
        # Check required tables
        missing_tables = []
        for table in REQUIRED_TABLES:
            if table not in existing_tables:
                missing_tables.append(table)
        
        if missing_tables:
            return False, f"Missing required tables: {', '.join(missing_tables)}", stats
        
        # Check columns for each required table
        for table, required_cols in REQUIRED_TABLES.items():
            cols_result = con.execute(f"DESCRIBE {table}").fetchall()
            existing_cols = {row[0] for row in cols_result}
            
            missing_cols = [col for col in required_cols if col not in existing_cols]
            if missing_cols:
                return False, f"Table '{table}' missing columns: {', '.join(missing_cols)}", stats
            
            # Get row count
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count
        
        # Check optional ms2_data table
        if 'ms2_data' in existing_tables:
            count = con.execute("SELECT COUNT(*) FROM ms2_data").fetchone()[0]
            stats['ms2_data'] = count
        
        return True, "", stats
        
    except duckdb.IOException as e:
        return False, f"Cannot read database file: {e}", stats
    except duckdb.InvalidInputException as e:
        return False, f"Invalid database file: {e}", stats
    except Exception as e:
        return False, f"Database validation error: {e}", stats
    finally:
        if con:
            con.close()


def import_database_as_workspace(
    db_path: str, 
    workspace_name: str, 
    mint_root: Path
) -> tuple[bool, str, str]:
    """
    Import a DuckDB file as a new workspace.
    
    Args:
        db_path: Path to the source database file
        workspace_name: Name for the new workspace
        mint_root: Root directory for MINT data (e.g., /path/to/MINT/Local)
        
    Returns:
        (success, error_message, workspace_key)
        
    Steps:
        1. Validate database
        2. Create workspace record in mint.db
        3. Create workspace folder
        4. Copy database to workspace folder
    """
    import shutil
    import uuid
    
    db_path = Path(db_path)
    mint_root = Path(mint_root)
    
    # Step 1: Validate the source database
    is_valid, error_msg, stats = validate_mint_database(str(db_path))
    if not is_valid:
        return False, error_msg, ""
    
    # Step 2: Create workspace record
    workspace_key = None
    try:
        with duckdb_connection_mint(mint_root) as mint_conn:
            if mint_conn is None:
                return False, "Cannot connect to MINT database", ""
            
            # Check if name already exists
            existing = mint_conn.execute(
                "SELECT COUNT(*) FROM workspaces WHERE name = ?", 
                (workspace_name,)
            ).fetchone()[0]
            
            if existing > 0:
                return False, f"Workspace name '{workspace_name}' already exists", ""
            
            # Deactivate current active workspace
            mint_conn.execute("UPDATE workspaces SET active = false WHERE active = true")
            
            # Insert new workspace
            result = mint_conn.execute(
                """INSERT INTO workspaces (name, description, active, created_at, last_activity) 
                   VALUES (?, ?, true, NOW(), NOW()) RETURNING key""",
                (workspace_name, f"Imported from {db_path.name}")
            ).fetchone()
            
            if result:
                workspace_key = str(result[0])
            else:
                return False, "Failed to create workspace record", ""
                
    except Exception as e:
        logger.error(f"Error creating workspace record: {e}", exc_info=True)
        return False, f"Failed to create workspace: {e}", ""
    
    # Step 3: Create workspace folder
    workspace_path = mint_root / 'workspaces' / workspace_key
    try:
        workspace_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # Rollback: delete the workspace record
        try:
            with duckdb_connection_mint(mint_root) as mint_conn:
                if mint_conn:
                    mint_conn.execute("DELETE FROM workspaces WHERE key = ?", (workspace_key,))
        except Exception:
            pass
        return False, f"Failed to create workspace folder: {e}", ""
    
    # Step 4: Copy database file
    dest_db_path = workspace_path / 'workspace_mint.db'
    try:
        shutil.copy2(str(db_path), str(dest_db_path))
        logger.info(f"Imported database from {db_path} to workspace {workspace_name} (key: {workspace_key})")
    except Exception as e:
        # Rollback: delete folder and workspace record
        try:
            shutil.rmtree(workspace_path, ignore_errors=True)
            with duckdb_connection_mint(mint_root) as mint_conn:
                if mint_conn:
                    mint_conn.execute("DELETE FROM workspaces WHERE key = ?", (workspace_key,))
        except Exception:
            pass
        return False, f"Failed to copy database file: {e}", ""
    
    return True, "", workspace_key


def _send_progress(set_progress, percent, stage: str = "", detail: str = ""):
    """
    Safely call the provided set_progress callback.

    Supports custom stage/detail strings when the callback accepts them,
    and falls back to simple percent-only updates otherwise.
    
    IMPORTANT: Re-raises Cancelled/PreventUpdate exceptions for proper cancellation.
    """
    if not set_progress:
        return
    try:
        set_progress(percent, stage, detail)
    except TypeError:
        try:
            set_progress(percent)
        except (SystemExit, KeyboardInterrupt):
            raise  # Always re-raise these
        except Exception as e:
            # Re-raise cancel-related exceptions
            if 'Cancelled' in type(e).__name__ or 'PreventUpdate' in type(e).__name__:
                raise
            pass  # Suppress other exceptions
    except (SystemExit, KeyboardInterrupt):
        raise  # Always re-raise these
    except Exception as e:
        # Re-raise cancel-related exceptions (dash.exceptions.Cancelled)
        if 'Cancelled' in type(e).__name__ or 'PreventUpdate' in type(e).__name__:
            raise
        pass  # Suppress other exceptions




def _update_workspace_activity(mint_root: Path, workspace_id: str, retries: int = 3, delay_s: float = 0.05):
    for attempt in range(retries):
        try:
            with duckdb_connection_mint(mint_root) as mint_conn:
                if mint_conn:
                    mint_conn.execute(
                        "UPDATE workspaces SET last_activity = NOW() WHERE key = ?",
                        [workspace_id],
                    )
            return
        except Exception as e:
            message = str(e)
            if "TransactionContext Error: Conflict on update!" in message:
                time.sleep(delay_s * (attempt + 1))
                continue
            logger.error(f"Error updating workspace activity: {e}")
            return


class DatabaseCorruptionError(Exception):
    """Raised when the DuckDB file is corrupted."""
    pass


# Global tracker for corrupted workspaces - allows plugins to show notifications
_corrupted_workspaces: set[str] = set()


def is_workspace_corrupted(workspace_path: Path | str) -> bool:
    """Check if a workspace has been marked as corrupted."""
    if not workspace_path:
        return False
    return str(workspace_path) in _corrupted_workspaces


def clear_corruption_flag(workspace_path: Path | str):
    """Clear the corruption flag for a workspace (e.g., after user acknowledges)."""
    _corrupted_workspaces.discard(str(workspace_path))


def _mark_corrupted(workspace_path: Path | str):
    """Mark a workspace as corrupted."""
    _corrupted_workspaces.add(str(workspace_path))


def get_corruption_notification():
    """
    Returns a notification component for a corrupted database.
    
    Plugins can use this when they detect conn is None and is_workspace_corrupted() is True.
    Returns a dict suitable for fac.AntdNotification or None if not applicable.
    """
    return {
        'message': "⚠️ Database Corrupted",
        'description': "This workspace has a corrupted database. Please delete it and restore from backup or recreate the workspace.",
        'type': "error",
        'duration': 10,
        'placement': 'bottom',
        'showProgress': True,
    }


@contextmanager
def duckdb_connection(workspace_path: Path | str, register_activity=True, n_cpus=None, ram=None):
    """
    Provides a DuckDB connection as a context manager.

    The database file will be named 'mint.db' and will be located inside the workspace directory.

    :param workspace_path: The path to the MINT workspace directory.
    """
    if not workspace_path:
        yield None
        return
    workspace_path = Path(workspace_path)
    db_file = Path(workspace_path, 'workspace_mint.db')
    # print(f"Connecting to DuckDB at: {db_file}")
    con = None
    try:
        con = duckdb.connect(database=str(db_file), read_only=False)
        con.execute("PRAGMA enable_checkpoint_on_shutdown")
        con.execute("SET enable_progress_bar = true")
        con.execute("SET enable_progress_bar_print = false")
        con.execute("SET progress_bar_time = 0")
        # Try to set temp_directory, but don't fail if it can't be changed
        try:
            con.execute(f"SET temp_directory = '{workspace_path.as_posix()}';")
        except Exception:
            pass  # temp_directory already set or can't be changed - not critical
        if n_cpus:
            # Cap CPUs at RAM (1GB per CPU minimum to prevent resource imbalance)
            effective_cpus = get_effective_cpus(n_cpus, ram) if ram else n_cpus
            con.execute(f"SET threads = {effective_cpus}")
            if effective_cpus != n_cpus:
                logger.info(f"DuckDB threads capped at {effective_cpus} (requested {n_cpus}, RAM limit {ram}GB)")
            else:
                logger.debug(f"DuckDB threads set to {effective_cpus}")
        if ram:
            con.execute(f"SET memory_limit = '{ram}GB'")
            # Verify the setting was applied
            actual_limit = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
            logger.info(f"DuckDB memory_limit set to {ram}GB (verified: {actual_limit})")
        _create_tables(con)
    except duckdb.IOException as e:
        if "Corrupt database file" in str(e):
            _mark_corrupted(workspace_path)  # Mark for UI notification
            logger.critical(
                f"⚠️ DATABASE CORRUPTION DETECTED in {db_file}: {e}\n"
                "This usually happens due to a system crash or forced termination during a write operation.\n"
                "Please delete this workspace and restore from backup or recreate it."
            )
            # Return None so that all existing "if conn is None" checks handle this gracefully
            yield None
            return
        logger.error(f"Error connecting to DuckDB: {e}")
        yield None
        return
    except Exception as e:
        logger.error(f"Error connecting to DuckDB: {e}")
        yield None
        return
    try:
        yield con
    finally:
        if con:
            if register_activity:
                try:
                    workspace_id = Path(workspace_path).name
                    mint_root = workspace_path.parent.parent
                    _update_workspace_activity(mint_root, workspace_id)
                except Exception as e:
                    logger.error(f"Error updating workspace activity: {e}")
            if n_cpus:
                con.execute("RESET threads")
            if ram:
                con.execute("RESET memory_limit")
            con.close()


@contextmanager
def duckdb_connection_mint(mint_path: Path, workspace=None):
    if not mint_path:
        yield None
        return

    db_file = Path(mint_path, 'mint.db')
    con = None
    try:
        con = duckdb.connect(database=db_file, read_only=False)
        _create_workspace_tables(con)
    except Exception as e:
        logger.error(f"Error connecting to DuckDB: {e}")
        yield None
        return
    try:
        yield con
    finally:
        if con:
            if workspace:
                try:
                    con.execute("UPDATE workspaces SET last_activity = NOW() WHERE key = ?", [workspace])
                except Exception:
                    pass
            con.close()


def compact_database(workspace_path: Path | str, max_retries: int = 5, initial_delay: float = 0.5) -> tuple[bool, str]:
    """
    Compact the workspace database by rebuilding it.
    
    DuckDB doesn't automatically reclaim space after DELETEs, so this function
    creates a new database file with only the current data, then replaces
    the old file.
    
    Args:
        workspace_path: Path to the workspace directory
        max_retries: Maximum number of retry attempts for lock conflicts
        initial_delay: Initial delay between retries (doubles each attempt)
        
    Returns:
        (success, error_message)
    """
    import shutil
    import uuid
    
    workspace_path = Path(workspace_path)
    db_file = workspace_path / 'workspace_mint.db'
    
    if not db_file.exists():
        return False, f"Database file not found: {db_file}"
    
    # Get original file size for logging
    original_size = db_file.stat().st_size
    original_size_mb = original_size / (1024 * 1024)
    
    # Create temporary file name for the new database
    temp_db_file = workspace_path / f'workspace_mint_{uuid.uuid4().hex[:8]}.db.tmp'
    backup_file = workspace_path / 'workspace_mint.db.bak'
    
    try:
        # Create new database and copy all tables
        new_con = None
        try:
            new_con = duckdb.connect(database=str(temp_db_file), read_only=False)
            new_con.execute("PRAGMA enable_checkpoint_on_shutdown")
            
            # Limit memory to 50% of available RAM to prevent system freeze
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            memory_limit_gb = max(2, int(available_ram_gb * 0.5))  # At least 2GB, max 50%
            new_con.execute(f"SET memory_limit = '{memory_limit_gb}GB'")
            logger.debug(f"Compaction memory limit set to {memory_limit_gb}GB (50% of {available_ram_gb:.1f}GB available)")
            
            # Create all types and tables in the new database
            _create_tables(new_con)
            
            # Attach the old database with retry logic for lock conflicts
            delay = initial_delay
            attached = False
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    new_con.execute(f"ATTACH '{db_file}' AS old_db (READ_ONLY)")
                    attached = True
                    break
                except duckdb.IOException as e:
                    last_error = e
                    if "lock" in str(e).lower() and attempt < max_retries - 1:
                        logger.debug(f"Compact: lock conflict on attach, retry {attempt + 1}/{max_retries} after {delay:.1f}s")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        raise
            
            if not attached:
                raise last_error or Exception("Failed to attach old database")
            
            # Copy data from all tables
            tables_to_copy = ['samples', 'targets', 'ms1_data', 'ms2_data', 'chromatograms', 'results']
            for table in tables_to_copy:
                # Check if table has data in old database
                try:
                    count = new_con.execute(f"SELECT COUNT(*) FROM old_db.{table}").fetchone()[0]
                    if count > 0:
                        # Copy data - INSERT INTO ... SELECT * FROM ...
                        new_con.execute(f"INSERT INTO {table} SELECT * FROM old_db.{table}")
                        logger.debug(f"Compacted table '{table}': {count} rows")
                except Exception as e:
                    # Table might not exist in old DB, which is fine
                    logger.debug(f"Skipping table '{table}' during compaction: {e}")
            
            # Detach old database
            new_con.execute("DETACH old_db")
            
            # Checkpoint to ensure all data is written
            new_con.execute("CHECKPOINT")
            
        finally:
            if new_con:
                new_con.close()
        
        # Now swap the files
        # Rename old to backup
        if backup_file.exists():
            backup_file.unlink()
        db_file.rename(backup_file)
        
        # Rename new to original
        temp_db_file.rename(db_file)
        
        # Remove backup
        if backup_file.exists():
            backup_file.unlink()
        
        # Log the size reduction
        new_size = db_file.stat().st_size
        new_size_mb = new_size / (1024 * 1024)
        reduction_pct = ((original_size - new_size) / original_size * 100) if original_size > 0 else 0
        
        logger.info(f"Database compacted: {original_size_mb:.1f}MB -> {new_size_mb:.1f}MB ({reduction_pct:.0f}% reduction)")
        
        return True, f"Compacted {original_size_mb:.1f}MB -> {new_size_mb:.1f}MB"
        
    except Exception as e:
        logger.error(f"Error compacting database: {e}", exc_info=True)
        
        # Cleanup: restore backup if it exists
        if backup_file.exists() and not db_file.exists():
            try:
                backup_file.rename(db_file)
                logger.info("Restored database from backup after failed compaction")
            except Exception as restore_error:
                logger.error(f"Failed to restore backup: {restore_error}")
        
        # Remove temp file if it exists
        if temp_db_file.exists():
            try:
                temp_db_file.unlink()
            except Exception:
                pass
        
        return False, str(e)


def _create_tables(conn: duckdb.DuckDBPyConnection):
    # Create tables if they don't exist
    conn.execute("CREATE TYPE IF NOT EXISTS ms_type_enum AS ENUM ('ms1', 'ms2');")
    conn.execute("CREATE TYPE IF NOT EXISTS polarity_enum AS ENUM ('Positive', 'Negative');")
    conn.execute("CREATE TYPE IF NOT EXISTS unit_type_enum AS ENUM ('s', 'min');")

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS samples
                 (
                     ms_file_label        VARCHAR PRIMARY KEY,
                     ms_type              ms_type_enum,
                     file_type            VARCHAR,
                     use_for_optimization BOOLEAN DEFAULT true,
                     use_for_processing   BOOLEAN DEFAULT true,
                     use_for_analysis     BOOLEAN DEFAULT true,
                     polarity             polarity_enum,
                     color                VARCHAR DEFAULT '#BBBBBB',
                     label                VARCHAR,
                     sample_type          VARCHAR DEFAULT 'Sample',
                     group_1              VARCHAR,
                     group_2              VARCHAR,
                     group_3              VARCHAR,
                     group_4              VARCHAR,
                     group_5              VARCHAR
                 );
                 """)

    # Backfill new processing flag for existing DBs
    conn.execute("ALTER TABLE samples ADD COLUMN IF NOT EXISTS use_for_processing BOOLEAN DEFAULT true;")
    conn.execute("ALTER TABLE samples ADD COLUMN IF NOT EXISTS file_type VARCHAR;")
    conn.execute("ALTER TABLE samples ADD COLUMN IF NOT EXISTS acquisition_datetime TIMESTAMP;")
    for col in GROUP_COLUMNS:
        conn.execute(f"ALTER TABLE samples ADD COLUMN IF NOT EXISTS {col} VARCHAR;")
    try:
        conn.execute("""
                     UPDATE samples
                     SET use_for_processing = COALESCE(use_for_processing, use_for_analysis, TRUE)
                     """)
    except Exception:
        # Avoid failing during initialization if another write is in flight.
        pass

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS ms1_data
                 (
                     ms_file_label      VARCHAR,  -- Label of the MS file, linking to samples
                     scan_id            INTEGER,  -- Scan ID
                     mz                 DOUBLE,   -- Mass-to-charge ratio
                     intensity          DOUBLE,   -- Intensity
                     scan_time          DOUBLE    -- Scan time
                 );
                 """)
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS ms2_data
                 (
                     ms_file_label      VARCHAR,  -- Label of the MS file, linking to samples
                     scan_id            INTEGER,  -- Scan ID
                     mz                 DOUBLE,   -- Mass-to-charge ratio
                     intensity          DOUBLE,   -- Intensity
                     scan_time          DOUBLE,   -- Scan time
                     mz_precursor       DOUBLE,   -- Precursor m/z
                     filterLine         VARCHAR,  -- Filter line from the raw file
                     filterLine_ELMAVEN VARCHAR   -- Filter line formatted for El-Maven
                 );
                 """)

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS targets
                 (
                     peak_label          VARCHAR PRIMARY KEY, -- Label for the peak
                     mz_mean             DOUBLE,              -- Mean mass-to-charge ratio
                     mz_width            DOUBLE,              -- Width of the m/z window
                     mz                  DOUBLE,              -- Mass-to-charge ratio
                     rt                  DOUBLE,              -- Retention time
                     rt_min              DOUBLE,              -- Minimum retention time
                     rt_max              DOUBLE,              -- Maximum retention time
                     rt_unit             unit_type_enum,      -- Unit of retention time
                     intensity_threshold DOUBLE,              -- Intensity threshold
                     polarity            polarity_enum,       -- Polarity of the target
                     filterLine          VARCHAR,             -- Filter line from the raw file
                     ms_type             ms_type_enum,        -- MS type (ms1 or ms2)
                     category            VARCHAR,             -- Category of the target
                     peak_selection      BOOLEAN,             -- Preselected target
                     score               DOUBLE,              -- Score of the target
                     bookmark            BOOLEAN,             -- Bookmark the target
                     source              VARCHAR,             -- Filename of the target list
                     notes               VARCHAR,             -- Additional notes for the target
                     rt_auto_adjusted    BOOLEAN DEFAULT FALSE -- RT was auto-adjusted (outside span)
                 );
                 """)
    # Backfill rt_auto_adjusted for existing DBs
    conn.execute("ALTER TABLE targets ADD COLUMN IF NOT EXISTS rt_auto_adjusted BOOLEAN DEFAULT FALSE;")
    
    # RT Alignment columns for storing alignment parameters
    conn.execute("ALTER TABLE targets ADD COLUMN IF NOT EXISTS rt_align_enabled BOOLEAN DEFAULT FALSE;")
    conn.execute("ALTER TABLE targets ADD COLUMN IF NOT EXISTS rt_align_reference_rt DOUBLE;")
    conn.execute("ALTER TABLE targets ADD COLUMN IF NOT EXISTS rt_align_shifts JSON;")
    conn.execute("ALTER TABLE targets ADD COLUMN IF NOT EXISTS rt_align_rt_min DOUBLE;")
    conn.execute("ALTER TABLE targets ADD COLUMN IF NOT EXISTS rt_align_rt_max DOUBLE;")

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS chromatograms
                 (
                     peak_label    VARCHAR,
                     ms_file_label VARCHAR,
                     scan_time     DOUBLE[],
                     intensity     DOUBLE[],
                     scan_time_full_ds DOUBLE[],
                     intensity_full_ds DOUBLE[],
                     mz_arr        DOUBLE[],
                     ms_type       ms_type_enum,
                     PRIMARY KEY (ms_file_label, peak_label)
                 );
                 """)

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS results
                 (
                     peak_label        VARCHAR,
                     ms_file_label     VARCHAR,
                     total_intensity   DOUBLE,
                     peak_area         DOUBLE,
                     peak_area_top3    DOUBLE,
                     peak_max          DOUBLE,
                     peak_min          DOUBLE,
                     peak_mean         DOUBLE,
                     peak_rt_of_max    DOUBLE,
                     peak_median       DOUBLE,
                     peak_n_datapoints INT,
                     rt_aligned        BOOLEAN,  -- TRUE if RT alignment was applied
                     rt_shift          DOUBLE,   -- Shift value applied (0 if not aligned)
                     scan_time         DOUBLE[],
                     intensity         DOUBLE[],
                     PRIMARY KEY (ms_file_label, peak_label)
                 );
                 """)

    # Migration: Add mz_arr column to chromatograms table
    try:
        chrom_cols = {row[0] for row in conn.execute("DESCRIBE chromatograms").fetchall()}
        if 'mz_arr' not in chrom_cols:
            conn.execute("ALTER TABLE chromatograms ADD COLUMN mz_arr DOUBLE[]")
            logger.debug("Migration: Added 'mz_arr' column to chromatograms table")
        if 'scan_time_full_ds' not in chrom_cols:
            conn.execute("ALTER TABLE chromatograms ADD COLUMN scan_time_full_ds DOUBLE[]")
            logger.debug("Migration: Added 'scan_time_full_ds' column to chromatograms table")
        if 'intensity_full_ds' not in chrom_cols:
            conn.execute("ALTER TABLE chromatograms ADD COLUMN intensity_full_ds DOUBLE[]")
            logger.debug("Migration: Added 'intensity_full_ds' column to chromatograms table")
    except Exception:
        pass
    
    # Migration: Add rt_aligned, rt_shift, peak_mz_of_max columns to existing results tables
    existing_cols = {
        row[0] for row in conn.execute("DESCRIBE results").fetchall()
    }
    if 'rt_aligned' not in existing_cols:
        conn.execute("ALTER TABLE results ADD COLUMN rt_aligned BOOLEAN")
        logger.debug("Migration: Added 'rt_aligned' column to results table")
    if 'rt_shift' not in existing_cols:
        conn.execute("ALTER TABLE results ADD COLUMN rt_shift DOUBLE")
        logger.debug("Migration: Added 'rt_shift' column to results table")
    if 'peak_mz_of_max' not in existing_cols:
        conn.execute("ALTER TABLE results ADD COLUMN peak_mz_of_max DOUBLE")
        logger.debug("Migration: Added 'peak_mz_of_max' column to results table")
    
    # Migration: Add EMG peak fitting columns to existing results tables
    fitting_columns = {
        'peak_area_fitted': 'DOUBLE',      # Area under fitted EMG curve
        'peak_sigma': 'DOUBLE',            # Gaussian width (σ)
        'peak_tau': 'DOUBLE',              # Exponential tail decay (τ/gamma)
        'peak_asymmetry': 'DOUBLE',        # τ/σ ratio
        'peak_rt_fitted': 'DOUBLE',        # Peak center from EMG fit
        'fit_r_squared': 'DOUBLE',         # Goodness of fit (R²)
        'fit_success': 'BOOLEAN',          # Whether fitting converged
    }
    for col_name, col_type in fitting_columns.items():
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE results ADD COLUMN {col_name} {col_type}")
            logger.debug(f"Migration: Added '{col_name}' column to results table")



def _create_workspace_tables(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS workspaces
                 (
                     key           UUID DEFAULT uuidv4() PRIMARY KEY,
                     name          VARCHAR,
                     description   VARCHAR,
                     active        BOOLEAN,
                     created_at    TIMESTAMP,
                     last_activity TIMESTAMP
                 )
                 """
                 )


def build_where_and_params(filter_, filterOptions):
    where_sql, params = [], []

    if not isinstance(filter_, dict) or not filter_:
        return "", []

    for key, value in filter_.items():
        if not value:
            continue
        # keyword (ILIKE)
        if filterOptions[key].get('filterMode') == 'keyword':
            where_sql.append(f'"{key}" ILIKE ?')
            params.append(f"%{value[0]}%")
        # multiple selection (IN)
        else:
            ph = ",".join("?" for _ in value)
            where_sql.append(f'"{key}" IN ({ph})')
            params.extend(value)
    where_clause = f"WHERE {' AND '.join(where_sql)}" if where_sql else ""
    return where_clause, params


def build_order_by(
        sorter: dict | None,
        column_types: dict[str, str],
        *,
        tie: tuple[str, str] | None = None,  # e.g. ("id", "ASC"); used ONLY when a sorter is present
        nocase_text: bool = True
) -> str:
    """
    Returns 'ORDER BY ...' or '' if there is no valid sorter.
    - sorter: {'columns': [...], 'orders': ['ascend'|'descend', ...]}
    - column_types: map {col -> DUCKDB type} (from DESCRIBE)
    - tie: optional (col, dir); added ONLY when there is at least one sortable column in sorter
    """
    # 0) Normalize input
    cols_in = (sorter or {}).get("columns") or []
    ords_in = (sorter or {}).get("orders") or []
    if not cols_in:
        return ""  # no sorter => no ORDER BY

    # Fill missing order entries
    if len(ords_in) < len(cols_in):
        ords_in = ords_in + ["ascend"] * (len(cols_in) - len(ords_in))

    order_map = {"ascend": "ASC", "descend": "DESC"}
    parts: list[str] = []
    used_cols: set[str] = set()

    for col, ord_ in zip(cols_in, ords_in):
        if col not in column_types:
            continue  # ignore invalid columns
        direction = order_map.get(ord_, "ASC")
        nulls = "NULLS LAST" if direction == "ASC" else "NULLS FIRST"
        ctype = (column_types.get(col) or "").upper()
        is_text = any(t in ctype for t in ("CHAR", "VARCHAR", "TEXT", "STRING"))
        expr = f'"{col}" COLLATE NOCASE' if (nocase_text and is_text) else f'"{col}"'
        parts.append(f"{expr} {direction} {nulls}")
        used_cols.add(col)

    # If nothing valid remains, do not sort (and do not add tie)
    if not parts:
        return ""

    # Add tie ONLY if there is a valid sorter, tie was requested, and the column is not duplicated
    if tie:
        tie_col, tie_dir = tie[0], tie[1].upper()
        if tie_col not in used_cols and tie_col in column_types:
            parts.append(f'"{tie_col}" {tie_dir}')

    return f"ORDER BY {', '.join(parts)}"


def build_paginated_query_by_peak(
        conn,
        filter_: dict | None = None,
        filterOptions: dict | None = None,
        sorter: dict | None = None,
        limit: int = 10,
        offset: int = 0
) -> tuple[str, list]:
    """
    Build a paginated query grouped by peak_label.
    Uses build_where_and_params() and build_order_by() without modifying them.
    """

    # Get column types
    column_types = {
        row[0]: row[1]
        for row in conn.execute("DESCRIBE results").fetchall()
    }

    # 1. Build WHERE clause and params to filter rows
    where_sql, where_params = build_where_and_params(filter_, filterOptions or {})

    # 2. Build ORDER BY for individual rows
    order_by_sql = build_order_by(
        sorter,
        column_types,
        tie=("peak_label", "ASC"),
        nocase_text=True
    )

    # 3. Extract ordering columns to aggregate by peak_label
    agg_exprs = []
    order_exprs = []

    if order_by_sql:
        # Parse ORDER BY to extract columns
        order_part = order_by_sql.replace("ORDER BY", "").strip()
        for clause in order_part.split(","):
            clause = clause.strip()
            # Remove modifiers
            clean = clause.replace("COLLATE NOCASE", "").replace("NULLS LAST", "").replace("NULLS FIRST", "")
            parts = clean.split()
            if not parts:
                continue

            col = parts[0].strip('"')
            direction = parts[1] if len(parts) > 1 else "ASC"

            if col == "peak_label":
                agg_exprs.append(f"peak_label AS _ord_{col}")
                order_exprs.append(f"_ord_{col} {direction}")
            else:
                # For other columns, use MAX/MIN depending on direction
                ctype = (column_types.get(col) or "").upper()
                is_numeric = any(t in ctype for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "REAL"))

                if is_numeric:
                    # For numeric: MAX if DESC, MIN if ASC (representative value)
                    agg_func = "MAX" if "DESC" in direction else "MIN"
                    agg_exprs.append(f'{agg_func}("{col}") AS _ord_{col}')
                    order_exprs.append(f"_ord_{col} {direction}")
                else:
                    # For text: MAX/MIN depending on direction
                    agg_func = "MAX" if "DESC" in direction else "MIN"
                    agg_exprs.append(f'{agg_func}("{col}") AS _ord_{col}')
                    order_exprs.append(f"_ord_{col} {direction}")

    # If there is no ordering, use peak_label by default
    if not agg_exprs:
        agg_exprs.append("peak_label AS _ord_peak_label")
        order_exprs.append("_ord_peak_label ASC")

    # 4. Build the query with CTEs
    sql = f"""
    WITH filtered AS (
      SELECT *
      FROM results
      {where_sql}
    ),
    peak_ordering AS (
      SELECT 
        peak_label,
        {', '.join(agg_exprs)},
        ROW_NUMBER() OVER(ORDER BY {', '.join(order_exprs)}) AS _rn
      FROM filtered
      GROUP BY peak_label
    ),
    total_peaks AS (
      SELECT COUNT(*) AS __total__
      FROM peak_ordering
    ),
    paged_peaks AS (
      SELECT peak_label, _rn
      FROM peak_ordering
      WHERE _rn > ? AND _rn <= ? + ?
    ),
    paged AS (
      SELECT 
        f.*,
        pp._rn AS __peak_order__,
        (SELECT __total__ FROM total_peaks) AS __total__
      FROM filtered f
      JOIN paged_peaks pp ON f.peak_label = pp.peak_label
      ORDER BY pp._rn, f.ms_file_label
    )
    SELECT * FROM paged;
    """

    # 5. Combine parameters: first WHERE params, then pagination
    all_params = where_params + [offset, offset, limit]

    return sql, all_params


def compute_and_insert_chromatograms_from_ms_data(con: duckdb.DuckDBPyConnection,
                                                  set_progress=None,
                                                  for_optimization=True,
                                                  recompute_ms1=False,
                                                  recompute_ms2=False):
    """
    Computes chromatograms from raw MS data and inserts them into the 'chromatograms' table.

    :param con: An active DuckDB connection.
    :param set_progress: Optional callback function to report progress (0-100).
    :param recompute_ms1: If True, deletes existing MS1 chromatograms before recomputing.
    :param recompute_ms2: If True, deletes existing MS2 chromatograms before recomputing.
    """

    info = con.execute("""
                       WITH samples_to_use AS (SELECT DISTINCT ms_file_label
                                               FROM samples
                                               WHERE use_for_optimization = TRUE
                                                  OR use_for_processing = TRUE),
                            ms1_targets AS (SELECT DISTINCT t.peak_label, s.ms_file_label
                                            FROM targets t
                                                     CROSS JOIN samples_to_use s
                                            WHERE t.mz_mean IS NOT NULL
                                                AND t.mz_width IS NOT NULL
                                                AND (t.peak_selection IS TRUE
                                                   OR NOT EXISTS (SELECT 1
                                                                  FROM targets t1
                                                                  WHERE t1.peak_selection IS TRUE))
                                                AND
                                                  EXISTS(SELECT 1 FROM ms1_data md WHERE md.ms_file_label = s.ms_file_label)),
                            ms2_targets AS (SELECT DISTINCT t.peak_label, s.ms_file_label
                                            FROM targets t
                                                     CROSS JOIN samples_to_use s
                                            WHERE t.filterLine IS NOT NULL -- ensures this is MS2
                                                AND (t.peak_selection IS TRUE
                                                   OR NOT EXISTS (SELECT 1
                                                                  FROM targets t1
                                                                  WHERE t1.peak_selection IS TRUE))
                                                AND
                                                  EXISTS(SELECT 1 FROM ms2_data md WHERE md.ms_file_label = s.ms_file_label)),
                            existing_chromatograms AS (SELECT DISTINCT peak_label, ms_file_label
                                                       FROM chromatograms)
                       SELECT
                           -- MS1 info
                           (SELECT COUNT(*) FROM ms1_targets)                          AS ms1_total_pairs,
                           (SELECT COUNT(*)
                            FROM ms1_targets mt
                                     JOIN existing_chromatograms ec
                                          ON ec.peak_label = mt.peak_label
                                              AND ec.ms_file_label = mt.ms_file_label) AS ms1_existing_pairs,
                           (SELECT COUNT(*)
                            FROM ms1_targets mt
                                     LEFT JOIN existing_chromatograms ec
                                               ON ec.peak_label = mt.peak_label
                                                   AND ec.ms_file_label = mt.ms_file_label
                            WHERE ec.peak_label IS NULL)                               AS ms1_missing_pairs,

                           -- MS2 info
                           (SELECT COUNT(*) FROM ms2_targets)                          AS ms2_total_pairs,
                           (SELECT COUNT(*)
                            FROM ms2_targets mt
                                     JOIN existing_chromatograms ec
                                          ON ec.peak_label = mt.peak_label
                                              AND ec.ms_file_label = mt.ms_file_label) AS ms2_existing_pairs,
                           (SELECT COUNT(*)
                            FROM ms2_targets mt
                                     LEFT JOIN existing_chromatograms ec
                                               ON ec.peak_label = mt.peak_label
                                                   AND ec.ms_file_label = mt.ms_file_label
                            WHERE ec.peak_label IS NULL)                               AS ms2_missing_pairs
                       """).fetchone()

    (ms1_total, ms1_existing, ms1_missing,
     ms2_total, ms2_existing, ms2_missing) = info

    # Decide what to process
    process_ms1 = ms1_total > 0 and (recompute_ms1 or ms1_missing > 0)
    process_ms2 = ms2_total > 0 and (recompute_ms2 or ms2_missing > 0)

    # Informational logging
    if ms1_total > 0:
        logger.info(f"MS1: {ms1_existing} existing, {ms1_missing} missing (total: {ms1_total})")
    if ms2_total > 0:
        logger.info(f"MS2: {ms2_existing} existing, {ms2_missing} missing (total: {ms2_total})")

    if not process_ms1 and not process_ms2:
        logger.info("No chromatograms to process.")
        return

    # Remove existing chromatograms if recomputation is requested
    if recompute_ms1 and ms1_existing > 0:
        logger.info(f"Deleting {ms1_existing} existing MS1 chromatograms for recalculation...")
        con.execute("""
                    DELETE
                    FROM chromatograms
                    WHERE EXISTS(SELECT 1
                                 FROM targets t
                                          CROSS JOIN samples s
                                 WHERE s.use_for_optimization = TRUE
                                    OR s.use_for_processing = TRUE
                                     AND t.mz_mean IS NOT NULL
                                     AND t.mz_width IS NOT NULL
                                     AND (t.peak_selection IS TRUE
                                        OR NOT EXISTS (SELECT 1
                                                       FROM targets t1
                                                       WHERE t1.peak_selection IS TRUE))
                                     AND chromatograms.peak_label = t.peak_label
                                     AND chromatograms.ms_file_label = s.ms_file_label)
                    """)
        ms1_to_compute = ms1_total
    else:
        ms1_to_compute = ms1_missing

    if recompute_ms2 and ms2_existing > 0:
        logger.info(f"Deleting {ms2_existing} existing MS2 chromatograms for recalculation...")
        con.execute("""
                    DELETE
                    FROM chromatograms
                    WHERE EXISTS(SELECT 1
                                 FROM targets t
                                          CROSS JOIN samples s
                                 WHERE s.use_for_optimization = TRUE
                                    OR s.use_for_processing = TRUE
                                     AND t.filterLine IS NOT NULL
                                     AND (t.peak_selection IS TRUE
                                        OR NOT EXISTS (SELECT 1
                                                       FROM targets t1
                                                       WHERE t1.peak_selection IS TRUE))
                                     AND chromatograms.peak_label = t.peak_label
                                     AND chromatograms.ms_file_label = s.ms_file_label)
                    """)
        ms2_to_compute = ms2_total
    else:
        ms2_to_compute = ms2_missing

    # Compute weights for progress reporting
    total_to_compute = ms1_to_compute + ms2_to_compute
    ms1_weight = ms1_to_compute / total_to_compute if process_ms1 else 0
    ms2_weight = ms2_to_compute / total_to_compute if process_ms2 else 0

    logger.info(f"Computing {ms1_to_compute} MS1 and {ms2_to_compute} MS2 chromatograms...")

    query_ms1 = """
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                WITH pairs_to_process AS (SELECT t.peak_label,
                                                 t.mz_mean,
                                                 t.mz_width,
                                                 CASE WHEN t.rt_unit = 'min' THEN t.rt_min * 60 ELSE t.rt_min END AS rt_min,
                                                 CASE WHEN t.rt_unit = 'min' THEN t.rt_max * 60 ELSE t.rt_max END AS rt_max,
                                                 s.ms_file_label
                                          FROM targets t
                                                   JOIN samples s
                                                        ON (CASE WHEN ? THEN s.use_for_optimization ELSE s.use_for_processing END) =
                                                           TRUE
                                          WHERE t.mz_mean IS NOT NULL
                                              AND t.mz_width IS NOT NULL
                                              AND (t.peak_selection IS TRUE
                                                 OR NOT EXISTS (SELECT 1
                                                                FROM targets t1
                                                                WHERE t1.peak_selection IS TRUE))
                                              AND (
                                                    ? -- recompute_ms1
                                                        OR NOT EXISTS (SELECT 1
                                                                       FROM chromatograms c
                                                                       WHERE c.peak_label = t.peak_label
                                                                         AND c.ms_file_label = s.ms_file_label)
                                                    )),
                     filtered AS (SELECT p.peak_label,
                                         p.ms_file_label,
                                         ROUND(ms1.scan_time, 3) AS scan_time,
                                         ROUND(ms1.intensity, 0) AS intensity
                                  FROM pairs_to_process p
                                           JOIN ms1_data ms1
                                                ON ms1.ms_file_label = p.ms_file_label
                                                    AND ms1.mz BETWEEN p.mz_mean - (p.mz_mean * p.mz_width / 1e6)
                                                       AND p.mz_mean + (p.mz_mean * p.mz_width / 1e6)
                                                    -- RT filter: only extract ±30s around target window
                                                    AND ms1.scan_time BETWEEN COALESCE(p.rt_min, 0) - 30 
                                                                          AND COALESCE(p.rt_max, 999999) + 30
                                  QUALIFY
                                        ROW_NUMBER() OVER (
                                          PARTITION BY p.peak_label, p.ms_file_label, ROUND(ms1.scan_time, 2)
                                            ORDER BY ms1.intensity DESC
                                        ) = 1),
                     agg AS (SELECT peak_label,
                                    ms_file_label,
                                    LIST(scan_time ORDER BY scan_time) AS scan_time,
                                    LIST(intensity ORDER BY scan_time) AS intensity
                             FROM filtered
                             GROUP BY peak_label, ms_file_label)
                SELECT peak_label, ms_file_label, scan_time, intensity, 'ms1' AS ms_type
                FROM agg;
                """

    query_ms2 = """
                INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                WITH pairs_to_process AS (SELECT t.peak_label,
                                                 t.filterLine,
                                                 CASE WHEN t.rt_unit = 'min' THEN t.rt_min * 60 ELSE t.rt_min END AS rt_min,
                                                 CASE WHEN t.rt_unit = 'min' THEN t.rt_max * 60 ELSE t.rt_max END AS rt_max,
                                                 s.ms_file_label
                                          FROM targets AS t
                                                   JOIN samples s
                                                        ON (CASE WHEN ? THEN s.use_for_optimization ELSE s.use_for_processing END) =
                                                           TRUE
                                          WHERE t.filterLine IS NOT NULL
                                              AND (t.peak_selection IS TRUE
                                                 OR NOT EXISTS (SELECT 1
                                                                FROM targets t1
                                                                WHERE t1.peak_selection IS TRUE))
                                              AND (
                                                    ? -- recompute_ms2
                                                        OR NOT EXISTS (SELECT 1
                                                                       FROM chromatograms c
                                                                       WHERE c.peak_label = t.peak_label
                                                                         AND c.ms_file_label = s.ms_file_label)
                                                    )),
                     pre AS (SELECT p.peak_label,
                                    p.ms_file_label,
                                    ROUND(ms2.scan_time, 3) AS scan_time,
                                    ROUND(ms2.intensity, 0) AS intensity
                             -- ms2.mz
                             FROM pairs_to_process p
                                      JOIN ms2_data ms2
                                           ON ms2.ms_file_label = p.ms_file_label
                                               AND ms2.filterLine = p.filterLine
                                               -- RT filter: only extract ±30s around target window
                                               AND ms2.scan_time BETWEEN COALESCE(p.rt_min, 0) - 30 
                                                                     AND COALESCE(p.rt_max, 999999) + 30),
                     grouped AS (SELECT peak_label,
                                        ms_file_label,
                                        scan_time,
                                        MAX(intensity) AS intensity -- max per time bin
                                 -- AVG(mz_mean)   AS mz_mean    -- stable within the bin
                                 FROM pre
                                 GROUP BY peak_label, ms_file_label, scan_time),
                     aggregated_chromatograms AS (SELECT peak_label,
                                                         ms_file_label,
                                                         LIST(scan_time ORDER BY scan_time) AS scan_time,
                                                         LIST(intensity ORDER BY scan_time) AS intensity
                                                  -- list(mz ORDER BY scan_time) AS mz
                                                  FROM grouped
                                                  GROUP BY peak_label, ms_file_label)
                SELECT peak_label, ms_file_label, scan_time, intensity, 'ms2' AS ms_type
                FROM aggregated_chromatograms;
                """

    # Shared variable to accumulate progress
    accumulated_progress = [0.0]
    stop_monitoring = [False]
    current_query_type = ['ms1']  # Track which query is running

    def monitor_progress():
        """Monitor progress of the current query"""
        while not stop_monitoring[0]:
            try:
                qp = con.query_progress()
                if qp != -1 and qp > 0:
                    # Total progress depends on which query is executing
                    if current_query_type[0] == 'ms1':
                        total_progress = qp * ms1_weight
                    else:  # ms2
                        total_progress = (ms1_weight * 100) + (qp * ms2_weight)

                    accumulated_progress[0] = total_progress
                    if set_progress:
                        set_progress(round(total_progress, 1))
                    logger.info(f"Progress: {total_progress:.1f}%")

                time.sleep(0.05)

            except (duckdb.InvalidInputException, duckdb.ConnectionException):
                break
            except Exception as e:
                logger.error(f"Progress monitoring error: {e}")
                break

    # Start monitoring
    if set_progress:
        progress_thread = Thread(target=monitor_progress, daemon=True)
        progress_thread.start()

    try:
        # Run MS1
        if process_ms1:
            logger.info("Processing MS1 chromatograms...")
            current_query_type[0] = 'ms1'
            con.execute(query_ms1, [for_optimization, recompute_ms1])
            accumulated_progress[0] = ms1_weight * 100
            if set_progress:
                set_progress(round(accumulated_progress[0], 1))

        # Run MS2
        if process_ms2:
            logger.info("Processing MS2 chromatograms...")
            current_query_type[0] = 'ms2'
            con.execute(query_ms2, [for_optimization, recompute_ms2])
            accumulated_progress[0] = 100.0
            if set_progress:
                set_progress(100.0)

        logger.info("Chromatograms computed and inserted into DuckDB.")

    finally:
        stop_monitoring[0] = True
        if set_progress:
            progress_thread.join(timeout=0.5)

    logger.info("Chromatograms computed and inserted into DuckDB.")




def compute_chromatograms_in_batches(wdir: str,
                                     # conn: duckdb.DuckDBPyConnection,
                                     use_for_optimization: bool,
                                     batch_size: int = None,
                                     checkpoint_every: int = 10,
                                     set_progress=None,
                                     recompute_ms1=False,
                                     recompute_ms2=False,
                                     n_cpus=None,
                                     ram=None,
                                     use_bookmarked: bool = False,
                                     ):

    logger.info(f"Computing chromatograms in batches. wDir: {wdir}")
    QUERY_CREATE_SCAN_LOOKUP = """
                               CREATE TABLE IF NOT EXISTS ms_file_scans AS
                               SELECT DISTINCT ms_file_label,
                                      scan_id,
                                      scan_time,
                                      'ms1' AS ms_type
                               FROM ms1_data
                                   UNION ALL
                               SELECT DISTINCT ms_file_label,
                                      scan_id,
                                      scan_time,
                                      'ms2' AS ms_type
                               FROM ms2_data
                               ORDER BY ms_file_label, scan_id, ms_type;

                               CREATE INDEX IF NOT EXISTS idx_ms_file_scans_file
                                   ON ms_file_scans (ms_file_label);

                               CREATE INDEX IF NOT EXISTS idx_ms_file_scans_file_scan
                                   ON ms_file_scans (ms_file_label, scan_id, ms_type);
                               """

    QUERY_CREATE_PENDING_PAIRS = """
                                 CREATE TABLE IF NOT EXISTS pending_pairs AS
                                 WITH target_filter AS (SELECT peak_label,
                                                               ms_type,
                                                               mz_mean,
                                                               mz_width,
                                                               filterLine,
                                                               rt_min,
                                                               rt_max,
                                                               rt_unit,
                                                               bookmark
                                                        FROM targets t
                                                        WHERE (
                                                            t.peak_selection IS TRUE
                                                                OR NOT EXISTS (SELECT 1
                                                                               FROM targets t1
                                                                               WHERE t1.peak_selection IS TRUE
                                                                                 AND t1.ms_type = t.ms_type)
                                                            )
                                                          AND (
                                                            CASE
                                                                WHEN ?
                                                                    THEN t.bookmark IS TRUE -- use_bookmarked = True → solo marcados
                                                                ELSE TRUE -- use_bookmarked = False → no filtra
                                                                END
                                                            )),
                                      sample_filter AS (SELECT ms_file_label
                                                        FROM samples
                                                        WHERE (CASE WHEN ? THEN use_for_optimization ELSE use_for_processing END) = TRUE),
                                      existing_pairs AS (SELECT DISTINCT peak_label,
                                                                         ms_file_label,
                                                                         ms_type
                                                         FROM chromatograms),
                                      all_possible_pairs AS (SELECT t.peak_label,
                                                                    s.ms_file_label,
                                                                    t.ms_type,
                                                                    t.mz_mean,
                                                                    t.mz_width,
                                                                    t.filterLine,
                                                                    CASE WHEN t.rt_unit = 'min' THEN t.rt_min * 60 ELSE t.rt_min END AS rt_min,
                                                                    CASE WHEN t.rt_unit = 'min' THEN t.rt_max * 60 ELSE t.rt_max END AS rt_max
                                                             FROM target_filter t
                                                                      CROSS JOIN sample_filter s),
                                      pending AS (SELECT a.peak_label,
                                                         a.ms_file_label,
                                                         a.ms_type,
                                                         a.mz_mean,
                                                         a.mz_width,
                                                         a.filterLine,
                                                         a.rt_min,
                                                         a.rt_max,
                                                         ROW_NUMBER() OVER () AS pair_id
                                                  FROM all_possible_pairs a
                                                           LEFT JOIN existing_pairs e
                                                                     ON a.peak_label = e.peak_label
                                                                         AND a.ms_file_label = e.ms_file_label
                                                                         AND a.ms_type = e.ms_type
                                                  WHERE e.peak_label IS NULL)
                                 SELECT pair_id,
                                        peak_label,
                                        ms_file_label,
                                        ms_type,
                                        mz_mean,
                                        mz_width,
                                        filterLine,
                                        rt_min,
                                        rt_max
                                 FROM pending
                                 ORDER BY pair_id;
                                 """

    QUERY_PROCESS_BATCH_MS1 = """
                              INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, mz_arr, ms_type)
                              WITH batch_pairs AS (SELECT peak_label, ms_file_label, ms_type, mz_mean, mz_width, rt_min, rt_max
                                                   FROM pending_pairs
                                                   WHERE ms_type = 'ms1'
                                                     AND pair_id BETWEEN ? AND ?),
                                   -- Step 1: Find max intensity and corresponding mz per scan
                                   matched_with_mz AS (
                                       SELECT bp.peak_label,
                                              bp.ms_file_label,
                                              ms1.scan_id,
                                              ms1.intensity,
                                              ms1.mz,
                                              ROW_NUMBER() OVER (
                                                  PARTITION BY bp.peak_label, bp.ms_file_label, ms1.scan_id
                                                  ORDER BY ms1.intensity DESC
                                              ) AS rn
                                       FROM batch_pairs bp
                                       JOIN ms1_data ms1
                                            ON ms1.ms_file_label = bp.ms_file_label
                                                AND ms1.mz BETWEEN
                                                   bp.mz_mean - (bp.mz_mean * bp.mz_width / 1e6)
                                                   AND
                                                   bp.mz_mean + (bp.mz_mean * bp.mz_width / 1e6)
                                   ),
                                   matched_intensities AS (
                                       SELECT peak_label, ms_file_label, scan_id, intensity, mz
                                       FROM matched_with_mz
                                       WHERE rn = 1
                                   ),
                                   -- Step 2: Expand to scans within RT window (±30s margin)
                                   all_scans_needed AS (SELECT DISTINCT bp.peak_label,
                                                                        bp.ms_file_label,
                                                                        s.scan_id,
                                                                        s.scan_time
                                                        FROM batch_pairs bp
                                                                 JOIN ms_file_scans s ON s.ms_file_label = bp.ms_file_label
                                                                     -- RT filter: only include scans ±30s around target window
                                                                     AND s.scan_time BETWEEN COALESCE(bp.rt_min, 0) - 30 
                                                                                         AND COALESCE(bp.rt_max, 999999) + 30),
                                   -- Step 3: LEFT JOIN (both tables are small)
                                   complete_data AS (SELECT a.peak_label,
                                                            a.ms_file_label,
                                                            a.scan_time,
                                                            a.scan_id,
                                                            COALESCE(ROUND(m.intensity, 0), 1) AS intensity,
                                                            m.mz AS mz_val
                                                     FROM all_scans_needed a
                                                              LEFT JOIN matched_intensities m
                                                                        ON a.peak_label = m.peak_label
                                                                            AND a.ms_file_label = m.ms_file_label
                                                                            AND a.scan_id = m.scan_id),
                                   agg AS (SELECT peak_label,
                                                  ms_file_label,
                                                  LIST(scan_time ORDER BY scan_time) AS scan_time,
                                                  LIST(intensity ORDER BY scan_time) AS intensity,
                                                  LIST(mz_val ORDER BY scan_time) AS mz_arr
                                           FROM complete_data
                                           GROUP BY peak_label, ms_file_label)
                              SELECT peak_label, ms_file_label, scan_time, intensity, mz_arr, 'ms1' AS ms_type
                              FROM agg;
                              """

    QUERY_PROCESS_BATCH_MS2 = """
                              INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                              WITH batch_pairs AS (SELECT peak_label, ms_file_label, ms_type, filterLine, rt_min, rt_max
                                                   FROM pending_pairs
                                                   WHERE ms_type = 'ms2'
                                                     AND pair_id BETWEEN ? AND ?),
                                   -- Step 1: Find intensities (only rows with signal)
                                   matched_filterline AS (SELECT bp.peak_label,
                                                      bp.ms_file_label,
                                                      ms2.scan_id,
                                                      ms2.intensity
                                               FROM batch_pairs bp
                                                        JOIN ms2_data ms2
                                                             ON ms2.ms_file_label = bp.ms_file_label
                                                                 AND ms2.filterLine = bp.filterLine),
                                  -- Step 2: Expand to scans within RT window (±30s margin)
                                   all_scans_needed AS (SELECT DISTINCT bp.peak_label,
                                                                        bp.ms_file_label,
                                                                        s.scan_id,
                                                                        s.scan_time
                                                        FROM batch_pairs bp
                                                                 JOIN ms_file_scans s ON s.ms_file_label = bp.ms_file_label
                                                                     -- RT filter: only include scans ±30s around target window
                                                                     AND s.scan_time BETWEEN COALESCE(bp.rt_min, 0) - 30 
                                                                                         AND COALESCE(bp.rt_max, 999999) + 30),
                                  -- Step 3: LEFT JOIN (both tables are small)
                                   complete_data AS (SELECT a.peak_label,
                                                            a.ms_file_label,
                                                            a.scan_time,
                                                            a.scan_id,
                                                            a.scan_time,
                                                            COALESCE(ROUND(m.intensity, 0), 1) AS intensity
                                                     FROM all_scans_needed a
                                                              LEFT JOIN matched_filterline m
                                                                        ON a.peak_label = m.peak_label
                                                                            AND a.ms_file_label = m.ms_file_label
                                                                            AND a.scan_id = m.scan_id),
                                   -- Step 2: Aggregate after expanding scans
                                   agg AS (SELECT peak_label,
                                                  ms_file_label,
                                                  LIST(scan_time ORDER BY scan_time) AS scan_time,
                                                  LIST(intensity ORDER BY scan_time) AS intensity
                                           FROM complete_data
                                           GROUP BY peak_label, ms_file_label)
                              SELECT peak_label,
                                     ms_file_label,
                                     scan_time,
                                     intensity,
                                     'ms2' AS ms_type
                              FROM agg;
                              """

    if recompute_ms1:
        logger.info("Deleting existing MS1 chromatograms for recalculation...")
        with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
            con.execute("DELETE FROM chromatograms WHERE ms_type = 'ms1'")
    if recompute_ms2:
        logger.info("Deleting existing MS2 chromatograms for recalculation...")
        with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
            con.execute("DELETE FROM chromatograms WHERE ms_type = 'ms2'")

    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        # Ensure clean database state before processing (clears any accumulated WAL)
        logger.info("Running CHECKPOINT to ensure clean database state...")
        conn.execute("CHECKPOINT")
        
        conn.execute("DROP TABLE IF EXISTS pending_pairs")
        try:
            count = conn.execute("SELECT COUNT(*) FROM ms_file_scans").fetchone()[0]
            logger.info(f"Lookup table exists ({count:,} entries)")
        except:
            logger.warning("Lookup table does not exist. Creating...")

            start = time.perf_counter()
            conn.execute(QUERY_CREATE_SCAN_LOOKUP)
            elapsed = time.perf_counter() - start

            result = conn.execute("""
                                  SELECT COUNT(*)                      as total_entries,
                                         COUNT(DISTINCT ms_file_label) as total_files,
                                         AVG(scans_per_file)           as avg_scans
                                  FROM (SELECT ms_file_label, COUNT(*) as scans_per_file
                                        FROM ms_file_scans
                                        GROUP BY ms_file_label)
                                  """).fetchone()
            logger.info(f"  Total entries: {result[0]:,}")
            logger.info(f"  Total MS files: {result[1]:,}")
            logger.info(f"  Average scans per file: {result[2]:.0f}")
            logger.info(f"  Time elapsed: {elapsed:.2f}s")

    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        # Ensure clean database state before processing (clears any accumulated WAL)
        logger.info("Running CHECKPOINT to ensure clean database state...")
        conn.execute("CHECKPOINT")
        
        logger.info("Getting pending pairs...")
        start_time = time.time()
        conn.execute(QUERY_CREATE_PENDING_PAIRS, [use_bookmarked, use_for_optimization])
        rows = conn.execute("""
                            SELECT ms_type,
                                   COUNT(*)     AS total,
                                   MIN(pair_id) AS min_id,
                                   MAX(pair_id) AS max_id
                            FROM pending_pairs
                            GROUP BY ms_type
                            ORDER BY ms_type
                            """).fetchall()
        elapsed = time.time() - start_time

        if not rows:
            logger.info(f"No pending pairs ({elapsed:.2f}s)")
            conn.execute("DROP TABLE IF EXISTS pending_pairs")
            return {
                'total_pairs': 0,
                'processed': 0,
                'failed': 0,
                'batches': 0
            }

        global_total_pairs = sum(r[1] for r in rows)
        files_per_type = {}
        if set_progress:
            files_per_type = dict(conn.execute("""
                                               SELECT ms_type,
                                                      COUNT(DISTINCT ms_file_label) AS total_files
                                               FROM pending_pairs
                                               GROUP BY ms_type
                                               """).fetchall())

        _send_progress(
            set_progress,
            0,
            stage="Chromatograms",
            detail=f"Pending pairs: {global_total_pairs:,}",
        )



        logger.info(f"{global_total_pairs:,} pending pairs ({elapsed:.2f}s)")
        
        # Auto-calculate optimal batch size if not explicitly provided
        if batch_size is None:
            ram_gb = ram if ram else 16  # Default to 16GB if not specified
            cpus = n_cpus if n_cpus else 4  # Default to 4 if not specified
            batch_size = calculate_optimal_batch_size(ram_gb, global_total_pairs, cpus)
            logger.info(f"Auto-calculated batch size: {batch_size} (based on {ram_gb}GB RAM, {cpus} CPUs, {global_total_pairs:,} pairs)")
        else:
            logger.info(f"Using specified batch size: {batch_size}")

        global_processed = 0  # accumulated counter
        global_stats: dict[str, dict] = {}

        for ms_type, total_pairs_type, min_id, max_id in rows:
            logger.info(f"--- Processing {ms_type} ---")
            logger.info(f"Pending pairs: {total_pairs_type:,} (pair_id {min_id}-{max_id})")

            if total_pairs_type == 0 or min_id is None or max_id is None:
                global_stats[ms_type] = {
                    'total_pairs': 0,
                    'processed': 0,
                    'failed': 0,
                    'batches': 0,
                }
                continue

            processed = 0
            failed = 0
            batches = 0
            processed_files: set[str] = set()

            current_id = min_id
            batch_num = 1
            total_batches = (total_pairs_type + batch_size - 1) // batch_size
            batches_since_checkpoint = 0
            total_files_type = files_per_type.get(ms_type, 0)

            with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
                # Process batches in a single connection; checkpoint periodically to avoid WAL stalls
                conn.execute("BEGIN TRANSACTION")

                while current_id <= max_id:
                    batch_count = 0
                    start_id = current_id
                    end_id = current_id + batch_size - 1

                    try:
                        batch_count = conn.execute("""
                                                   SELECT COUNT(*)
                                                   FROM pending_pairs
                                                   WHERE ms_type = ?
                                                     AND pair_id BETWEEN ? AND ?
                                                   """, [ms_type, start_id, end_id]).fetchone()[0]

                        if batch_count == 0:
                            current_id += batch_size
                            continue



                        batch_start = time.time()

                        if ms_type == 'ms1':
                            conn.execute(QUERY_PROCESS_BATCH_MS1, [start_id, end_id])
                        elif ms_type == 'ms2':
                            conn.execute(QUERY_PROCESS_BATCH_MS2, [start_id, end_id])

                        if set_progress:
                            batch_files = conn.execute("""
                                                       SELECT DISTINCT ms_file_label
                                                       FROM pending_pairs
                                                       WHERE ms_type = ?
                                                         AND pair_id BETWEEN ? AND ?
                                                       """, [ms_type, start_id, end_id]).fetchall()
                            processed_files.update(
                                row[0] for row in batch_files if row and row[0] is not None
                            )

                        batch_elapsed = time.time() - batch_start
                        processed += batch_count
                        batches += 1
                        batches_since_checkpoint += 1

                        logger.info(f"Batch {batch_num:>4}/{total_batches} | "
                                    f"IDs {start_id:>6}-{end_id:>6} | "
                                    f"{batch_count:>3} pairs | "
                                    f"Batch time: {batch_elapsed:>5.2f}s | "
                                    f"Progress {processed:>6,}/{total_pairs_type:,}")
                        log_line = (f"Batch {batch_num}/{total_batches} | "
                                    f"Progress {processed:,}/{total_pairs_type:,} | "
                                    f"Time/batch {batch_elapsed:0.2f}s")


                        if checkpoint_every and batches_since_checkpoint >= checkpoint_every:
                            conn.execute("COMMIT")
                            conn.execute("CHECKPOINT")
                            conn.execute("BEGIN TRANSACTION")
                            batches_since_checkpoint = 0

                        batch_num += 1

                    except Exception as e:
                        batch_elapsed = time.time() - batch_start if 'batch_start' in locals() else 0
                        failed += batch_count

                        logger.error(f"Error processing batch: {batch_elapsed:>5.2f}s | Error: {str(e)[:80]}")


                        with open(f'failed_batches_{ms_type}.log', 'a') as f:
                            f.write(f"\n{'=' * 60}\n")
                            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"ms_type: {ms_type}\n")
                            f.write(f"Batch {batch_num}/{total_batches}\n")
                            f.write(f"IDs: {start_id}-{end_id}\n")
                            f.write(f"Error: {str(e)}\n")
                            f.write(f"{'=' * 60}\n")

                        try:
                            conn.execute("ROLLBACK")
                            conn.execute("BEGIN TRANSACTION")
                            batches_since_checkpoint = 0
                        except Exception:
                            pass

                    finally:
                        # Update global progress even if the batch failed
                        if batch_count > 0:
                            global_processed += batch_count
                            progress_pct_global = (global_processed / global_total_pairs) * 100
                            files_done = len(processed_files) if set_progress else 0
                            detail_text = (
                                f"{log_line}"
                                if log_line else
                                f"{ms_type.upper()} batch {batch_num}/{total_batches} | "
                                f"Pairs {processed:,}/{total_pairs_type:,}"
                            )
                            _send_progress(
                                set_progress,
                                round(progress_pct_global, 1),
                                stage="Chromatograms",
                                detail=detail_text,
                            )

                    current_id += batch_size

                # Final checkpoint/commit for this ms_type
                conn.execute("COMMIT")
                conn.execute("CHECKPOINT")

    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        conn.execute("DROP TABLE IF EXISTS ms_file_scans")
        conn.execute("DROP TABLE IF EXISTS pending_pairs")


def populate_full_range_downsampled_chromatograms(wdir: str,
                                                  n_out: int = FULL_RANGE_DOWNSAMPLE_POINTS,
                                                  batch_size: int = FULL_RANGE_DOWNSAMPLE_BATCH,
                                                  set_progress=None,
                                                  n_cpus=None,
                                                  ram=None):
    if _lttbc is None:
        logger.warning("Full-range downsampling skipped: 'lttbc' is not available.")
        _send_progress(set_progress, 100, stage="Downsampling", detail="lttbc not available")
        return

    logger.info("Preparing full-range downsampled chromatograms (MS1 only)...")
    _send_progress(
        set_progress,
        0,
        stage="Downsampling",
        detail="Preparing full-range downsampled chromatograms",
    )

    query_create_scan_lookup = """
        CREATE TABLE IF NOT EXISTS ms_file_scans AS
        SELECT DISTINCT ms_file_label,
               scan_id,
               scan_time,
               'ms1' AS ms_type
        FROM ms1_data
            UNION ALL
        SELECT DISTINCT ms_file_label,
               scan_id,
               scan_time,
               'ms2' AS ms_type
        FROM ms2_data
        ORDER BY ms_file_label, scan_id, ms_type;

        CREATE INDEX IF NOT EXISTS idx_ms_file_scans_file
            ON ms_file_scans (ms_file_label);

        CREATE INDEX IF NOT EXISTS idx_ms_file_scans_file_scan
            ON ms_file_scans (ms_file_label, scan_id, ms_type);
        """

    query_create_pending = """
        CREATE TABLE IF NOT EXISTS pending_full_ds_pairs AS
        SELECT
            ROW_NUMBER() OVER () AS pair_id,
            c.peak_label,
            c.ms_file_label,
            t.mz_mean,
            t.mz_width
        FROM chromatograms c
        JOIN targets t ON c.peak_label = t.peak_label
        WHERE c.ms_type = 'ms1'
          AND c.scan_time_full_ds IS NULL
          AND t.mz_mean IS NOT NULL
          AND t.mz_width IS NOT NULL
        ORDER BY c.ms_file_label, c.peak_label;
        """

    query_batch_full_range = """
        WITH batch_pairs AS (
            SELECT peak_label, ms_file_label, mz_mean, mz_width
            FROM pending_full_ds_pairs
            WHERE pair_id BETWEEN ? AND ?
        ),
        matched_with_mz AS (
            SELECT
                bp.peak_label,
                bp.ms_file_label,
                ms1.scan_id,
                ms1.intensity,
                ROW_NUMBER() OVER (
                    PARTITION BY bp.peak_label, bp.ms_file_label, ms1.scan_id
                    ORDER BY ms1.intensity DESC
                ) AS rn
            FROM batch_pairs bp
            JOIN ms1_data ms1
                ON ms1.ms_file_label = bp.ms_file_label
                AND ms1.mz BETWEEN
                    bp.mz_mean - (bp.mz_mean * bp.mz_width / 1e6)
                    AND bp.mz_mean + (bp.mz_mean * bp.mz_width / 1e6)
        ),
        matched_intensities AS (
            SELECT peak_label, ms_file_label, scan_id, intensity
            FROM matched_with_mz
            WHERE rn = 1
        ),
        all_scans AS (
            SELECT
                bp.peak_label,
                bp.ms_file_label,
                s.scan_id,
                s.scan_time
            FROM batch_pairs bp
            JOIN ms_file_scans s
                ON s.ms_file_label = bp.ms_file_label
                AND s.ms_type = 'ms1'
        ),
        complete_data AS (
            SELECT
                a.peak_label,
                a.ms_file_label,
                a.scan_time,
                COALESCE(ROUND(m.intensity, 0), 1) AS intensity
            FROM all_scans a
            LEFT JOIN matched_intensities m
                ON a.peak_label = m.peak_label
                AND a.ms_file_label = m.ms_file_label
                AND a.scan_id = m.scan_id
        ),
        agg AS (
            SELECT
                peak_label,
                ms_file_label,
                LIST(scan_time ORDER BY scan_time) AS scan_time,
                LIST(intensity ORDER BY scan_time) AS intensity
            FROM complete_data
            GROUP BY peak_label, ms_file_label
        )
        SELECT peak_label, ms_file_label, scan_time, intensity
        FROM agg;
        """

    created_lookup = False
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        try:
            conn.execute("SELECT COUNT(*) FROM ms_file_scans").fetchone()
        except Exception:
            logger.info("Creating scan lookup table for full-range downsampling...")
            conn.execute(query_create_scan_lookup)
            created_lookup = True

        conn.execute("DROP TABLE IF EXISTS pending_full_ds_pairs")
        conn.execute(query_create_pending)

        total_pairs = conn.execute(
            "SELECT COUNT(*) FROM pending_full_ds_pairs"
        ).fetchone()[0]

        if not total_pairs:
            logger.info("No full-range downsampled chromatograms to compute.")
            conn.execute("DROP TABLE IF EXISTS pending_full_ds_pairs")
            if created_lookup:
                conn.execute("DROP TABLE IF EXISTS ms_file_scans")
            _send_progress(set_progress, 100, stage="Downsampling", detail="No pending pairs")
            return

        total_batches = (total_pairs + batch_size - 1) // batch_size
        logger.info(f"Downsampling {total_pairs:,} MS1 chromatograms in {total_batches} batches...")

        for batch_idx in range(total_batches):
            start_id = batch_idx * batch_size + 1
            end_id = min((batch_idx + 1) * batch_size, total_pairs)

            rows = conn.execute(query_batch_full_range, [start_id, end_id]).fetchall()
            updates = []
            for peak_label, ms_file_label, scan_time, intensity in rows:
                if scan_time is None or intensity is None:
                    continue
                smoothed = _apply_savgol_smoothing(intensity)
                down_x, down_y = _apply_lttb_downsampling(scan_time, smoothed, n_out=n_out)
                if len(down_x) > n_out and len(scan_time) > n_out:
                    logger.warning(
                        "Downsampling failed for %s/%s (points=%d, n_out=%d)",
                        peak_label,
                        ms_file_label,
                        len(scan_time),
                        n_out,
                    )
                    continue
                updates.append(
                    (list(down_x), list(down_y), peak_label, ms_file_label)
                )

            if updates:
                conn.executemany(
                    """
                    UPDATE chromatograms
                    SET scan_time_full_ds = ?, intensity_full_ds = ?
                    WHERE peak_label = ? AND ms_file_label = ?
                    """,
                    updates,
                )

            progress_pct = round((end_id / total_pairs) * 100, 1)
            _send_progress(
                set_progress,
                progress_pct,
                stage="Downsampling",
                detail=f"Batch {batch_idx + 1}/{total_batches}",
            )

        conn.execute("DROP TABLE IF EXISTS pending_full_ds_pairs")
        if created_lookup:
            conn.execute("DROP TABLE IF EXISTS ms_file_scans")

    _send_progress(set_progress, 100, stage="Downsampling", detail="Done")
    logger.info("Full-range downsampled chromatograms computed.")


def populate_full_range_downsampled_chromatograms_for_target(wdir: str | None,
                                                             peak_label: str,
                                                             n_out: int = FULL_RANGE_DOWNSAMPLE_POINTS,
                                                             n_cpus=None,
                                                             ram=None,
                                                             conn: duckdb.DuckDBPyConnection | None = None) -> bool:
    if _lttbc is None:
        logger.warning("On-demand downsampling skipped: 'lttbc' is not available.")
        return False

    query_create_scan_lookup = """
        CREATE TABLE IF NOT EXISTS ms_file_scans AS
        SELECT DISTINCT ms_file_label,
               scan_id,
               scan_time,
               'ms1' AS ms_type
        FROM ms1_data
            UNION ALL
        SELECT DISTINCT ms_file_label,
               scan_id,
               scan_time,
               'ms2' AS ms_type
        FROM ms2_data
        ORDER BY ms_file_label, scan_id, ms_type;

        CREATE INDEX IF NOT EXISTS idx_ms_file_scans_file
            ON ms_file_scans (ms_file_label);

        CREATE INDEX IF NOT EXISTS idx_ms_file_scans_file_scan
            ON ms_file_scans (ms_file_label, scan_id, ms_type);
        """

    query_full_range = """
        WITH target AS (
            SELECT ?::DOUBLE AS mz_mean, ?::DOUBLE AS mz_width
        ),
        samples AS (
            SELECT DISTINCT ms_file_label
            FROM chromatograms
            WHERE peak_label = ?
              AND ms_type = 'ms1'
              AND scan_time_full_ds IS NULL
        ),
        matched_with_mz AS (
            SELECT
                s.ms_file_label,
                ms1.scan_id,
                ms1.intensity,
                ROW_NUMBER() OVER (
                    PARTITION BY s.ms_file_label, ms1.scan_id
                    ORDER BY ms1.intensity DESC
                ) AS rn
            FROM samples s
            JOIN ms1_data ms1
                ON ms1.ms_file_label = s.ms_file_label
                AND ms1.mz BETWEEN
                    (SELECT mz_mean FROM target) - ((SELECT mz_mean FROM target) * (SELECT mz_width FROM target) / 1e6)
                    AND (SELECT mz_mean FROM target) + ((SELECT mz_mean FROM target) * (SELECT mz_width FROM target) / 1e6)
        ),
        matched_intensities AS (
            SELECT ms_file_label, scan_id, intensity
            FROM matched_with_mz
            WHERE rn = 1
        ),
        all_scans AS (
            SELECT
                s.ms_file_label,
                sc.scan_id,
                sc.scan_time
            FROM samples s
            JOIN ms_file_scans sc
                ON sc.ms_file_label = s.ms_file_label
                AND sc.ms_type = 'ms1'
        ),
        complete_data AS (
            SELECT
                a.ms_file_label,
                a.scan_time,
                COALESCE(ROUND(m.intensity, 0), 1) AS intensity
            FROM all_scans a
            LEFT JOIN matched_intensities m
                ON a.ms_file_label = m.ms_file_label
                AND a.scan_id = m.scan_id
        ),
        agg AS (
            SELECT
                ms_file_label,
                LIST(scan_time ORDER BY scan_time) AS scan_time,
                LIST(intensity ORDER BY scan_time) AS intensity
            FROM complete_data
            GROUP BY ms_file_label
        )
        SELECT ms_file_label, scan_time, intensity
        FROM agg;
        """

    connection_ctx = None
    if conn is None:
        if wdir is None:
            return False
        connection_ctx = duckdb_connection(wdir, n_cpus=n_cpus, ram=ram)
        conn = connection_ctx.__enter__()
        if conn is None:
            return False
    try:
        target = conn.execute(
            """
            SELECT mz_mean, mz_width
            FROM targets
            WHERE peak_label = ?
              AND mz_mean IS NOT NULL
              AND mz_width IS NOT NULL
            """,
            [peak_label],
        ).fetchone()
        if not target:
            return False

        missing = conn.execute(
            """
            SELECT COUNT(*)
            FROM chromatograms
            WHERE peak_label = ?
              AND ms_type = 'ms1'
              AND scan_time_full_ds IS NULL
            """,
            [peak_label],
        ).fetchone()[0]
        if not missing:
            return True

        created_lookup = False
        try:
            conn.execute("SELECT COUNT(*) FROM ms_file_scans").fetchone()
        except Exception:
            conn.execute(query_create_scan_lookup)
            created_lookup = True

        logger.info("On-demand downsampling for target %s (%d chromatograms)", peak_label, missing)
        rows = conn.execute(query_full_range, [target[0], target[1], peak_label]).fetchall()
        updates = []
        for ms_file_label, scan_time, intensity in rows:
            if scan_time is None or intensity is None:
                continue
            smoothed = _apply_savgol_smoothing(intensity)
            down_x, down_y = _apply_lttb_downsampling(scan_time, smoothed, n_out=n_out)
            updates.append((list(down_x), list(down_y), peak_label, ms_file_label))

        if updates:
            conn.executemany(
                """
                UPDATE chromatograms
                SET scan_time_full_ds = ?, intensity_full_ds = ?
                WHERE peak_label = ? AND ms_file_label = ?
                """,
                updates,
            )

        if created_lookup:
            conn.execute("DROP TABLE IF EXISTS ms_file_scans")
    finally:
        if connection_ctx is not None:
            connection_ctx.__exit__(None, None, None)

    return True


def compute_results_in_batches(wdir: str,
                               use_bookmarked: bool = False,
                               recompute: bool = False,
                               batch_size: int = None,
                               checkpoint_every: int = 20,
                               set_progress=None,
                               n_cpus=None,
                               ram=None):
    """
    Compute results with efficient macros.
    include_arrays=False: numeric metrics only (FAST)
    include_arrays=True: include scan_time and intensity arrays (SLOWER)
    """


    # OPTIMIZED macro using list functions - avoids UNNEST memory explosion
    # This approach filters arrays directly without creating intermediate rows,
    # reducing memory usage from 15GB+ to under 4GB for large chromatograms (37K+ points)
    QUERY_CREATE_HELPERS = """
        CREATE OR REPLACE MACRO compute_chromatogram_metrics(scan_times, intensities, rt_min, rt_max) AS TABLE (
            WITH 
            -- Filter arrays using list operations (no UNNEST = no memory explosion)
            filtered AS (
                SELECT list_filter(
                    list_transform(
                        range(1, len(scan_times) + 1),
                        i -> struct_pack(t := list_extract(scan_times, i), i := list_extract(intensities, i))
                    ),
                    p -> p.t >= rt_min AND p.t <= rt_max
                ) AS pairs
            ),
            -- Extract arrays from filtered pairs
            arrays AS (
                SELECT
                    list_transform(pairs, p -> p.t) AS scan_time_arr,
                    list_transform(pairs, p -> p.i) AS intensity_arr
                FROM filtered
            ),
            -- Compute trapezoid integration for peak_area
            -- Trapezoid rule: Area = Σ (y[i] + y[i+1])/2 × (x[i+1] - x[i])
            trapezoid AS (
                SELECT
                    scan_time_arr,
                    intensity_arr,
                    -- Build list of trapezoid areas for each segment
                    list_transform(
                        range(1, len(scan_time_arr)),
                        i -> (
                            (list_extract(intensity_arr, i) + list_extract(intensity_arr, i + 1)) / 2.0
                            * (list_extract(scan_time_arr, i + 1) - list_extract(scan_time_arr, i))
                        )
                    ) AS trapezoid_areas
                FROM arrays
                -- Removed: WHERE len(intensity_arr) > 0
            ),
            -- Compute all metrics from arrays
            metrics AS (
                SELECT
                    len(intensity_arr) AS peak_n_datapoints,
                    -- Trapezoid integration
                    ROUND(COALESCE(list_sum(trapezoid_areas), 0), 0) AS peak_area,
                    ROUND(COALESCE(list_max(intensity_arr), 0), 0) AS peak_max,
                    ROUND(COALESCE(list_min(intensity_arr), 0), 0) AS peak_min,
                    ROUND(COALESCE(list_avg(intensity_arr), 0), 0) AS peak_mean,
                    -- Median: handle empty list case
                    ROUND(COALESCE(list_sort(intensity_arr)[CAST(len(intensity_arr) / 2 + 1 AS BIGINT)], 0), 0) AS peak_median,
                    -- RT of max intensity
                    scan_time_arr[CAST(list_position(intensity_arr, list_max(intensity_arr)) AS BIGINT)] AS peak_rt_of_max,
                    -- Index of max for top3 calculation
                    CAST(list_position(intensity_arr, list_max(intensity_arr)) AS BIGINT) AS max_idx,
                    scan_time_arr,
                    intensity_arr
                FROM trapezoid
            )
            -- Final output with peak_area_top3
            SELECT
                peak_area,
                ROUND(
                    COALESCE(intensity_arr[max_idx - 1], 0) +
                    list_max(intensity_arr) +
                    COALESCE(intensity_arr[max_idx + 1], 0),
                    0
                ) AS peak_area_top3,
                peak_max,
                peak_min,
                peak_mean,
                peak_rt_of_max,
                peak_median,
                peak_n_datapoints,
                scan_time_arr AS scan_time_list,
                intensity_arr AS intensity_list
            FROM metrics
        );
    """

    QUERY_CREATE_PENDING_PAIRS = """
                                 CREATE TABLE IF NOT EXISTS pending_result_pairs AS
                                 WITH pairs_to_process AS (
                                 SELECT c.peak_label,
                                                                  c.ms_file_label
                                                           FROM chromatograms c
                                                                    JOIN targets t ON c.peak_label = t.peak_label
                                                           WHERE CASE
                WHEN ? THEN c.peak_label IN (
                    SELECT peak_label FROM targets WHERE bookmark = TRUE
                )
                                                                     ELSE TRUE
                                                               END
                                                             AND (
                                                               ? OR NOT EXISTS (
                                                               SELECT 1 FROM results r
                                                                                WHERE r.peak_label = c.peak_label
                    AND r.ms_file_label = c.ms_file_label
                )
            )
        )
                                 SELECT peak_label,
                                        ms_file_label,
                                        ROW_NUMBER() OVER () AS pair_id
                                 FROM pairs_to_process
                                 ORDER BY peak_label, ms_file_label;
                                 """

    # Direct query - no intermediate CTE, streaming insert
    # When rt_align_enabled=TRUE, applies per-sample RT shifts to the integration window
    QUERY_PROCESS_BATCH = """
        INSERT INTO results (
            peak_label,
            ms_file_label,
            peak_area,
            peak_area_top3,
            peak_max,
            peak_min,
            peak_mean,
            peak_rt_of_max,
            peak_mz_of_max,
            peak_median,
            peak_n_datapoints,
            rt_aligned,
            rt_shift,
            scan_time,
            intensity
        )
        WITH batch_pairs AS (
            SELECT peak_label, ms_file_label
            FROM pending_result_pairs
            WHERE pair_id BETWEEN ? AND ?
        ),
        -- Join chromatograms with targets and calculate per-sample shift
        paired_data AS (
            SELECT 
                c.peak_label,
                c.ms_file_label,
                c.scan_time,
                c.intensity,
                c.mz_arr,
                t.rt_min,
                t.rt_max,
                COALESCE(t.rt_align_enabled, FALSE) AS rt_align_enabled,
                -- Extract per-sample shift from JSON, default to 0.0 if not found
                COALESCE(
                    CASE 
                        WHEN t.rt_align_enabled AND t.rt_align_shifts IS NOT NULL 
                        THEN TRY_CAST(json_extract(t.rt_align_shifts, '$."' || c.ms_file_label || '"') AS DOUBLE)
                        ELSE NULL
                    END,
                    0.0
                ) AS sample_shift
            FROM chromatograms c
            JOIN batch_pairs bp ON c.peak_label = bp.peak_label AND c.ms_file_label = bp.ms_file_label
            JOIN targets t ON c.peak_label = t.peak_label
        )
        SELECT 
            pd.peak_label,
            pd.ms_file_label,
            m.peak_area,
            m.peak_area_top3,
            m.peak_max,
            m.peak_min,
            m.peak_mean,
            m.peak_rt_of_max,
            -- peak_mz_of_max: Get mz at same index as peak max intensity
            CASE 
                WHEN pd.mz_arr IS NOT NULL AND list_position(pd.intensity, m.peak_max) IS NOT NULL
                THEN pd.mz_arr[CAST(list_position(pd.intensity, m.peak_max) AS BIGINT)]
                ELSE NULL
            END AS peak_mz_of_max,
            m.peak_median,
            m.peak_n_datapoints,
            pd.rt_align_enabled AS rt_aligned,
            pd.sample_shift AS rt_shift,
            m.scan_time_list,
            m.intensity_list
        FROM paired_data pd
        CROSS JOIN LATERAL compute_chromatogram_metrics(
            pd.scan_time, 
            pd.intensity, 
            -- Apply inverse shift: if peak was shifted by +X for visualization,
            -- we need to look at [rt_min - X, rt_max - X] in original data
            pd.rt_min - pd.sample_shift,
            pd.rt_max - pd.sample_shift
        ) AS m;
                          """



    if recompute:
        logger.info("Deleting existing results for recalculation...")
        with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
            con.execute("DELETE FROM results")

    # Create helper macro
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        logger.info("Creating helper macro...")
        conn.execute(QUERY_CREATE_HELPERS)

    # Create pending pairs table
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        # Ensure clean database state before processing (clears any accumulated WAL)
        logger.info("Running CHECKPOINT to ensure clean database state...")
        conn.execute("CHECKPOINT")
        
        conn.execute("DROP TABLE IF EXISTS pending_result_pairs")

        logger.info("Getting pending pairs...")
        start_time = time.time()
        conn.execute(QUERY_CREATE_PENDING_PAIRS, [use_bookmarked, recompute])

        total_pairs = conn.execute("""
            SELECT COUNT(*) AS total,
                                          MIN(pair_id) AS min_id,
                                          MAX(pair_id) AS max_id
                                   FROM pending_result_pairs
                                   """).fetchone()

        elapsed = time.time() - start_time

        if total_pairs[0] == 0 or total_pairs[1] is None:
            logger.info(f"No pending pairs ({elapsed:.2f}s)")
            conn.execute("DROP TABLE IF EXISTS pending_result_pairs")
            return {'total_pairs': 0, 'processed': 0, 'failed': 0, 'batches': 0}

        total_count, min_id, max_id = total_pairs
        logger.info(f"{total_count:,} pending pairs ({elapsed:.2f}s)")
        
        # Auto-calculate optimal batch size if not explicitly provided
        if batch_size is None:
            ram_gb = ram if ram else 16  # Default to 16GB if not specified
            cpus = n_cpus if n_cpus else 4  # Default to 4 if not specified
            batch_size = calculate_optimal_batch_size(ram_gb, total_count, cpus)
            logger.info(f"Auto-calculated batch size: {batch_size} (based on {ram_gb}GB RAM, {cpus} CPUs, {total_count:,} pairs)")
        else:
            logger.info(f"Using specified batch size: {batch_size}")
        total_files = 0
        if set_progress:
            total_files = conn.execute("""
                                       SELECT COUNT(DISTINCT ms_file_label)
                                       FROM pending_result_pairs
                                       """).fetchone()[0]
        _send_progress(
            set_progress,
            0,
            stage="Results",
            detail=f"Pending pairs: {total_count:,}",
        )

    # Process in batches
    processed = 0
    failed = 0
    batches = 0
    processed_files: set[str] = set()

    current_id = min_id
    batch_num = 1
    total_batches = (total_count + batch_size - 1) // batch_size

    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        # Settings for bulk writes
        conn.execute("SET wal_autocheckpoint='1GB'")
        conn.execute("BEGIN TRANSACTION")

        batches_in_txn = 0

        while current_id <= max_id:
            start_id = current_id
            end_id = current_id + batch_size - 1
            log_line = None  # Initialize before try block to avoid UnboundLocalError in finally

            try:
                batch_count = conn.execute("""
                                           SELECT COUNT(*)
                                           FROM pending_result_pairs
                                           WHERE pair_id BETWEEN ? AND ?
                                           """, [start_id, end_id]).fetchone()[0]

                if batch_count == 0:
                    current_id += batch_size
                    continue

                batch_start = time.time()

                conn.execute(QUERY_PROCESS_BATCH, [start_id, end_id])

                if set_progress:
                    batch_files = conn.execute("""
                                               SELECT DISTINCT ms_file_label
                                               FROM pending_result_pairs
                                               WHERE pair_id BETWEEN ? AND ?
                                               """, [start_id, end_id]).fetchall()
                    processed_files.update(
                        row[0] for row in batch_files if row and row[0] is not None
                    )

                batch_elapsed = time.time() - batch_start
                processed += batch_count
                batches += 1
                batches_in_txn += 1

                pairs_per_sec = batch_count / batch_elapsed
                logger.info(f"Batch {batch_num:>4}/{total_batches} | "
                      f"IDs {start_id:>6}-{end_id:>6} | "
                      f"{batch_count:>4} pairs | "
                      f"Batch time: {batch_elapsed:>5.2f}s | "
                      f"Progress {processed:>6,}/{total_count:,}")
                log_line = (f"Batch {batch_num}/{total_batches} | "
                            f"Progress {processed:,}/{total_count:,} | "
                            f"Time/batch {batch_elapsed:0.2f}s"
                            # f"Processing ({pairs_per_sec:0.1f} pairs/s)"
                            )


                # Periodic checkpoint
                if batches_in_txn >= checkpoint_every:
                    flush_start = time.time()
                    conn.execute("COMMIT")
                    conn.execute("CHECKPOINT")
                    conn.execute("BEGIN TRANSACTION")
                    logger.debug(f"  [Commit + Checkpoint]... {time.time() - flush_start:.2f}s")
                    batches_in_txn = 0

                batch_num += 1

            except Exception as e:
                batch_elapsed = time.time() - batch_start if 'batch_start' in locals() else 0
                failed += batch_count if 'batch_count' in locals() else 0

                logger.error(f"Error processing batch: {batch_elapsed:>5.2f}s | Error: {str(e)[:80]}")

                # Recover from aborted transaction to allow subsequent batches to proceed
                try:
                    conn.execute("ROLLBACK")
                    conn.execute("BEGIN TRANSACTION")
                    batches_in_txn = 0
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback transaction: {rollback_error}")

                with open('failed_batches_results.log', 'a') as f:
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Batch {batch_num}/{total_batches}\n")
                    f.write(f"IDs: {start_id}-{end_id}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"{'=' * 60}\n")
                
                batch_num += 1  # Move to next batch even on failure

            finally:
                if 'batch_count' in locals() and batch_count > 0:
                    progress_pct = (processed / total_count) * 100
                    files_done = len(processed_files) if set_progress else 0
                    detail_text = (
                        f"{log_line}"
                        if log_line else
                        f"Results batch {batch_num}/{total_batches} | "
                        f"Pairs {processed:,}/{total_count:,}"
                    )
                    _send_progress(
                        set_progress,
                        round(progress_pct, 1),
                        stage="Results",
                        detail=detail_text,
                    )

            current_id += batch_size

        # Commit final
        logger.info("Final commit + checkpoint...")
        flush_start = time.time()
        conn.execute("COMMIT")
        conn.execute("CHECKPOINT")
        logger.info(f"Checkpoint completed in {time.time() - flush_start:.2f}s")

    # Clean up
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        conn.execute("DROP TABLE IF EXISTS pending_result_pairs")

    logger.info(
        f"Results computation complete. "
        f"Total pairs: {total_count:,}, Processed: {processed:,}, "
        f"Failed: {failed:,}, Batches: {batches:,}"
    )

    return {
        'total_pairs': total_count,
        'processed': processed,
        'failed': failed,
        'batches': batches
    }


def compute_fitted_results(
    wdir: str,
    use_bookmarked: bool = False,
    recompute: bool = False,
    n_workers: int = 8,
    set_progress=None,
    n_cpus=None,
    ram=None
) -> dict:
    """
    Compute EMG peak fitting for all results.
    
    This function:
    1. Queries existing results with chromatogram data
    2. Fits EMG model to each peak using parallel processing
    3. Updates results table with fitted metrics
    
    Parameters:
        wdir: Working directory path
        use_bookmarked: Only process bookmarked targets
        recompute: Recompute even if fitting was already done
        n_workers: Number of parallel workers (default 8)
        set_progress: Progress callback function
        n_cpus: CPU cores for DuckDB
        ram: RAM allocation for DuckDB
    
    Returns:
        Dictionary with processing statistics
    """
    from .peak_fitting import fit_peaks_batch
    
    _send_progress(set_progress, 0, stage="Peak Fitting", detail="Preparing data...")
    
    # Query results that need fitting
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        # Build query to get peaks needing fitting
        where_conditions = []
        if use_bookmarked:
            where_conditions.append(
                "r.peak_label IN (SELECT peak_label FROM targets WHERE bookmark = TRUE)"
            )
        if not recompute:
            where_conditions.append("(r.fit_success IS NULL OR r.fit_success = FALSE)")
        
        where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        
        # Get chromatogram data for fitting
        query = f"""
            SELECT 
                r.peak_label,
                r.ms_file_label,
                r.scan_time,
                r.intensity,
                t.rt
            FROM results r
            JOIN targets t ON r.peak_label = t.peak_label
            {where_clause}
        """
        
        data = conn.execute(query).fetchall()
        
    if not data:
        logger.info("No peaks require fitting")
        _send_progress(set_progress, 100, stage="Peak Fitting", detail="No peaks to fit")
        return {'total': 0, 'fitted': 0, 'failed': 0}
    
    total_peaks = len(data)
    logger.info(f"Fitting {total_peaks:,} peaks with {n_workers} workers...")
    
    _send_progress(
        set_progress, 
        5, 
        stage="Peak Fitting", 
        detail=f"Fitting {total_peaks:,} peaks..."
    )
    
    # Prepare data for parallel processing
    # Format: (peak_label, ms_file_label, scan_time, intensity, expected_rt)
    peaks_data = [
        (row[0], row[1], row[2], row[3], row[4])
        for row in data
    ]
    
    # Progress callback for fitting - maps to 5-80% of progress bar
    batch_start_time = [time.time()]  # Start of current batch
    last_batch_duration = [0.0]       # Duration of last completed batch
    last_logged_batch = [0]
    batch_size = max(100, total_peaks // 20)  # ~20 batches or 100 peaks minimum
    
    def fitting_progress(completed, total, rate):
        # Map fitting progress (0-100%) to UI progress (5-80%)
        fit_pct = (completed / total) * 100 if total > 0 else 0
        ui_pct = 5 + (fit_pct * 0.75)  # 5% + up to 75% = 80% max
        
        # Calculate batch info
        current_batch_idx = (completed - 1) // batch_size
        batch_num = current_batch_idx + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        current_time = time.time()
        
        # Check if we moved to a new batch (or finished)
        if batch_num > last_logged_batch[0] or completed == total:
            duration = current_time - batch_start_time[0]
            last_batch_duration[0] = duration
            
            logger.info(
                f"Batch {batch_num:4}/{total_batches} | "
                f"Peaks {completed:6,}/{total:,} | "
                f"Batch time: {duration:5.2f}s | "
                f"Rate: {rate:.0f}/sec"
            )
            
            # Prepare for next batch
            if completed < total:
                batch_start_time[0] = current_time
                last_logged_batch[0] = batch_num
        
        # Display time: Use last batch duration for stability, or current elapsed for batch 1
        display_time = last_batch_duration[0] if batch_num > 1 else (current_time - batch_start_time[0])
        
        # Consistent format for UI: "Batch X/Y | Progress X/Y | Time/batch Zs"
        detail = f"Batch {batch_num}/{total_batches} | Progress {completed:,}/{total:,} | Time/batch {display_time:.2f}s"
        
        _send_progress(set_progress, round(ui_pct, 1), stage="Peak Fitting", detail=detail)
    
    # Run parallel fitting with progress updates
    start_time = time.time()
    fit_results = fit_peaks_batch(peaks_data, n_workers=n_workers, progress_callback=fitting_progress)
    elapsed = time.time() - start_time
    
    logger.info(f"Fitting completed in {elapsed:.2f}s ({total_peaks/elapsed:.0f} peaks/sec)")
    
    _send_progress(
        set_progress, 
        80, 
        stage="Peak Fitting", 
        detail=f"Updating database..."
    )
    
    # Update database with fit results in batches for progress tracking
    fitted_count = 0
    failed_count = 0
    total_results = len(fit_results)
    update_batch_size = max(100, total_results // 10)  # ~10 updates
    db_start_time = time.time()
    last_db_batch_time = time.time()
    last_db_duration = 0.0
    
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        conn.execute("BEGIN TRANSACTION")
        
        for i, result in enumerate(fit_results):
            try:
                conn.execute("""
                    UPDATE results
                    SET peak_area_fitted = ?,
                        peak_sigma = ?,
                        peak_tau = ?,
                        peak_asymmetry = ?,
                        peak_rt_fitted = ?,
                        fit_r_squared = ?,
                        fit_success = ?
                    WHERE peak_label = ? AND ms_file_label = ?
                """, [
                    result['peak_area_fitted'],
                    result['peak_sigma'],
                    result['peak_tau'],
                    result['peak_asymmetry'],
                    result['peak_rt_fitted'],
                    result['fit_r_squared'],
                    result['fit_success'],
                    result['peak_label'],
                    result['ms_file_label'],
                ])
                
                if result['fit_success']:
                    fitted_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to update {result['peak_label']}: {e}")
                failed_count += 1
            
            # Progress update every batch
            if (i + 1) % update_batch_size == 0 or (i + 1) == total_results:
                # Map progress from 80-98%
                db_pct = ((i + 1) / total_results) * 18  # 18% range for DB updates
                ui_pct = 80 + db_pct
                
                current_time = time.time()
                batch_duration = current_time - last_db_batch_time
                last_db_duration = batch_duration
                last_db_batch_time = current_time
                
                db_elapsed = current_time - db_start_time
                rate = (i + 1) / db_elapsed if db_elapsed > 0 else 0
                batch_num = (i + 1) // update_batch_size
                total_batches = (total_results // update_batch_size) + (1 if total_results % update_batch_size else 0)
                
                # Consistent format for UI: "Batch X/Y | Progress X/Y | Time/batch Zs"
                detail = f"Batch {batch_num}/{total_batches} | Progress {i+1:,}/{total_results:,} | Time/batch {batch_duration:.2f}s"
                
                logger.info(
                    f"Batch {batch_num:4}/{total_batches} | "
                    f"Rows {i+1:6,}/{total_results:,} | "
                    f"Batch time: {batch_duration:5.2f}s | "
                    f"Progress {i+1:,}/{total_results:,}"
                )
                _send_progress(set_progress, round(ui_pct, 1), stage="Peak Fitting", detail=detail)
        
        conn.execute("COMMIT")
        conn.execute("CHECKPOINT")
    
    _send_progress(
        set_progress, 
        100, 
        stage="Peak Fitting", 
        detail=f"Fitted {fitted_count:,} peaks"
    )
    
    total_elapsed = time.time() - start_time
    logger.info(
        f"Peak fitting complete. "
        f"Total: {total_peaks:,}, Fitted: {fitted_count:,}, Failed: {failed_count:,}"
    )
    
    return {
        'total': total_peaks,
        'fitted': fitted_count,
        'failed': failed_count,
        'elapsed_seconds': total_elapsed
    }


def compute_peak_properties(con: duckdb.DuckDBPyConnection,
                            set_progress=None,
                            recompute=False,
                            bookmarked=False
                            ):
    if recompute:
        logger.info("Deleting existing results for recalculation...")
        con.execute("DELETE FROM results")

    query = """
            INSERT INTO results (peak_label,
                                 ms_file_label,
                                 total_intensity,
                                 peak_area,
                                 peak_area_top3,
                                 peak_max,
                                 peak_min,
                                 peak_mean,
                                 peak_rt_of_max,
                                 peak_median,
                                 peak_n_datapoints,
                                 scan_time,
                                 intensity)
            WITH pairs_to_process AS (SELECT c.peak_label,
                                             c.ms_file_label
                                      FROM chromatograms c
                                               JOIN targets t ON c.peak_label = t.peak_label
                                      WHERE c.ms_file_label IN
                                            (SELECT ms_file_label FROM samples WHERE use_for_processing = TRUE)
                                        AND t.rt_min IS NOT NULL
                                        AND t.rt_max IS NOT NULL
                                        AND CASE
                                                WHEN ? -- bookmarked
                                                    THEN c.peak_label IN (SELECT peak_label FROM targets WHERE bookmark = TRUE)
                                                ELSE TRUE
                                          END
                                        AND (
                                          ? -- recompute
                                              OR NOT EXISTS (SELECT 1
                                                             FROM results r
                                                             WHERE r.peak_label = c.peak_label
                                                               AND r.ms_file_label = c.ms_file_label)
                                          )),
                 unnested AS (SELECT c.peak_label,
                                     c.ms_file_label,
                                     UNNEST(c.scan_time) AS scan_time,
                                     UNNEST(c.intensity) AS intensity
                              FROM chromatograms c
                                       JOIN pairs_to_process p ON c.peak_label = p.peak_label
                                  AND c.ms_file_label = p.ms_file_label),
-- Compute total_intensity (without rt filter)
                 total_stats AS (SELECT peak_label,
                                        ms_file_label,
                                        SUM(intensity) AS total_intensity
                                 FROM unnested
                                 GROUP BY peak_label, ms_file_label),
-- Filter by rt_min - rt_max window
                 filtered_range AS (SELECT u.peak_label,
                                           u.ms_file_label,
                                           u.scan_time,
                                           u.intensity
                                    FROM unnested u
                                             JOIN targets t ON u.peak_label = t.peak_label
                                    WHERE u.scan_time BETWEEN t.rt_min AND t.rt_max),
-- Compute trapezoid segments for each consecutive pair of points
                 trapezoid_segments AS (
                     SELECT peak_label,
                            ms_file_label,
                            scan_time,
                            intensity,
                            -- Trapezoid area for segment [i, i+1]: (y[i] + y[i+1])/2 * (x[i+1] - x[i])
                            CASE 
                                WHEN LEAD(scan_time) OVER w IS NOT NULL 
                                THEN (intensity + LEAD(intensity) OVER w) / 2.0 
                                     * (LEAD(scan_time) OVER w - scan_time)
                                ELSE 0
                            END AS segment_area
                     FROM filtered_range
                     WINDOW w AS (PARTITION BY peak_label, ms_file_label ORDER BY scan_time)
                 ),
-- Group the filtered data into lists
                 aggregated AS (SELECT peak_label,
                                       ms_file_label,
                                       LIST(scan_time ORDER BY scan_time) AS scan_time,
                                       LIST(intensity ORDER BY scan_time) AS intensity,
                                       -- Trapezoid integration (more accurate than simple sum)
                                       ROUND(SUM(segment_area), 0)        AS peak_area,
                                       ROUND(MAX(intensity), 0)           AS peak_max,
                                       ROUND(MIN(intensity), 0)           AS peak_min,
                                       ROUND(AVG(intensity), 0)           AS peak_mean,
                                       ROUND(MEDIAN(intensity), 0)        AS peak_median,
                                       COUNT(*)                           AS peak_n_datapoints
                                FROM trapezoid_segments
                                GROUP BY peak_label, ms_file_label),
-- Compute peak_area_top3
                 top3_calc AS (
                     SELECT 
                         peak_label,
                         ms_file_label,
                         ROUND(intensity + prev_intensity + next_intensity, 0) AS peak_area_top3
                     FROM (
                         SELECT 
                             peak_label,
                             ms_file_label,
                             intensity,
                             COALESCE(LAG(intensity) OVER (PARTITION BY peak_label, ms_file_label ORDER BY scan_time), 0) AS prev_intensity,
                             COALESCE(LEAD(intensity) OVER (PARTITION BY peak_label, ms_file_label ORDER BY scan_time), 0) AS next_intensity,
                             ROW_NUMBER() OVER (PARTITION BY peak_label, ms_file_label ORDER BY intensity DESC) AS rn
                         FROM filtered_range
                     ) ranked
                     WHERE rn = 1
                 ),
-- Find scan_time of peak_max
                 rt_of_max AS (SELECT peak_label,
                                      ms_file_label,
                                      scan_time AS peak_rt_of_max
                               FROM (SELECT peak_label,
                                            ms_file_label,
                                            scan_time,
                                            intensity,
                                            ROW_NUMBER() OVER (PARTITION BY peak_label, ms_file_label ORDER BY intensity DESC) AS rn
                                     FROM filtered_range) sub
                               WHERE rn = 1)
            SELECT a.peak_label,
                   a.ms_file_label,
                   ts.total_intensity,
                   a.peak_area,
                   t3.peak_area_top3,
                   a.peak_max,
                   a.peak_min,
                   a.peak_mean,
                   rm.peak_rt_of_max,
                   a.peak_median,
                   a.peak_n_datapoints,
                   a.scan_time,
                   a.intensity
            FROM aggregated a
                     JOIN total_stats ts ON a.peak_label = ts.peak_label AND a.ms_file_label = ts.ms_file_label
                     JOIN top3_calc t3 ON a.peak_label = t3.peak_label AND a.ms_file_label = t3.ms_file_label
                     JOIN rt_of_max rm ON a.peak_label = rm.peak_label AND a.ms_file_label = rm.ms_file_label
            ORDER BY a.peak_label, a.ms_file_label;
            """

    # Shared variable to accumulate progress
    accumulated_progress = [0.0]
    stop_monitoring = [False]

    def monitor_progress():
        """Monitor progress of the current query"""
        while not stop_monitoring[0]:
            try:
                qp = con.query_progress()
                if qp != -1 and qp > 0:
                    # Total progress depends on which query is executing
                    total_progress = qp

                    accumulated_progress[0] = total_progress
                    if set_progress:
                        set_progress(round(total_progress, 1))
                    logger.info(f"Progress: {total_progress:.1f}%")
                time.sleep(0.05)

            except (duckdb.InvalidInputException, duckdb.ConnectionException):
                break
            except Exception as e:
                logger.error(f"Progress monitoring error: {e}")
                break

    # Start monitoring
    if set_progress:
        progress_thread = Thread(target=monitor_progress, daemon=True)
        progress_thread.start()

    try:
        # Run MS1
        logger.info("Processing MS1 chromatograms...")
        con.execute(query, [bookmarked, recompute])
        accumulated_progress[0] = 100
        if set_progress:
            set_progress(round(accumulated_progress[0], 1))

        logger.info("Chromatograms computed and inserted into DuckDB.")

    finally:
        stop_monitoring[0] = True
        if set_progress:
            progress_thread.join(timeout=0.5)

    logger.info("Peak properties computed and inserted into DuckDB.")


def create_pivot(conn, rows=None, cols=None, value='peak_area', table='results'):
    """
    Create pivot from DuckDB for unique per-pair data
    """

    # Use fetchall() for faster list extraction (3.25x speedup vs DataFrame)
    ordered_pl = [row[0] for row in conn.execute(f"""
        SELECT DISTINCT r.peak_label
        FROM {table} r
        JOIN targets t ON r.peak_label = t.peak_label
        ORDER BY t.ms_type
    """).fetchall()]

    group_cols_sql = ",\n                ".join([f"s.{col}" for col in GROUP_COLUMNS])

    query = f"""
        PIVOT (
            SELECT
                s.ms_type,
                s.sample_type,
                {group_cols_sql},
                r.ms_file_label,
                r.peak_label,
                r.{value}
            FROM {table} r
            JOIN samples s ON s.ms_file_label = r.ms_file_label
            WHERE s.use_for_analysis = TRUE
            ORDER BY s.ms_type, r.peak_label
        )
        ON peak_label
        USING FIRST({value})
        -- GROUP BY ms_type
        ORDER BY ms_type
    """
    df = conn.execute(query).df()
    meta_cols = ['ms_type', 'sample_type', *GROUP_COLUMNS, 'ms_file_label']
    keep_cols = [col for col in meta_cols if col in df.columns] + ordered_pl
    return df[keep_cols]


def compute_and_insert_chromatograms_iteratively(con: duckdb.DuckDBPyConnection, set_progress=None):
    """
    Computes and inserts chromatograms iteratively by batching targets.

    :param con: An active DuckDB connection.
    :param set_progress: A callback function to update the progress bar.
    """
    # Use SQL filtering for faster list extraction (1.46x speedup vs DataFrame filtering)
    ms1_targets = [row[0] for row in con.execute(
        "SELECT peak_label FROM targets WHERE ms_type = 'ms1'"
    ).fetchall()]
    ms2_targets = [row[0] for row in con.execute(
        "SELECT peak_label FROM targets WHERE ms_type = 'ms2'"
    ).fetchall()]
    ms_files_count = con.execute("SELECT count(*) FROM samples WHERE use_for_optimization = TRUE").fetchone()[0]

    if ms_files_count == 0:
        if set_progress:
            set_progress(100)
        return

    n_total = len(ms1_targets) + len(ms2_targets)
    processed_count = 0

    def process_batch(targets_batch):
        nonlocal processed_count
        if not targets_batch:
            return

        placeholders = ', '.join(['?'] * len(targets_batch))
        query = f"""
        INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity)
        WITH precomputed AS (
            SELECT
                t.peak_label,
                m.ms_file_label,
                list(m.scan_time ORDER BY m.scan_time) AS scan_time,
                list(m.intensity ORDER BY m.scan_time) AS intensity
            FROM ms_data AS m
            JOIN targets AS t ON m.mz BETWEEN t.mz_mean - (t.mz_mean * t.mz_width / 1e6)
                                       AND t.mz_mean + (t.mz_mean * t.mz_width / 1e6)
            JOIN samples AS s ON m.ms_file_label = s.ms_file_label
            WHERE s.use_for_optimization = TRUE AND t.peak_label IN ({placeholders})
            GROUP BY t.peak_label, m.ms_file_label
        )
        SELECT peak_label, ms_file_label, scan_time, intensity
        FROM precomputed
        ON CONFLICT (ms_file_label, peak_label) DO UPDATE
            SET scan_time = excluded.scan_time,
                intensity = excluded.intensity;
        """
        con.execute(query, targets_batch)
        processed_count += len(targets_batch)
        if set_progress and n_total > 0:
            progress = round(processed_count / n_total * 100, 2)
            set_progress(progress)

    # Process MS2 targets in batches of 10
    for i in range(0, len(ms2_targets), 10):
        process_batch(ms2_targets[i:i + 10])

    # Determine batch size for MS1 targets
    ms1_batch_size = 1 if ms_files_count >= 50 else 5
    for i in range(0, len(ms1_targets), ms1_batch_size):
        process_batch(ms1_targets[i:i + ms1_batch_size])

    if set_progress and n_total > 0:
        set_progress(100)

    logger.info("Iterative chromatogram computation complete.")
