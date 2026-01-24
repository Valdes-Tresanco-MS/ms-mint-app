from contextlib import contextmanager
from pathlib import Path
from threading import Thread

import duckdb
import time
import logging

logger = logging.getLogger(__name__)

from .sample_metadata import GROUP_COLUMNS


def calculate_optimal_batch_size(ram_gb: int, total_pairs: int, n_cpus: int = None) -> int:
    """
    Calculate optimal batch size for chromatogram/results extraction based on resources.
    
    Formula:
    - Base: 500 pairs
    - RAM factor: scales by 4GB increments (1000 per 4GB)
    - CPU factor: scales with cores (max 2x boost)
    - Cap: 10000 (diminishing returns above this)
    - Minimum: 500 (to avoid too many small batches)
    - At least 10 batches for progress reporting
    
    Args:
        ram_gb: Available RAM in GB for DuckDB
        total_pairs: Total number of pairs to process
        n_cpus: Number of CPUs allocated to DuckDB
        
    Returns:
        Optimal batch size
    """
    ram_gb = ram_gb or 16  # Default to 16GB if not specified
    
    # RAM factor: 1000 pairs per 4GB (using float division for smooth scaling)
    ram_factor = max(ram_gb, 4) / 4
    
    # CPU factor: modest scaling (max 1.5x boost)
    # Based on benchmarks: larger batches have diminishing returns
    effective_cpus = min(n_cpus or 4, ram_gb)  # Cap CPUs at RAM GB
    cpu_factor = min(effective_cpus / 8 + 0.5, 1.5)  # Gentler scaling, max 1.5x
    
    base_batch = 750  # Conservative base for good throughput
    optimal = int(base_batch * ram_factor * cpu_factor)
    
    # Ensure at least 10 batches for progress reporting
    if total_pairs > 0:
        optimal = min(optimal, max(total_pairs // 10, 500))
    
    return min(max(optimal, 500), 8000)  # Cap at 8000 (benchmarks show diminishing returns above 5000)


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
    return min(n_cpus, ram_gb*2)


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
                     ms_type       ms_type_enum,
                     -- mz            DOUBLE[],
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
    
    # Migration: Add rt_aligned and rt_shift columns to existing results tables
    existing_cols = {
        row[0] for row in conn.execute("DESCRIBE results").fetchall()
    }
    if 'rt_aligned' not in existing_cols:
        conn.execute("ALTER TABLE results ADD COLUMN rt_aligned BOOLEAN")
        logger.info("Migration: Added 'rt_aligned' column to results table")
    if 'rt_shift' not in existing_cols:
        conn.execute("ALTER TABLE results ADD COLUMN rt_shift DOUBLE")
        logger.info("Migration: Added 'rt_shift' column to results table")



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
                                                AND t.peak_selection IS TRUE
                                               OR NOT EXISTS (SELECT 1
                                                              FROM targets t1
                                                              WHERE t1.peak_selection IS TRUE)
                                                AND
                                                  EXISTS(SELECT 1 FROM ms1_data md WHERE md.ms_file_label = s.ms_file_label)),
                            ms2_targets AS (SELECT DISTINCT t.peak_label, s.ms_file_label
                                            FROM targets t
                                                     CROSS JOIN samples_to_use s
                                            WHERE t.filterLine IS NOT NULL -- ensures this is MS2
                                                AND t.peak_selection IS TRUE
                                               OR NOT EXISTS (SELECT 1
                                                              FROM targets t1
                                                              WHERE t1.peak_selection IS TRUE)
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
                                     AND t.peak_selection IS TRUE
                                    OR NOT EXISTS (SELECT 1
                                                   FROM targets t1
                                                   WHERE t1.peak_selection IS TRUE)
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
                                     AND t.peak_selection IS TRUE
                                    OR NOT EXISTS (SELECT 1
                                                   FROM targets t1
                                                   WHERE t1.peak_selection IS TRUE)
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
                                                 t.rt_min,
                                                 t.rt_max,
                                                 s.ms_file_label
                                          FROM targets t
                                                   JOIN samples s
                                                        ON (CASE WHEN ? THEN s.use_for_optimization ELSE s.use_for_processing END) =
                                                           TRUE
                                          WHERE t.mz_mean IS NOT NULL
                                              AND t.mz_width IS NOT NULL
                                              AND t.peak_selection IS TRUE
                                             OR NOT EXISTS (SELECT 1
                                                            FROM targets t1
                                                            WHERE t1.peak_selection IS TRUE)
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
                                                 t.rt_min,
                                                 t.rt_max,
                                                 s.ms_file_label
                                          FROM targets AS t
                                                   JOIN samples s
                                                        ON (CASE WHEN ? THEN s.use_for_optimization ELSE s.use_for_processing END) =
                                                           TRUE
                                          WHERE t.filterLine IS NOT NULL
                                              AND t.peak_selection IS TRUE
                                             OR NOT EXISTS (SELECT 1
                                                            FROM targets t1
                                                            WHERE t1.peak_selection IS TRUE)
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
                                                                    t.rt_min,
                                                                    t.rt_max
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
                              INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                              WITH batch_pairs AS (SELECT peak_label, ms_file_label, ms_type, mz_mean, mz_width, rt_min, rt_max
                                                   FROM pending_pairs
                                                   WHERE ms_type = 'ms1'
                                                     AND pair_id BETWEEN ? AND ?),
                                   -- Step 1: Find intensities (only rows with signal)
                                   matched_intensities AS (SELECT bp.peak_label,
                                                                  bp.ms_file_label,
                                                                  ms1.scan_id,
                                                                  MAX(ms1.intensity) AS intensity
                                                           FROM batch_pairs bp
                                                                    JOIN ms1_data ms1
                                                                         ON ms1.ms_file_label = bp.ms_file_label
                                                                             AND ms1.mz BETWEEN
                                                                                bp.mz_mean - (bp.mz_mean * bp.mz_width / 1e6)
                                                                                AND
                                                                                bp.mz_mean + (bp.mz_mean * bp.mz_width / 1e6)
                                                           GROUP BY bp.peak_label, bp.ms_file_label, ms1.scan_id),
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
                                                              LEFT JOIN matched_intensities m
                                                                        ON a.peak_label = m.peak_label
                                                                            AND a.ms_file_label = m.ms_file_label
                                                                            AND a.scan_id = m.scan_id),
                                   agg AS (SELECT peak_label,
                                                  ms_file_label,
                                                  LIST(scan_time ORDER BY scan_time) AS scan_time,
                                                  LIST(intensity ORDER BY scan_time) AS intensity
                                           FROM complete_data
                                           GROUP BY peak_label, ms_file_label)
                              SELECT peak_label, ms_file_label, scan_time, intensity, 'ms1' AS ms_type
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


def compute_chromatograms_optimized(
        wdir: str,
        use_for_optimization: bool,
        checkpoint_every: int = 10,
        set_progress=None,
        recompute_ms1: bool = False,
        recompute_ms2: bool = False,
        n_cpus=None,
        ram=None,
        use_bookmarked: bool = False,
        # objetivo de “pares pendientes” por ciclo (define dinámicamente cuántos archivos entran)
        pairs_per_cycle: int = 10_000,
        # airbag por si cada archivo tiene pocos pares pendientes (p.ej. +1 target)
        max_files_per_cycle: int = 50,
):
    """
    Versión por “batch de archivos”: cada ciclo ejecuta 1 query que procesa *varios* ms_file_label a la vez
    (y opcionalmente sub-batches de targets), para que DuckDB tenga suficiente volumen y use threads.

    La query calcula pares pendientes *en el momento de ejecución* (anti-join contra chromatograms), así:
      - nuevo archivo -> faltan muchos pares (todos los targets) para ese archivo
      - nuevo target  -> falta ese target para muchos archivos
    y lo resuelve eficientemente sin computar archivo-por-archivo.
    """

    logger.info(f"Computing chromatograms (batch-files). wDir: {wdir}")

    # === Query: scan lookup ===
    QUERY_CREATE_SCAN_LOOKUP = """
                               CREATE TABLE IF NOT EXISTS ms_file_scans AS
                               SELECT DISTINCT ms_file_label, scan_id, scan_time, 'ms1' AS ms_type
                               FROM ms1_data
                               UNION ALL
                               SELECT DISTINCT ms_file_label, scan_id, scan_time, 'ms2' AS ms_type
                               FROM ms2_data
                               ORDER BY ms_file_label, scan_id, ms_type;

                               CREATE INDEX IF NOT EXISTS idx_ms_file_scans_file_scan
                                   ON ms_file_scans (ms_file_label, scan_id, ms_type); \
                               """

    # === Batch query MS1: procesa varios archivos a la vez; calcula pares pendientes dentro del SQL ===
    # Params: [file_labels_list, peak_labels_list, use_bookmarked]
    QUERY_PROCESS_FILES_MS1 = """
                              INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                              WITH file_batch AS (SELECT UNNEST(?) ::VARCHAR AS ms_file_label),
                                   target_filter AS (SELECT peak_label, mz_mean, mz_width, rt_min, rt_max
                                                     FROM targets t
                                                     WHERE t.ms_type = 'ms1'
                                                       AND t.peak_label = ANY(?)
                                                       AND (t.peak_selection IS TRUE OR NOT EXISTS (SELECT 1
                                                                                                    FROM targets t1
                                                                                                    WHERE t1.peak_selection IS TRUE
                                                                                                      AND t1.ms_type = 'ms1'))
                                                       AND (CASE WHEN ? THEN t.bookmark IS TRUE ELSE TRUE END)),
                                   pending_pairs AS (SELECT fb.ms_file_label,
                                                            tf.peak_label,
                                                            tf.mz_mean,
                                                            tf.mz_width,
                                                            tf.rt_min,
                                                            tf.rt_max
                                                     FROM file_batch fb
                                                              CROSS JOIN target_filter tf
                                                     WHERE NOT EXISTS (SELECT 1
                                                                       FROM chromatograms c
                                                                       WHERE c.ms_type = 'ms1'
                                                                         AND c.ms_file_label = fb.ms_file_label
                                                                         AND c.peak_label = tf.peak_label)),
                                   file_scans AS (SELECT s.ms_file_label, s.scan_id, s.scan_time
                                                  FROM ms_file_scans s
                                                           JOIN file_batch fb USING (ms_file_label)
                                                  WHERE s.ms_type = 'ms1'),
                                   matched_intensities AS (SELECT pp.peak_label,
                                                                  pp.ms_file_label,
                                                                  ms1.scan_id,
                                                                  MAX(ms1.intensity) AS intensity
                                                           FROM pending_pairs pp
                                                                    JOIN ms1_data ms1
                                                                         ON ms1.ms_file_label = pp.ms_file_label
                                                                             AND
                                                                            ms1.mz BETWEEN pp.mz_mean - (pp.mz_mean * pp.mz_width / 1e6)
                                                                                AND pp.mz_mean + (pp.mz_mean * pp.mz_width / 1e6)
                                                           GROUP BY pp.peak_label, pp.ms_file_label, ms1.scan_id),
                                   scans_in_window AS (SELECT pp.peak_label,
                                                              pp.ms_file_label,
                                                              fs.scan_id,
                                                              fs.scan_time
                                                       FROM pending_pairs pp
                                                                JOIN file_scans fs
                                                                     ON fs.ms_file_label = pp.ms_file_label
                                                                         AND
                                                                        fs.scan_time BETWEEN COALESCE(pp.rt_min, 0) - 30
                                                                            AND COALESCE(pp.rt_max, 999999) + 30),
                                   complete_data AS (SELECT sw.peak_label,
                                                            sw.ms_file_label,
                                                            sw.scan_time,
                                                            COALESCE(ROUND(mi.intensity, 0), 1) AS intensity
                                                     FROM scans_in_window sw
                                                              LEFT JOIN matched_intensities mi
                                                                        ON sw.peak_label = mi.peak_label
                                                                            AND sw.ms_file_label = mi.ms_file_label
                                                                            AND sw.scan_id = mi.scan_id)
                              SELECT peak_label,
                                     ms_file_label,
                                     LIST(scan_time ORDER BY scan_time) AS scan_time,
                                     LIST(intensity ORDER BY scan_time) AS intensity,
                                     'ms1'                              AS ms_type
                              FROM complete_data
                              GROUP BY peak_label, ms_file_label; \
                              """

    # === Batch query MS2: idem ===
    # Params: [file_labels_list, peak_labels_list, use_bookmarked]
    QUERY_PROCESS_FILES_MS2 = """
                              INSERT INTO chromatograms (peak_label, ms_file_label, scan_time, intensity, ms_type)
                              WITH file_batch AS (SELECT UNNEST(?) ::VARCHAR AS ms_file_label),
                                   target_filter AS (SELECT peak_label, filterLine, rt_min, rt_max
                                                     FROM targets t
                                                     WHERE t.ms_type = 'ms2'
                                                       AND t.peak_label = ANY(?)
                                                       AND (t.peak_selection IS TRUE OR NOT EXISTS (SELECT 1
                                                                                                    FROM targets t1
                                                                                                    WHERE t1.peak_selection IS TRUE
                                                                                                      AND t1.ms_type = 'ms2'))
                                                       AND (CASE WHEN ? THEN t.bookmark IS TRUE ELSE TRUE END)),
                                   pending_pairs AS (SELECT fb.ms_file_label,
                                                            tf.peak_label,
                                                            tf.filterLine,
                                                            tf.rt_min,
                                                            tf.rt_max
                                                     FROM file_batch fb
                                                              CROSS JOIN target_filter tf
                                                     WHERE NOT EXISTS (SELECT 1
                                                                       FROM chromatograms c
                                                                       WHERE c.ms_type = 'ms2'
                                                                         AND c.ms_file_label = fb.ms_file_label
                                                                         AND c.peak_label = tf.peak_label)),
                                   file_scans AS (SELECT s.ms_file_label, s.scan_id, s.scan_time
                                                  FROM ms_file_scans s
                                                           JOIN file_batch fb USING (ms_file_label)
                                                  WHERE s.ms_type = 'ms2'),
                                   matched_intensities AS (SELECT pp.peak_label,
                                                                  pp.ms_file_label,
                                                                  ms2.scan_id,
                                                                  ms2.intensity
                                                           FROM pending_pairs pp
                                                                    JOIN ms2_data ms2
                                                                         ON ms2.ms_file_label = pp.ms_file_label
                                                                             AND ms2.filterLine = pp.filterLine),
                                   scans_in_window AS (SELECT pp.peak_label,
                                                              pp.ms_file_label,
                                                              fs.scan_id,
                                                              fs.scan_time
                                                       FROM pending_pairs pp
                                                                JOIN file_scans fs
                                                                     ON fs.ms_file_label = pp.ms_file_label
                                                                         AND
                                                                        fs.scan_time BETWEEN COALESCE(pp.rt_min, 0) - 30
                                                                            AND COALESCE(pp.rt_max, 999999) + 30),
                                   complete_data AS (SELECT sw.peak_label,
                                                            sw.ms_file_label,
                                                            sw.scan_time,
                                                            COALESCE(ROUND(mi.intensity, 0), 1) AS intensity
                                                     FROM scans_in_window sw
                                                              LEFT JOIN matched_intensities mi
                                                                        ON sw.peak_label = mi.peak_label
                                                                            AND sw.ms_file_label = mi.ms_file_label
                                                                            AND sw.scan_id = mi.scan_id)
                              SELECT peak_label,
                                     ms_file_label,
                                     LIST(scan_time ORDER BY scan_time) AS scan_time,
                                     LIST(intensity ORDER BY scan_time) AS intensity,
                                     'ms2'                              AS ms_type
                              FROM complete_data
                              GROUP BY peak_label, ms_file_label; \
                              """

    # === Pending work (para armar ciclos por pares pendientes) ===
    QUERY_GET_PENDING = """
                        WITH target_filter AS (SELECT peak_label, ms_type
                                               FROM targets t
                                               WHERE (t.peak_selection IS TRUE OR NOT EXISTS (SELECT 1
                                                                                              FROM targets t1
                                                                                              WHERE t1.peak_selection IS TRUE
                                                                                                AND t1.ms_type = t.ms_type))
                                                 AND (CASE WHEN ? THEN t.bookmark IS TRUE ELSE TRUE END)),
                             sample_filter AS (SELECT ms_file_label
                                               FROM samples
                                               WHERE (CASE WHEN ? THEN use_for_optimization ELSE use_for_processing END) = TRUE),
                             existing_pairs AS (SELECT DISTINCT peak_label, ms_file_label, ms_type
                                                FROM chromatograms)
                        SELECT ms_type,
                               ms_file_label,
                               LIST(peak_label) AS peak_labels,
                               COUNT(*)         AS pair_count
                        FROM (SELECT t.peak_label, s.ms_file_label, t.ms_type
                              FROM target_filter t
                                       CROSS JOIN sample_filter s
                              WHERE NOT EXISTS (SELECT 1
                                                FROM existing_pairs e
                                                WHERE e.peak_label = t.peak_label
                                                  AND e.ms_file_label = s.ms_file_label
                                                  AND e.ms_type = t.ms_type))
                        GROUP BY ms_type, ms_file_label
                        ORDER BY ms_type, ms_file_label; \
                        """

    # === Cleanup inicial ===
    if recompute_ms1:
        logger.info("Deleting existing MS1 chromatograms...")
        with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
            con.execute("DELETE FROM chromatograms WHERE ms_type = 'ms1'")
            con.execute("CHECKPOINT")

    if recompute_ms2:
        logger.info("Deleting existing MS2 chromatograms...")
        with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
            con.execute("DELETE FROM chromatograms WHERE ms_type = 'ms2'")
            con.execute("CHECKPOINT")

    # === Crear scan lookup ===
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        conn.execute("CHECKPOINT")
        try:
            _ = conn.execute("SELECT 1 FROM ms_file_scans LIMIT 1").fetchone()
        except Exception:
            logger.info("Creating scan lookup table...")
            t0 = time.perf_counter()
            conn.execute(QUERY_CREATE_SCAN_LOOKUP)
            conn.execute("CHECKPOINT")
            logger.info(f"Scan lookup created in {time.perf_counter() - t0:.2f}s")

    # Limitar targets por sub-batch (para evitar explosión de intermedios cuando el batch de files es grande)
    max_targets_per_batch = recommend_max_targets(available_ram_gb=ram, safety_factor=0.95)

    # === Obtener pending ===
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        logger.info("Getting pending work...")
        pending = conn.execute(QUERY_GET_PENDING, [use_bookmarked, use_for_optimization]).fetchall()

    if not pending:
        logger.info("No pending work")
        return {'total_pairs': 0, 'processed': 0, 'failed': 0, 'batches': 0}

    total_pairs = sum(row[3] for row in pending)
    total_files = len(pending)
    logger.info(f"{total_pairs:,} pending pairs across {total_files} file-type combinations")

    def _take_cycle(pending_rows, start_i):
        """Selecciona un ciclo sin mezclar ms_type, hasta alcanzar pairs_per_cycle o max_files_per_cycle."""
        ms_type0 = pending_rows[start_i][0]
        files = []
        peaks = set()
        pairs = 0
        i = start_i

        while i < len(pending_rows):
            ms_type, ms_file_label, peak_labels_list, pair_count = pending_rows[i]
            if ms_type != ms_type0:
                break

            files.append(ms_file_label)
            peaks.update(peak_labels_list)
            pairs += pair_count
            i += 1

            if pairs >= pairs_per_cycle or len(files) >= max_files_per_cycle:
                break

        return ms_type0, files, list(peaks), pairs, i

    processed_est = 0
    failed = 0
    cycles = 0
    cycles_since_checkpoint = 0

    # === Ejecutar (1 conexión) ===
    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as conn:
        conn.execute("CHECKPOINT")
        conn.execute("BEGIN TRANSACTION")

        i = 0
        cycle_id = 1

        while i < len(pending):
            ms_type, file_batch, peak_batch, cycle_pairs, next_i = _take_cycle(pending, i)

            # Sub-batching de targets si el set es grande (sigue siendo 1 query por sub-batch, no por archivo)
            peak_batch = list(peak_batch)
            num_target_sub = (len(peak_batch) + max_targets_per_batch - 1) // max_targets_per_batch

            logger.info(
                f"Cycle {cycle_id}: {ms_type.upper()} | "
                f"{len(file_batch)} files | ~{cycle_pairs:,} pending pairs | "
                f"{len(peak_batch)} targets | target_sub_batches={num_target_sub}"
            )

            t_cycle = time.time()
            for sb in range(num_target_sub):
                t0 = sb * max_targets_per_batch
                t1 = min(t0 + max_targets_per_batch, len(peak_batch))
                sb_peaks = peak_batch[t0:t1]

                t_sb = time.time()
                if ms_type == "ms1":
                    conn.execute(QUERY_PROCESS_FILES_MS1, [file_batch, sb_peaks, use_bookmarked])
                else:
                    conn.execute(QUERY_PROCESS_FILES_MS2, [file_batch, sb_peaks, use_bookmarked])
                sb_dt = time.time() - t_sb

                logger.info(
                    f"  - Target sub-batch {sb + 1}/{num_target_sub}: {len(sb_peaks)} targets | {sb_dt:.2f}s"
                )

            cycle_dt = time.time() - t_cycle

            # progreso estimado con snapshot de pending (en la práctica coincide salvo cambios concurrentes)
            processed_est += cycle_pairs
            cycles += 1
            cycles_since_checkpoint += 1

            logger.info(
                f"Cycle {cycle_id} done in {cycle_dt:.2f}s | "
                f"Progress(est): {processed_est:,}/{total_pairs:,} ({100 * processed_est / total_pairs:.1f}%)"
            )

            if set_progress:
                progress_pct = (processed_est / total_pairs) * 100
                _send_progress(
                    set_progress,
                    round(progress_pct, 1),
                    stage="Chromatograms",
                    detail=f"{ms_type.upper()} | {processed_est:,}/{total_pairs:,} pairs"
                )

            # checkpoint/commit por ciclos (igual idea que antes, pero ahora “batch” = ciclo)
            if checkpoint_every and cycles_since_checkpoint >= checkpoint_every:
                conn.execute("COMMIT")
                conn.execute("CHECKPOINT")
                conn.execute("BEGIN TRANSACTION")
                cycles_since_checkpoint = 0

            i = next_i
            cycle_id += 1

        conn.execute("COMMIT")
        conn.execute("CHECKPOINT")

    logger.info(f"Complete: ~{processed_est:,} processed(est), {failed:,} failed, {cycles} cycles")

    return {
        'total_pairs': total_pairs,
        'processed': processed_est,  # estimado (basado en pending snapshot)
        'failed': failed,
        'batches': cycles
    }



def recommend_max_targets(available_ram_gb: int,
                          avg_scans_per_file: int = 500,
                          safety_factor: float = 0.5) -> int:
    """
    Recomienda max_targets_per_batch basado en RAM disponible.

    Args:
        available_ram_gb: RAM disponible en GB
        avg_scans_per_file: Promedio de scans por archivo
        safety_factor: Factor de seguridad (0.5 = usar solo 50% de RAM)

    Returns:
        Número recomendado de targets por batch
    """
    available_ram_mb = available_ram_gb * 1024 * safety_factor

    # RAM por target ≈ scans × 30 bytes
    ram_per_target_mb = (avg_scans_per_file * 30) / (1024 * 1024)

    max_targets = int(available_ram_mb / ram_per_target_mb)

    # Mínimo 10, máximo 200
    return max(10, min(10000, max_targets))


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
            -- Compute all metrics from arrays
            metrics AS (
                SELECT
                    len(intensity_arr) AS peak_n_datapoints,
                    ROUND(list_sum(intensity_arr), 0) AS peak_area,
                    ROUND(list_max(intensity_arr), 0) AS peak_max,
                    ROUND(list_min(intensity_arr), 0) AS peak_min,
                    ROUND(list_avg(intensity_arr), 0) AS peak_mean,
                    -- Median: sorted list at middle index
                    ROUND(list_sort(intensity_arr)[CAST(len(intensity_arr) / 2 + 1 AS BIGINT)], 0) AS peak_median,
                    -- RT of max intensity
                    scan_time_arr[CAST(list_position(intensity_arr, list_max(intensity_arr)) AS BIGINT)] AS peak_rt_of_max,
                    -- Index of max for top3 calculation
                    CAST(list_position(intensity_arr, list_max(intensity_arr)) AS BIGINT) AS max_idx,
                    scan_time_arr,
                    intensity_arr
                FROM arrays
                WHERE len(intensity_arr) > 0
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
-- Group the filtered data into lists
                 aggregated AS (SELECT peak_label,
                                       ms_file_label,
                                       LIST(scan_time ORDER BY scan_time) AS scan_time,
                                       LIST(intensity ORDER BY scan_time) AS intensity,
                                       ROUND(SUM(intensity), 0)           AS peak_area,
                                       ROUND(MAX(intensity), 0)           AS peak_max,
                                       ROUND(MIN(intensity), 0)           AS peak_min,
                                       ROUND(AVG(intensity), 0)           AS peak_mean,
                                       ROUND(MEDIAN(intensity), 0)        AS peak_median,
                                       COUNT(*)                           AS peak_n_datapoints
                                FROM filtered_range
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
