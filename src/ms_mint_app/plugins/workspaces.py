import re
import shutil
import json
import os
import psutil
from multiprocessing import cpu_count
import logging
from pathlib import Path

import dash
import feffery_antd_components as fac
import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from ..duckdb_manager import duckdb_connection_mint, duckdb_connection, validate_mint_database, import_database_as_workspace
from ..logging_setup import activate_workspace_logging, deactivate_workspace_logging
from ..plugin_interface import PluginInterface

_label = "Workspaces"
pattern = re.compile(r"^[A-Za-z0-9_]+$")



logger = logging.getLogger(__name__)

class WorkspacesPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 0
        logger.info(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)

    def outputs(self):
        return _outputs


_layout = html.Div(
    [
        dcc.Store(id="ws-action-store"),
        fac.AntdFlex(
            [
                fac.AntdTitle('Workspaces', level=4, style={'margin': '0'}),
                fac.AntdTooltip(
                    fac.AntdIcon(
                        id='workspace-tour-icon',
                        icon='pi-info',
                        style={"cursor": "pointer", 'paddingLeft': '10px'},
                        **{'aria-label': 'Show tutorial'}
                    ),
                    title='Show tutorial'
                )
            ],
            align='center'
        ),
        
        # --- Configuration Section ---
        fac.AntdFlex(
            [
                fac.AntdText("Current Data Directory:", strong=True),
                fac.AntdTooltip(
                    title="This is the global root directory where all your workspaces and data are stored.",
                    children=fac.AntdIcon(icon="antd-question-circle", style={"color": "#555", "marginRight": "2.5px"})
                ),
                fac.AntdText("Loading...", id="ws-current-data-dir-text", style={"marginRight": "10px", "fontWeight": "bold"}),
                fac.AntdTooltip(
                    fac.AntdButton("Change directory", id="ws-change-data-dir-btn", size="small"),
                    title="Select a different folder on your computer to store your MINT workspaces and data.",
                    placement="bottom"
                )
            ],
            style={"marginBottom": "1px", "marginTop": "100px"},
            gap=10,
            align="center"
        ),
        # -----------------------------
        # -----------------------------

        html.Div([
            fac.AntdTable(
                id='ws-table',
                columns=[
                    {'title': 'Name', 'dataIndex': 'name', 'align': 'left', 'width': '30%'},
                    {'title': 'Description', 'dataIndex': 'description', 'align': 'left', 'editable': True,
                     'width': '50%'},
                    {'title': 'Created at', 'dataIndex': 'created_at', 'align': 'center', 'width': '10%'},
                    {'title': 'Last Activity', 'dataIndex': 'last_activity', 'align': 'center', 'width': '10%'},
                ],
                filterOptions={
                    'name': {'filterSearch': True},
                },
                sortOptions={'sortDataIndexes': ['name', 'last_activity', 'created_at']},
                footer=[
                    fac.AntdFlex([
                        fac.AntdSpace([
                            fac.AntdTooltip(
                                fac.AntdButton('Create Workspace', id='ws-create', icon=fac.AntdIcon(icon='antd-plus')),
                                title="Create a new, empty workspace.",
                                placement="top"
                            ),
                            fac.AntdTooltip(
                                fac.AntdButton('Import Database', id='ws-import-db', icon=fac.AntdIcon(icon='antd-import')),
                                title="Import an existing MINT database file as a new workspace.",
                                placement="top"
                            ),
                        ]),
                        fac.AntdTooltip(
                            fac.AntdButton('Delete Workspace', id='ws-delete', danger=True, icon=fac.AntdIcon(icon='antd-minus')),
                            title="Permanently delete the selected workspace and all its data.",
                            placement="top"
                        )],
                        justify='space-between'
                    )
                ],
                pagination={'pageSize': 10, 'hideOnSinglePage': True},
                locale='en-us',
                rowSelectionType='radio',
                size='small',
            )],
            style={"marginTop": "2rem", "maxHeight": "calc(100vh - 280px)", "overflowY": "auto"},
        ),
        fac.AntdModal(
            [
                fac.AntdForm(
                    [
                        fac.AntdFormItem(
                            fac.AntdInput(id='ws-create-input-name', placeholder='New workspace name', value=None),
                            label='Name:',
                            hasFeedback=True,
                            id='ws-create-form-item'
                        ),
                        fac.AntdFormItem(
                            fac.AntdInput(id='ws-create-input-description', placeholder='Project description.',
                                          value=None, mode='text-area'),
                            label='Description:',
                        ),
                    ],
                ),
            ],
            title='Create Workspace',
            id='ws-create-modal',
            renderFooter=True,
            okText='Create',
            locale='en-us',
            okButtonProps={
                'disabled': True
            }
        ),
        fac.AntdModal([
            html.Div(fac.AntdText("This will delete all files and results in the selected workspace.")),

            html.Div(fac.AntdText('Are you sure you want to delete this workspace?', strong=True))],
            title="Delete Workspace",
            id="ws-delete-modal",
                okText="Delete",
                renderFooter=True,
                locale='en-us',
                okButtonProps={
                    'danger': True
                }
        ),
        
        # --- Change Data Directory Modal ---
        fac.AntdModal(
            [
                fac.AntdText("Enter the absolute path for the new Global Data Directory."),
                fac.AntdAlert(
                    message="Warning: Changing this will reload the application with the new directory. Existing workspaces in the old directory will remain there but won't be visible until you switch back.",
                    type="warning",
                    showIcon=True,
                    style={"marginBottom": "10px", "marginTop": "10px"}
                ),
                fac.AntdForm(
                    [
                        fac.AntdFormItem(
                            fac.AntdInput(id='ws-input-data-dir', placeholder='/absolute/path/to/MINT', value=None),
                            label='New Path:',
                            hasFeedback=True,
                            id='ws-data-dir-form-item'
                        ),
                    ]
                )
            ],
            title='Change Global Data Directory',
            id='ws-change-data-dir-modal',
            renderFooter=True,
            okText='Save & Reload',
            locale='en-us',
        ),
        # -----------------------------------

        # --- Import Database Modal ---
        fac.AntdModal(
            [
                fac.AntdAlert(
                    message="Import an existing MINT database file to create a new workspace.",
                    type="info",
                    showIcon=True,
                    style={"marginBottom": "15px"}
                ),
                # Path input with browse toggle
                fac.AntdSpace(
                    [
                        fac.AntdText("Database Path:", strong=True),
                        fac.AntdInput(
                            id='ws-import-db-path',
                            placeholder='/path/to/database.db',
                            value=None,
                            style={'flex': 1},
                            addonAfter=fac.AntdTooltip(
                                fac.AntdIcon(
                                    id='ws-import-db-browse-toggle',
                                    icon='antd-folder-open',
                                    style={'cursor': 'pointer'}
                                ),
                                title='Click to browse for .db files'
                            )
                        ),
                        fac.AntdTooltip(
                            fac.AntdIcon(icon='antd-question-circle', style={'color': '#555', 'fontSize': '16px'}),
                            title='Enter the path to an existing MINT database file (.db) or click the folder icon to browse. This will copy the database and create a new workspace.'
                        ),
                    ],
                    style={'width': '100%', 'marginBottom': '10px'}
                ),
                # File browser section (toggle via icon click)
                html.Div(
                    [
                        # Current path + Go
                        fac.AntdSpace(
                            [
                                fac.AntdInput(
                                    id='ws-import-db-browser-path',
                                    value=str(Path.home()),
                                    size='small',
                                    style={'flex': 1}
                                ),
                                fac.AntdButton('Go', id='ws-import-db-browser-go', size='small'),
                            ],
                            style={'width': '100%', 'marginBottom': '8px'}
                        ),
                        # Simple scrollable list
                        html.Div(
                            id='ws-import-db-browser-list',
                            style={
                                'maxHeight': '200px',
                                'overflowY': 'auto',
                                'border': '1px solid #d9d9d9',
                                'borderRadius': '4px',
                                'padding': '4px'
                            }
                        ),
                    ],
                    id='ws-import-db-browser-section',
                    style={'display': 'none', 'marginBottom': '10px', 'padding': '8px', 'background': '#fafafa', 'borderRadius': '4px'}
                ),
                # Stats
                html.Div(id='ws-import-db-stats', style={'marginBottom': '10px'}),
                # Workspace name
                fac.AntdSpace(
                    [
                        fac.AntdText("Workspace Name:", strong=True),
                        fac.AntdInput(id='ws-import-db-name', placeholder='Workspace name', value=None, style={'flex': 1}),
                    ],
                    style={'width': '100%'}
                ),
                html.Div(id='ws-import-db-name-error', style={'color': 'red', 'fontSize': '12px', 'marginTop': '4px'}),
                # Hidden stores
                dcc.Store(id='ws-import-db-current-dir', data=str(Path.home())),
            ],
            title='Import Database',
            id='ws-import-db-modal',
            width=550,
            renderFooter=True,
            okText='Import',
            locale='en-us',
            okButtonProps={'disabled': True}
        ),
        # -----------------------------------

        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'Use this short tutorial to pick, create, and clean up workspaces.',
                },
                {
                    'title': 'Pick a workspace',
                    'description': 'Click a row to load the workspace; sort or filter the table if you have many.',
                    'targetSelector': '#ws-table'
                },
                {
                    'title': 'Create a new one',
                    'description': 'Use “Create Workspace” to start a fresh project with its own files/results.',
                    'targetSelector': '#ws-create'
                },
                {
                    'title': 'Import Database',
                    'description': 'Import an existing MINT database file to create a new workspace.',
                    'targetSelector': '#ws-import-db'
                },
                {
                    'title': 'Clean up',
                    'description': 'Delete the selected workspace (and its files/results) when you no longer need it.',
                    'targetSelector': '#ws-delete'
                },
                {
                    'title': 'Tip',
                    'description': 'Changes are saved automatically after you switch workspaces.',
                    'targetSelector': '#ws-table'
                },
            ],
            id='workspace-tour',
            open=False,
            current=0,
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Click here for help',
                    # 'description': 'Click the info icon to open a quick tour of Workspaces.',
                    'targetSelector': '#workspace-tour-icon',
                },
            ],
            mask=False,
            placement='rightTop',
            open=False,
            current=0,
            id='workspace-tour-hint',
            className='targets-tour-hint',
            style={
                'background': '#ffffff',
                'border': '0.5px solid #1677ff',
                'boxShadow': '0 6px 16px rgba(0,0,0,0.15), 0 0 0 1px rgba(22,119,255,0.2)',
                'opacity': 1,
            },
        ),
        dcc.Store(id='workspace-tour-hint-store', data={'open': True}, storage_type='session'),
    ]
)

_outputs = html.Div(
    id="ws-outputs",
    children=[

    ],
)



def _create_ws_input_validation(value, tmpdir):
    if value is None:
        raise PreventUpdate

    if not tmpdir:
        okButtonProps = {'disabled': True}
        validateStatus = 'error'
        help = 'Workspace path not available'
    elif value is not None and bool(pattern.match(value)):
        with duckdb_connection_mint(tmpdir) as mint_conn:
            if mint_conn is None:
                okButtonProps = {'disabled': True}
                validateStatus = 'error'
                help = 'Cannot open workspace database'
            else:
                ws_df = mint_conn.execute("SELECT * FROM workspaces WHERE name = ?", (value,)).df()
                if ws_df.empty:
                    okButtonProps = {'disabled': False}
                    validateStatus = 'success'
                    help = None
                else:
                    okButtonProps = {'disabled': True}
                    validateStatus = 'error'
                    help = 'Workspace already exists!'
    else:
        okButtonProps = {'disabled': True}
        validateStatus = 'error'
        help = 'Workspace name can only contain: a-z, A-Z, 0-9 and _'
    return validateStatus, help, okButtonProps


def _create_workspace(okCounts, tmpdir, ws_name, ws_description):
    if not okCounts:
        raise PreventUpdate
    if not tmpdir:
        raise PreventUpdate
    with duckdb_connection_mint(tmpdir) as mint_conn:
        if mint_conn is None:
            raise PreventUpdate
        previous_active = mint_conn.execute("SELECT key FROM workspaces WHERE active = true").fetchone()

        key = mint_conn.execute("INSERT INTO workspaces (name, description, active, created_at, last_activity) "
                                "VALUES (?, ?, true, NOW(), NOW()) RETURNING key", (ws_name, ws_description)).fetchone()
        if previous_active:
            mint_conn.execute("UPDATE workspaces SET active = false WHERE key = ?", (previous_active[0],))
        if key:
            ws_path = Path(tmpdir, 'workspaces', str(key[0]))
            ws_path.mkdir(parents=True, exist_ok=True)
            # Note: "Created workspace" is logged after activation in _load_workspace_directories_and_tables

    return 'create', None, None, None


def _delete_workspace(okCounts, tmpdir, selectedRowKeys):
    if not okCounts:
        raise PreventUpdate
    if not tmpdir or not selectedRowKeys:
        raise PreventUpdate

    ws_key = selectedRowKeys[0]

    with duckdb_connection_mint(tmpdir) as mint_conn:
        try:
            mint_conn.execute("BEGIN")
            ws_row = mint_conn.execute(
                "SELECT name FROM workspaces WHERE key = ?",
                (ws_key,)
            ).fetchone()
            if not ws_row:
                mint_conn.execute("ROLLBACK")
                return fac.AntdNotification(
                    message="Workspace not found",
                    type="error",
                    duration=4,
                    placement='bottom',
                    showProgress=True,
                    stack=True
                ), {'type': 'delete', 'status': 'error'}

            next_active = mint_conn.execute(
                "SELECT key FROM workspaces WHERE key != ? ORDER BY last_activity DESC LIMIT 1",
                (ws_key,)
            ).fetchone()

            ws_path = Path(tmpdir, 'workspaces', str(ws_key))
            deactivate_workspace_logging()

            def _onerror(func, p, exc_info):
                import os
                try:
                    os.chmod(p, 0o700)
                    func(p)
                except Exception:
                    raise

            try:
                shutil.rmtree(ws_path, onerror=_onerror)
            except Exception as fs_err:
                mint_conn.execute("ROLLBACK")
                return fac.AntdNotification(
                    message="Failed to delete workspace files",
                    description=str(fs_err),
                    type="error",
                    duration=5,
                    placement='bottom',
                    showProgress=True,
                    stack=True
                ), {'type': 'delete', 'status': 'error'}

            mint_conn.execute("DELETE FROM workspaces WHERE key = ?", (ws_key,))
            if next_active:
                mint_conn.execute("UPDATE workspaces SET active = true WHERE key = ?", (next_active[0],))
            mint_conn.execute("COMMIT")

            ws_name = ws_row[0]
            logger.info(f"Deleted workspace: {ws_name} (key: {ws_key})")
            return fac.AntdNotification(message=f"Workspace '{ws_name}' deleted",
                                        type="success",
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        stack=True), {'type': 'delete', 'status': 'success'}
        except Exception as e:
            logger.error(f"Error deleting workspace {ws_key}: {e}", exc_info=True)
            mint_conn.execute("ROLLBACK")
            raise

    return dash.no_update, {'type': 'delete', 'status': 'error'}


def _ws_activate(selectedRowKeys, tmpdir, ws_action):
    if not selectedRowKeys:
        return dash.no_update, '', '', None

    ws_key = selectedRowKeys[0]

    with duckdb_connection_mint(tmpdir) as mint_conn:
        # Check if this workspace is already active
        is_active = mint_conn.execute("SELECT active FROM workspaces WHERE key = ?", (ws_key,)).fetchone()
        already_active = is_active and is_active[0]

        if already_active:
             # Just update activity time
            mint_conn.execute("UPDATE workspaces SET last_activity = NOW() WHERE key = ?", (ws_key,))
            name = mint_conn.execute("SELECT name FROM workspaces WHERE key = ?", (ws_key,)).fetchone()
            notification = dash.no_update
            is_new_activation = False
        else:
            # Full activation switch
            mint_conn.execute("UPDATE workspaces SET active = false WHERE key != ?", (ws_key,))
            name = mint_conn.execute(
                "UPDATE workspaces SET active = true, last_activity = NOW() WHERE key = ? RETURNING name",
                (ws_key,)
            ).fetchone()
            is_new_activation = True
            
            if name:
                ws_name = name[0]
                # Note: Log message moved to after activate_workspace_logging
                notification = fac.AntdNotification(message=f"Workspace '{ws_name}' activated",
                                                    type="success",
                                                    duration=3,
                                                    placement='bottom',
                                                    showProgress=True,
                                                    stack=True)
            else:
                 notification = dash.no_update

        if name:
            ws_name = name[0]
            wdir = Path(tmpdir, 'workspaces', ws_key)
        else:
            return dash.no_update, '', '', None

    # This will now be silent if the handler is already attached (idempotent)
    activate_workspace_logging(wdir, workspace_name=ws_name)
    
    # Log activation AFTER the handler is switched so it goes to the correct workspace's log
    if is_new_activation:
        # Check if this is a newly created workspace (ws_action came from create)
        if ws_action and ws_action.get('type') == 'create':
            logger.info(f"Created and activated workspace: {ws_name} at {wdir}")
        else:
            logger.info(f"Switched to workspace: {ws_name} (key: {ws_key})")

    return notification, ws_name, wdir.as_posix(), wdir.as_posix()


def _save_ws_table_on_edit(row_edited, column_edited, tmpdir):
    ctx = dash.callback_context
    # if not ctx.triggered:
    #     raise PreventUpdate

    if row_edited is None or column_edited is None:
        raise PreventUpdate

    if not tmpdir:
        raise PreventUpdate

    allowed_columns = {"description"}
    if column_edited not in allowed_columns:
        return fac.AntdNotification(
            message="Edit not allowed",
            description=f"Column '{column_edited}' cannot be edited.",
            type="error",
            duration=4,
            placement='bottom',
            showProgress=True,
            stack=True
        ), dash.no_update

    ws_key = row_edited.get('key')
    if not ws_key:
        raise PreventUpdate

    logger.info(f"Updating workspace {ws_key}: {column_edited} = {row_edited[column_edited]}")

    try:
        with duckdb_connection_mint(tmpdir, workspace=ws_key) as mint_conn:
            if mint_conn is None:
                raise PreventUpdate
            query = f"UPDATE workspaces SET {column_edited} = ?, last_activity = NOW() WHERE key = ?"
            mint_conn.execute(query, [row_edited[column_edited], ws_key])
        return fac.AntdNotification(message="Changes saved",
                                    type="success",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    ), {'type': 'edit'}
    except Exception as e:
        logger.error(f"Failed to save edit for workspace {ws_key}: {e}", exc_info=True)
        return fac.AntdNotification(message="Failed to save changes",
                                    description=f"Error: {e}",
                                    type="error",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    ), dash.no_update


def _save_new_data_dir(okCounts, new_path):
    if not okCounts:
        raise PreventUpdate
    
    if not new_path or not new_path.strip():
         return dash.no_update, fac.AntdNotification(message="Invalid path", type="error", placement="bottom"), True
    
    new_path = os.path.expanduser(new_path)
    new_path_obj = Path(new_path)

    if not new_path_obj.is_absolute():
        return dash.no_update, fac.AntdNotification(message="Path must be absolute", type="error", placement="bottom"), True
    
    # Try to create directory if it doesn't exist to verify permissions
    try:
        new_path_obj.mkdir(parents=True, exist_ok=True)
        # Create .cache inside
        (new_path_obj / ".cache").mkdir(exist_ok=True)
    except OSError as e:
         return dash.no_update, fac.AntdNotification(message=f"Permission denied or invalid path: {e}", type="error", placement="bottom"), True

    # Update Config
    config_path = os.path.expanduser(os.environ.get("MINT_CONFIG_PATH") or "~/.mint_config.json")
    try:
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        else:
            cfg = {}
        
        cfg["data_dir"] = str(new_path_obj)
        
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            
    except Exception as e:
         return dash.no_update, fac.AntdNotification(message=f"Failed to update config file: {e}", type="error", placement="bottom"), True

    # Update Environment Variable for this session (best effort)
    os.environ["MINT_DATA_DIR"] = str(new_path_obj)
    
    # Update global TMPDIR in app.py to ensure page refreshes pick up the new path
    try:
        import ms_mint_app.app as app_module
        app_module.TMPDIR = new_path_obj
        app_module.CACHEDIR = new_path_obj / ".cache"
        logger.info(f"Updated app.TMPDIR to {app_module.TMPDIR}")
    except Exception as e:
        logger.warning(f"Failed to update app.TMPDIR: {e}")

    new_user_path = new_path_obj / "Local"
    new_user_path.mkdir(parents=True, exist_ok=True)
    
    with duckdb_connection_mint(str(new_user_path)) as mint_conn:
        logger.info(f"Initialized DB in {new_user_path}")

    return str(new_user_path), fac.AntdNotification(message="Global Data Directory updated successfully!", type="success", placement="bottom"), False


def callbacks(app, fsc, cache):
    @app.callback(
        Output('workspace-tour', 'current'),
        Output('workspace-tour', 'open'),
        Input('workspace-tour-icon', 'nClicks'),
        prevent_initial_call=True,
    )
    def workspace_tour_open(n_clicks):
        return 0, True

    @app.callback(
        Output('workspace-tour-hint', 'open'),
        Output('workspace-tour-hint', 'current'),
        Input('workspace-tour-hint-store', 'data'),
    )
    def workspace_hint_sync(store_data):
        if not store_data:
            raise PreventUpdate
        return store_data.get('open', True), 0

    @app.callback(
        Output('workspace-tour-hint-store', 'data'),

        Input('workspace-tour-icon', 'nClicks'),
        State('workspace-tour-hint-store', 'data'),
        prevent_initial_call=True,
    )
    def workspace_hide_hint(n_clicks, store_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'workspace-tour-icon':
            return {'open': False}

        return store_data or {'open': True}

    @app.callback(
        Output('ws-create-form-item', 'validateStatus'),
        Output('ws-create-form-item', 'help'),
        Output('ws-create-modal', 'okButtonProps'),

        Input('ws-create-input-name', 'value'),
        State("tmpdir", "data"),
        prevent_initial_call=True
    )
    def create_ws_input_validation(value, tmpdir):
        return _create_ws_input_validation(value, tmpdir)

    @app.callback(
        Output('ws-action-store', 'data', allow_duplicate=True),
        Output('ws-create-input-name', 'value'),
        Output('ws-create-input-description', 'value'),
        Output('ws-create-form-item', 'validateStatus', allow_duplicate=True),

        Input('ws-create-modal', 'okCounts'),
        State("tmpdir", "data"),
        State('ws-create-input-name', 'value'),
        State('ws-create-input-description', 'value'),
        prevent_initial_call=True
    )
    def create_workspace(okCounts, tmpdir, ws_name, ws_description):
        return _create_workspace(okCounts, tmpdir, ws_name, ws_description)

    @app.callback(
        Output('ws-create-modal', 'visible'),
        Input("ws-create", "nClicks")
    )
    def create_workspace_modal(nClicks):
        if nClicks is None:
            raise PreventUpdate
        return True

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('ws-action-store', 'data', allow_duplicate=True),

        Input('ws-delete-modal', 'okCounts'),
        State("tmpdir", "data"),
        State('ws-table', 'selectedRowKeys'),
        prevent_initial_call=True
    )
    def delete_workspace(okCounts, tmpdir, selectedRowKeys):
        return _delete_workspace(okCounts, tmpdir, selectedRowKeys)

    @app.callback(
        Output('ws-delete-modal', 'visible'),
        Input("ws-delete", "nClicks")
    )
    def delete_workspace_modal(nClicks):
        if nClicks is None:
            raise PreventUpdate
        return True

    @app.callback(
        Output("ws-table", "data"),
        Output("ws-table", "expandedRowKeyToContent"),
        Output("ws-table", "selectedRowKeys"),

        Input('section-context', 'data'),
        Input('ws-action-store', 'data'),
        Input("tmpdir", "data"), # Changed from State to Input to auto-update
    )
    def ws_table(section_context, ws_action, tmpdir):

        if section_context and  section_context['page'] != 'Workspaces':
            raise PreventUpdate

        with duckdb_connection_mint(tmpdir) as mint_conn:
            data = mint_conn.execute("SELECT * FROM workspaces ORDER BY last_activity DESC").df()
            logger.debug(f"Loaded workspace table data: {len(data)} rows")

            if not data.empty:
                data["key"] = data["key"].astype(str)

            cols = ['created_at', 'last_activity']
            data[cols] = data[cols].apply(lambda col: col.dt.strftime("%y-%m-%d %H:%M:%S"))

            row_content = mint_conn.execute("SELECT key FROM workspaces").df()
            if not row_content.empty:
                row_content["key"] = row_content["key"].astype(str)

            def row_comp(key):
                _path = Path(tmpdir, 'workspaces', str(key))
                path_info = html.Div(
                    [
                        fac.AntdText('Workspace path:', strong=True, locale='en-us', style={'marginRight': '10px'}),
                        fac.AntdText(_path.as_posix(), copyable=True, locale='en-us')
                    ],
                    style={'minWidth': '200px', 'flexGrow': 1, 'padding': '10px'}
                )

                try:
                    # Avoid bumping last_activity just for rendering the preview table
                    with duckdb_connection(_path, register_activity=False) as conn:
                        if conn is None:
                            # Database is locked - return placeholder
                            return fac.AntdFlex(
                                [
                                    path_info,
                                    fac.AntdText("Database busy...", type='secondary')
                                ],
                                wrap=True
                            )
                        summary = conn.execute("""
                                               SELECT * FROM (
                                                   SELECT 'samples' AS table_name, COUNT(*) AS rows FROM samples
                                                   UNION ALL
                                                   SELECT 'ms1_data' AS table_name, COUNT(*) AS rows FROM ms1_data
                                                   UNION ALL
                                                   SELECT 'ms2_data' AS table_name, COUNT(*) AS rows FROM ms2_data
                                                   UNION ALL
                                                   SELECT 'targets' AS table_name, COUNT(*) AS rows FROM targets
                                                   UNION ALL
                                                   SELECT 'chromatograms' AS table_name, COUNT(*) AS rows FROM chromatograms
                                                   UNION ALL
                                                   SELECT 'results' AS table_name, COUNT(*) AS rows FROM results
                                               ) t
                                               ORDER BY CASE table_name
                                                            WHEN 'samples' THEN 1
                                                            WHEN 'ms1_data' THEN 2
                                                            WHEN 'ms2_data' THEN 3
                                                            WHEN 'targets' THEN 4
                                                            WHEN 'chromatograms' THEN 5
                                                            WHEN 'results' THEN 6
                                                            ELSE 7
                                                            END
                                               """).df()
                        if not summary.empty:
                            summary['rows'] = summary['rows'].apply(
                                lambda x: f"{int(x):,}" if pd.notna(x) else ""
                            )
                        db_info = fac.AntdTable(
                            columns=[
                                {'title': 'Table name', 'dataIndex': 'table_name', 'align': 'left', 'width': '50%'},
                                {'title': 'Rows', 'dataIndex': 'rows', 'align': 'center', 'width': '50%'},
                            ],
                            data=summary.to_dict('records'),
                            pagination=False,
                            locale='en-us',
                            size='small',
                            style={'minWidth': '200px', 'flexGrow': 1}
                        )
                    return fac.AntdFlex(
                        [
                            path_info,
                            db_info
                        ],
                        wrap=True
                    )
                except Exception as e:
                    # Check for database corruption
                    from ..duckdb_manager import DatabaseCorruptionError
                    if isinstance(e, DatabaseCorruptionError) or "Corrupt database file" in str(e):
                        return fac.AntdFlex(
                            [
                                path_info,
                                fac.AntdAlert(
                                    message="⚠️ Database Corrupted",
                                    description="This workspace's database is corrupted. Please delete this workspace and restore from backup or recreate it.",
                                    type='error',
                                    showIcon=True,
                                    style={'maxWidth': '400px'}
                                )
                            ],
                            wrap=True
                        )
                    # Other errors - show generic message
                    return fac.AntdFlex(
                        [
                            path_info,
                            fac.AntdText(f"Error loading workspace: {str(e)[:50]}", type='danger')
                        ],
                        wrap=True
                    )

            row_content['content'] = row_content['key'].apply(row_comp)
            selectedRowKeys = mint_conn.execute("SELECT key FROM workspaces WHERE active = true").fetchone()

            sk = [str(selectedRowKeys[0])] if selectedRowKeys else None
        return data.to_dict('records'), row_content.to_dict('records'), sk

    @app.callback(
        Output('ws-action-store', 'data', allow_duplicate=True),
        Input("ws-table", "data"),
        prevent_initial_call=True,
    )
    def reset_ws_action_store(_data):
        # Clear the action flag after the table refreshes so ordering/notifications return to normal
        return None

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output("ws-wdir-name-text", "children"),
        Output("ws-wdir-name", "text"),
        Output("wdir", "data"),
        Output("workspace-status", "data"),

        Input("ws-table", "selectedRowKeys"),
        State("tmpdir", "data"),
        State("ws-action-store", "data"),
        prevent_initial_call=True
    )
    def ws_activate(selectedRowKeys, tmpdir, ws_action):
        notification, ws_name, wdir_path, wdir_path2 = _ws_activate(selectedRowKeys, tmpdir, ws_action)
        
        # Populate workspace status store with counts from the activated workspace
        workspace_status = {
            'ms_files_count': 0,
            'targets_count': 0,
            'chromatograms_count': 0,
            'selected_targets_count': 0,
            'optimization_samples_count': 0
        }
        
        if wdir_path:
            from ..duckdb_manager import duckdb_connection, DatabaseCorruptionError
            try:
                with duckdb_connection(wdir_path) as conn:
                    if conn is not None:
                        counts = conn.execute("""
                            SELECT 
                                (SELECT COUNT(*) FROM samples) as ms_files,
                                (SELECT COUNT(*) FROM targets) as targets,
                                (SELECT COUNT(*) FROM chromatograms) as chroms,
                                (SELECT COUNT(*) FROM chromatograms WHERE ms_type = 'ms1') as chroms_ms1,
                                (SELECT COUNT(*) FROM chromatograms WHERE ms_type = 'ms2') as chroms_ms2,
                                (SELECT COUNT(*) FROM targets WHERE peak_selection = TRUE) as selected_targets,
                                (SELECT COUNT(*) FROM samples WHERE use_for_optimization = TRUE) as opt_samples,
                                (SELECT COUNT(*) FROM results) as results
                        """).fetchone()
                        if counts:
                            n_cpus = cpu_count()
                            default_cpus = max(1, n_cpus // 2)
                            ram_avail = psutil.virtual_memory().available / (1024 ** 3)
                            default_ram = round(min(float(default_cpus), ram_avail), 1)

                            workspace_status = {
                                'ms_files_count': counts[0] or 0,
                                'targets_count': counts[1] or 0,
                                'chromatograms_count': counts[2] or 0,
                                'chroms_ms1_count': counts[3] or 0,
                                'chroms_ms2_count': counts[4] or 0,
                                'selected_targets_count': counts[5] or 0,
                                'optimization_samples_count': counts[6] or 0,
                                'results_count': counts[7] or 0,
                                'n_cpus': n_cpus,
                                'default_cpus': default_cpus,
                                'ram_avail': round(ram_avail, 1),
                                'default_ram': default_ram
                            }
                            logger.info(f"workspace-status populated: {workspace_status}")
            except DatabaseCorruptionError as e:
                logger.error(f"Database corruption detected during activation: {e}")
                notification = fac.AntdNotification(
                    message="⚠️ Database Corrupted",
                    description="This workspace's database is corrupted. Please delete and restore from backup or recreate it.",
                    type="error",
                    duration=10,
                    placement='bottom',
                    showProgress=True,
                )
                # Mark status as corrupted
                workspace_status['corrupted'] = True
            except Exception as e:
                logger.error(f"Error loading workspace status: {e}")
        
        return notification, ws_name, wdir_path, wdir_path2, workspace_status

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output('ws-action-store', 'data', allow_duplicate=True),

        Input("ws-table", "recentlyChangedRow"),
        State("ws-table", "recentlyChangedColumn"),
        State("tmpdir", "data"),
        prevent_initial_call=True,
    )
    def save_ws_table_on_edit(row_edited, column_edited, tmpdir):
        return _save_ws_table_on_edit(row_edited, column_edited, tmpdir)

    @app.callback(
        Output("ws-change-data-dir-modal", "visible"),
        Input("ws-change-data-dir-btn", "nClicks"),
        prevent_initial_call=True
    )
    def open_change_data_dir_modal(nClicks):
        return True

    @app.callback(
        Output("tmpdir", "data", allow_duplicate=True),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("ws-change-data-dir-modal", "visible", allow_duplicate=True),
        
        Input("ws-change-data-dir-modal", "okCounts"),
        State("ws-input-data-dir", "value"),
        prevent_initial_call=True
    )
    def save_new_data_dir(okCounts, new_path):
        return _save_new_data_dir(okCounts, new_path)

    @app.callback(
        Output("ws-current-data-dir-text", "children"),
        Input("tmpdir", "data")
    )
    def update_data_dir_display(tmpdir):
        logger.debug(f"Updating Display with tmpdir: {tmpdir}")
        if not tmpdir:
            return "Unknown"
        # Strip /Local or /User/* suffix to show the root
        p = Path(tmpdir)
        if p.name == "Local":
            return str(p.parent)
        if p.parent.name == "User":
            return str(p.parent.parent)
        return str(p)

    # --- Import Database Callbacks ---
    
    @app.callback(
        Output("ws-import-db-modal", "visible"),
        Input("ws-import-db", "nClicks"),
        prevent_initial_call=True
    )
    def open_import_db_modal(nClicks):
        if nClicks is None:
            raise PreventUpdate
        return True

    def _build_file_list(dir_path: Path):
        """Build a simple list of clickable folders/files."""
        items = []
        
        # Parent (..)
        if dir_path.parent != dir_path:
            items.append(
                html.Div(
                    fac.AntdText("📂 ..", style={'cursor': 'pointer'}),
                    id={'type': 'import-browser-item', 'path': str(dir_path.parent), 'isfile': 'no'},
                    style={'padding': '2px 4px', 'cursor': 'pointer'},
                    className='browser-item'
                )
            )
        
        try:
            entries = list(dir_path.iterdir())
            folders = sorted([e for e in entries if e.is_dir() and not e.name.startswith('.')], 
                           key=lambda x: x.name.lower())
            files = sorted([e for e in entries if e.is_file() and e.suffix.lower() in ['.db', '.duckdb'] 
                          and not e.name.startswith('.')], key=lambda x: x.name.lower())
            
            for f in folders:
                items.append(
                    html.Div(
                        fac.AntdText(f"📂 {f.name}", style={'cursor': 'pointer'}),
                        id={'type': 'import-browser-item', 'path': str(f), 'isfile': 'no'},
                        style={'padding': '2px 4px', 'cursor': 'pointer'},
                        className='browser-item'
                    )
                )
            
            for f in files:
                try:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    items.append(
                        html.Div(
                            fac.AntdText(f"💾 {f.name} ({size_mb:.1f} MB)", style={'cursor': 'pointer', 'color': '#1890ff'}),
                            id={'type': 'import-browser-item', 'path': str(f), 'isfile': 'yes'},
                            style={'padding': '2px 4px', 'cursor': 'pointer', 'background': '#f0f8ff'},
                            className='browser-item'
                        )
                    )
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            items.append(fac.AntdText("Cannot access this directory", type='secondary'))
        
        return items if items else [fac.AntdText("No files found", type='secondary')]

    @app.callback(
        Output('ws-import-db-browser-section', 'style'),
        Output('ws-import-db-browser-list', 'children'),
        Output('ws-import-db-browser-path', 'value'),
        Output('ws-import-db-current-dir', 'data'),
        
        Input('ws-import-db-browse-toggle', 'nClicks'),
        Input('ws-import-db-browser-go', 'nClicks'),
        
        State('ws-import-db-browser-section', 'style'),
        State('ws-import-db-browser-path', 'value'),
        State('ws-import-db-current-dir', 'data'),
        prevent_initial_call=True
    )
    def toggle_and_update_browser(toggle_clicks, go_clicks, current_style, path_input, current_dir):
        """Toggle browser visibility and update file list."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger == 'ws-import-db-browse-toggle':
            # Toggle visibility
            is_visible = current_style.get('display', 'none') != 'none'
            if is_visible:
                # Hide
                return {'display': 'none', 'marginBottom': '10px', 'padding': '8px', 'background': '#fafafa', 'borderRadius': '4px'}, dash.no_update, dash.no_update, dash.no_update
            else:
                # Show and populate
                dir_path = Path(current_dir) if current_dir else Path.home()
                return {'display': 'block', 'marginBottom': '10px', 'padding': '8px', 'background': '#fafafa', 'borderRadius': '4px'}, _build_file_list(dir_path), str(dir_path), str(dir_path)
        
        elif trigger == 'ws-import-db-browser-go':
            # Go button - navigate to path
            dir_path = Path(path_input) if path_input else Path.home()
            if not dir_path.exists():
                dir_path = Path.home()
            if dir_path.is_file():
                dir_path = dir_path.parent
            return dash.no_update, _build_file_list(dir_path), str(dir_path), str(dir_path)
        
        raise PreventUpdate

    @app.callback(
        Output('ws-import-db-path', 'value'),
        Output('ws-import-db-browser-list', 'children', allow_duplicate=True),
        Output('ws-import-db-browser-path', 'value', allow_duplicate=True),
        Output('ws-import-db-current-dir', 'data', allow_duplicate=True),
        
        Input({'type': 'import-browser-item', 'path': ALL, 'isfile': ALL}, 'n_clicks'),
        State('ws-import-db-current-dir', 'data'),
        State('ws-import-db-path', 'value'),
        prevent_initial_call=True
    )
    def handle_browser_item_click(n_clicks, current_dir, current_path):
        """Handle clicking on a folder or file in the browser."""
        ctx = dash.callback_context
        if not ctx.triggered or not any(n_clicks):
            raise PreventUpdate
        
        # Find which item was clicked
        triggered = ctx.triggered[0]
        prop_id = triggered['prop_id']
        
        import json
        id_str = prop_id.rsplit('.', 1)[0]
        item_id = json.loads(id_str)
        
        clicked_path = Path(item_id['path'])
        is_file = item_id['isfile'] == 'yes'
        
        if is_file:
            # File selected - set path, keep browser open
            return str(clicked_path), dash.no_update, dash.no_update, dash.no_update
        else:
            # Folder - navigate into it
            return dash.no_update, _build_file_list(clicked_path), str(clicked_path), str(clicked_path)

    @app.callback(
        Output('ws-import-db-stats', 'children'),
        Output('ws-import-db-name', 'value'),
        
        Input('ws-import-db-path', 'value'),
        prevent_initial_call=True
    )
    def validate_and_show_stats(db_path):
        """Validate database and show stats."""
        if not db_path or not db_path.strip():
            return None, dash.no_update
        
        is_valid, error_msg, stats = validate_mint_database(db_path.strip())
        
        if not is_valid:
            return fac.AntdAlert(message=error_msg, type='error', showIcon=True), dash.no_update
        
        # Stats display
        stats_items = [fac.AntdTag(content=f"{t}: {c:,}", color='blue') for t, c in stats.items()]
        stats_display = fac.AntdSpace([
            fac.AntdTag(content="✓ Valid", color='green'),
            *stats_items
        ], wrap=True)
        
        # Suggest name
        suggested = Path(db_path).stem.replace('workspace_mint', 'imported').replace('.', '_')
        suggested = ''.join(c if c.isalnum() or c == '_' else '_' for c in suggested)[:30]
        
        return stats_display, suggested

    @app.callback(
        Output('ws-import-db-name-error', 'children'),
        Output('ws-import-db-modal', 'okButtonProps'),
        
        Input('ws-import-db-name', 'value'),
        State('ws-import-db-path', 'value'),
        State("tmpdir", "data"),
        prevent_initial_call=True
    )
    def validate_import_name(ws_name, db_path, tmpdir):
        """Validate workspace name and enable Import button."""
        if not ws_name or not db_path or not tmpdir:
            return None, {'disabled': True}
        
        # Check path is valid
        is_valid, _, _ = validate_mint_database(db_path.strip())
        if not is_valid:
            return None, {'disabled': True}
        
        # Check name pattern
        if not pattern.match(ws_name):
            return 'Name can only contain: a-z, A-Z, 0-9 and _', {'disabled': True}
        
        # Check duplicate
        with duckdb_connection_mint(tmpdir) as conn:
            if conn is None:
                return 'Database error', {'disabled': True}
            existing = conn.execute("SELECT COUNT(*) FROM workspaces WHERE name = ?", (ws_name,)).fetchone()[0]
            if existing > 0:
                return 'Workspace name already exists', {'disabled': True}
        
        return None, {'disabled': False}

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('ws-action-store', 'data', allow_duplicate=True),
        Output('ws-import-db-modal', 'visible', allow_duplicate=True),
        Output('ws-import-db-path', 'value', allow_duplicate=True),
        Output('ws-import-db-name', 'value', allow_duplicate=True),
        Output('ws-import-db-stats', 'children', allow_duplicate=True),

        Input('ws-import-db-modal', 'okCounts'),
        State('ws-import-db-path', 'value'),
        State('ws-import-db-name', 'value'),
        State("tmpdir", "data"),
        prevent_initial_call=True
    )
    def import_database(okCounts, db_path, ws_name, tmpdir):
        """Import database and create new workspace."""
        if not okCounts:
            raise PreventUpdate
        
        if not db_path or not ws_name or not tmpdir:
            return fac.AntdNotification(message="Invalid parameters", type="error", duration=4, placement='bottom'), dash.no_update, True, dash.no_update, dash.no_update, dash.no_update
        
        success, error_msg, workspace_key = import_database_as_workspace(db_path.strip(), ws_name.strip(), tmpdir)
        
        if not success:
            return fac.AntdNotification(message="Import failed", description=error_msg, type="error", duration=5, placement='bottom'), dash.no_update, True, dash.no_update, dash.no_update, dash.no_update
        
        logger.info(f"Successfully imported database from {db_path} as workspace '{ws_name}'")
        
        return (
            fac.AntdNotification(message=f"Workspace '{ws_name}' imported", type="success", duration=4, placement='bottom'),
            {'type': 'import', 'key': workspace_key},
            False, None, None, None,
        )


