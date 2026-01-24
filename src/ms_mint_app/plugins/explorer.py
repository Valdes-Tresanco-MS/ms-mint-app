import datetime
import json
import platform
from collections import Counter
from pathlib import Path

import dash
import feffery_antd_components as fac
import psutil
from dash import Output, Input, State, ALL, html, dcc
from dash.exceptions import PreventUpdate
import logging

from ms_mint_app.tools import process_ms_files, process_metadata, process_targets
from ms_mint_app.logging_setup import activate_workspace_logging

home_path = Path.home()

logger = logging.getLogger(__name__)


class FileExplorer:
    """
    FileExplores component. Creates a Modal to select files from the system.
    """

    def get_selected_tree_data(self, selected_files):
        """Generates the tree of selected files organized by folder."""
        tree_data = []

        # Group files by folder
        folders = {}
        for file_path in selected_files:
            path = Path(file_path)
            folder = path.parent.as_posix()
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(file_path)

        # Create tree nodes
        for folder, files in sorted(folders.items()):
            folder_node = {
                'key': folder,
                'title': f"{Path(folder).name} ({len(files)} files)",
                'icon': 'antd-folder',
                'checkable': True,
                'children': [
                    {
                        'key': file,
                        'title': Path(file).name,
                        'isLeaf': True,
                        'checkable': True,
                    }
                    for file in sorted(files)
                ]
            }
            tree_data.append(folder_node)

        return tree_data

    def outputs(self):
        return None

    def layout(self):
        return html.Div(
            [
                fac.AntdModal(
                    [
                        html.Div(
                            [
                                fac.AntdSpace(
                                    [
                                        html.Div(
                                            [
                                                # Breadcrumb for current path
                                                fac.AntdBreadcrumb(
                                                    id='current-path-modal',
                                                    items=[],
                                                    style={'margin': '10px 0', 'height': '24px'}
                                                ),
                                                # Table of files and directories
                                                html.Div(
                                                    [
                                                        fac.AntdTable(
                                                            id='file-table',
                                                            data=[],
                                                            columns=[
                                                                {
                                                                    'title': 'Name',
                                                                    'dataIndex': 'name',
                                                                    'align': 'left',
                                                                    'width': '50%',
                                                                    'renderOptions': {'renderType': 'link'},
                                                                },
                                                                {
                                                                    'title': 'Type',
                                                                    'dataIndex': 'type',
                                                                    'width': '12.5%',
                                                                },
                                                                {
                                                                    'title': 'Modified',
                                                                    'dataIndex': 'modified',
                                                                    'width': '30%',
                                                                },
                                                                {
                                                                    'title': 'Files',
                                                                    'dataIndex': 'file_count',
                                                                    'width': '12.5%',
                                                                },
                                                            ],
                                                            titlePopoverInfo={
                                                                'name': {
                                                                    'title': 'Name',
                                                                    'content': 'The name of the file or directory.',
                                                                },
                                                                'type': {
                                                                    'title': 'Type',
                                                                    'content': 'The type of the item (File or Folder).',
                                                                },
                                                                'modified': {
                                                                    'title': 'Modified',
                                                                    'content': 'The timestamp of the last modification.',
                                                                },
                                                                'file_count': {
                                                                    'title': 'Files',
                                                                    'content': 'The number of matching files in the directory.',
                                                                },
                                                            },
                                                            sortOptions={
                                                                'sortDataIndexes': ['modified', 'name', 'file_count']},
                                                            rowSelectionType='checkbox',
                                                            rowSelectionWidth=50,
                                                            size='small',
                                                            bordered=True,
                                                            pagination={
                                                                'pageSize': 50,
                                                                'showSizeChanger': True,
                                                                'pageSizeOptions': [20, 50, 100, 200],
                                                                'showQuickJumper': True,
                                                                'position': 'bottomCenter',
                                                                'size': 'small'
                                                            },
                                                            maxHeight=418,
                                                            showSorterTooltip=False,
                                                            locale='en-us',
                                                            enableCellClickListenColumns=['name']
                                                        )
                                                    ],
                                                    style={
                                                        'maxWidth': '800px', 'height': '524px'}
                                                ),
                                                fac.AntdFlex(
                                                    [
                                                        # CPUs configuration
                                                        fac.AntdForm(
                                                            [
                                                                fac.AntdFormItem(
                                                                    fac.AntdInputNumber(
                                                                        id='processing-cpu-input',
                                                                        defaultValue=max(1, psutil.cpu_count() // 2),
                                                                        min=1,
                                                                        max=psutil.cpu_count(),
                                                                        size='small',
                                                                    ),
                                                                    label='CPUs:',
                                                                    tooltip='Number of CPUs to use for processing in parallel',
                                                                ),
                                                            ],
                                                            layout='inline',
                                                            id='processing-cpu-form',
                                                            style={'marginRight': '10px'}
                                                        ),
                                                        # Filter by extension
                                                        fac.AntdSelect(
                                                            id='selected-files-extensions',
                                                            size="small",
                                                            mode="multiple",
                                                            placeholder='Filter by extension',
                                                            style={'flex': '1'},
                                                            locale="en-us",
                                                            allowClear=True,
                                                            disabled=True,
                                                        ),
                                                    ],
                                                    align='center',
                                                    style={'width': "100%", 'margin': '10px 0'},
                                                ),
                                            ],
                                        ),
                                        fac.AntdFlex(
                                            [
                                                fac.AntdFlex(
                                                    [
                                                        html.Strong("Selected files"),
                                                        fac.AntdSpace(
                                                            [
                                                                fac.AntdTooltip(
                                                                    fac.AntdButton(
                                                                        id='remove-marked-btn',
                                                                        size='small',
                                                                        danger=True,
                                                                        type='text',
                                                                        icon=fac.AntdIcon(
                                                                            icon='md-remove-circle-outline'),
                                                                        **{'aria-label': 'Remove selected files from list'},
                                                                    ),
                                                                    title='Remove selected files from list'
                                                                ),
                                                                fac.AntdTooltip(
                                                                    fac.AntdButton(
                                                                        id='clear-selection-btn',
                                                                        size='small',
                                                                        danger=True,
                                                                        type='primary',
                                                                        icon=fac.AntdIcon(icon='antd-delete'),
                                                                        **{'aria-label': 'Clear all selected files'},
                                                                    ),
                                                                    title='Clear all selected files'
                                                                ),
                                                            ],
                                                            size='small'
                                                        ),
                                                    ],
                                                    justify='space-between',
                                                    align='center',
                                                    style={'margin': '10px 0'}
                                                ),
                                                html.Div(
                                                    [
                                                        fac.AntdTree(
                                                            id='selected-files-tree',
                                                            treeData=[],
                                                            checkable=True,
                                                            showIcon=True,
                                                            showLine=True,
                                                            defaultExpandAll=True,
                                                            style={'display': 'none'},
                                                        ),
                                                        fac.AntdEmpty(
                                                            description='No files selected',
                                                            id='files-selection-empty',
                                                            locale='en-us',
                                                            image='simple',
                                                            style={'display': 'block'}
                                                        )
                                                    ],
                                                    style={
                                                        'minHeight': '524px',
                                                        'maxHeight': '524px',
                                                        'maxWidth': '340px',
                                                        'width': '340px',
                                                        'flex': '1',
                                                        'overflow': 'auto'
                                                    }
                                                ),
                                                # File counter
                                                html.Div(
                                                    'Total: 0 files selected',
                                                    id='selection-counter',
                                                    style={'marginTop': '10px', 'fontSize': '12px',
                                                           'color': '#444'}
                                                ),
                                            ],
                                            vertical=True,
                                        ),
                                    ],
                                    align='start',
                                ),

                            ],
                            id='selection-container',
                        ),
                        # Processing progress container
                        html.Div(
                            [
                                html.H4("Processing files..."),
                                fac.AntdText(
                                    id='ms-files-progress-stage',
                                    style={'marginBottom': '0.5rem'},
                                ),
                                fac.AntdProgress(
                                    id='sm-processing-progress',
                                    percent=0,
                                ),
                                fac.AntdText(
                                    id='ms-files-progress-detail',
                                    type='secondary',
                                    style={'marginTop': '0.5rem', 'marginBottom': '0.75rem'},
                                ),
                                fac.AntdButton(
                                    'Cancel',
                                    id='cancel-ms-processing',
                                    danger=True,
                                    style={
                                        'alignText': 'center',
                                    },
                                ),
                            ],
                            id='explorer-processing-progress-container',
                            style={'display': 'none', 'textAlign': 'center', 'padding': '20px'},
                        ),
                    ],
                    id="selection-modal",
                    title='Load Files',
                    width=1200,
                    renderFooter=True,
                    locale='en-us',
                    confirmAutoSpin=True,
                    loadingOkText='Processing Files...',
                    okClickClose=False,
                    closable=False,
                    maskClosable=False,
                    destroyOnClose=False,
                    okText="Process Files",
                    centered=True,
                    styles={'body': {'height': "70vh"}},
                ),
                dcc.Store(id="current-path-store", data=str(home_path)),
                dcc.Store(id="selected-files-store", data=[]),
                dcc.Store(id='processing-type-store', data={}),
                dcc.Store(id="processed-action-store"),
                dcc.Store(id='table-data-store', data=[]),
                dcc.Store(id='marked-for-removal', data=[]),
                dcc.Store(id='is-at-root', data=False),  # To check if we are at the drive level
            ]
        )

    def get_table_data(self, path, extensions, is_root=False):
        """Generates table data for the specified directory."""

        # If we are at the root of Windows, show the drives
        if is_root and platform.system() == 'Windows':
            import string
            table_data = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if Path(drive).exists():
                    try:
                         # counting files can fail due to permissions
                        count = len([f for ext in extensions for f in Path(drive).glob(f'*{ext}')])
                    except Exception:
                        count = '-'
                    
                    table_data.append({
                        'key': drive,
                        'name': {
                            'content': f"{letter}:",
                        },
                        'type': fac.AntdIcon(icon='antd-database'),
                        'file_count': count,
                        'path': drive,
                        'is_dir': True,
                    })
            return table_data

        # Normal navigation
        current_path = Path(path)

        def safe_iterdir(p):
            try:
                return list(p.iterdir())
            except (PermissionError, OSError):
                return []

        def safe_is_dir(p):
            try:
                return p.is_dir()
            except (PermissionError, OSError):
                return False

        items = safe_iterdir(current_path)
        table_data = []

        # Sort directories and files
        dirs = sorted([i for i in items if safe_is_dir(i) and not i.name.startswith('.')], key=lambda x: x.name)
        files = sorted([i for i in items if i.is_file() and i.suffix in extensions and not i.name.startswith('.')],
                       key=lambda x: x.name)

        # Add directories (show all, without checking content)
        for folder in dirs:
            folder_stats = folder.stat()

            table_data.append({
                'key': folder.as_posix(),
                'name': {
                    'content': folder.name,
                },
                'type': fac.AntdIcon(icon='antd-folder'),
                'file_count': len([f for ext in extensions for f in folder.glob(f'*{ext}')]) or '-',
                'modified': datetime.datetime.fromtimestamp(folder_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                'path': folder.as_posix(),
                'is_dir': True,
            })

        # Add files
        for file in files:
            file_stats = file.stat()
            table_data.append({
                'key': file.as_posix(),
                'name': {
                    'content': file.name,
                },
                'type': fac.AntdIcon(icon='antd-file'),
                'file_count': '-',
                'modified': datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                'path': file.as_posix(),
                'is_dir': False,
            })

        return table_data

    def callbacks(self, app, fsc, cache, args=None):
        @app.callback(
            Output("selection-modal", "visible", allow_duplicate=True),
            Output("selection-modal", 'title'),
            Output('selected-files-extensions', 'options'),
            Output('selected-files-extensions', 'value'),
            Output('processing-type-store', 'data', allow_duplicate=True),
            Output('processing-cpu-form', 'style'),

            Input({'action': 'file-explorer', 'type': ALL}, 'nClicks'),
            prevent_initial_call=True
        )
        def open_selection_modal(n_clicks):
            if n_clicks is None or not any(n_clicks):
                raise PreventUpdate

            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate

            prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
            prop_data = json.loads(prop_id)

            if prop_data['type'] in ("ms-files", "ms-files-empty"):
                title = "Load MS Files"
                file_extensions = [".mzXML", ".mzML"]
                style = {'display': 'block'}
                # Normalize type for downstream processing
                prop_data['type'] = "ms-files"
            elif prop_data['type'] == "metadata":
                title = "Load Metadata"
                file_extensions = [".csv", ".tsv", ".txt", ".xls", ".xlsx"]
                style = {'display': 'none'}
            elif prop_data['type'] in ("targets", "targets-empty"):
                title = "Load Targets"
                file_extensions = [".csv", ".tsv", ".txt", ".xls", ".xlsx"]
                style = {'display': 'none'}
                # Normalize type for downstream processing
                prop_data['type'] = "targets"
            else:
                # Fallback for metadata
                title = "Load Metadata"
                file_extensions = [".csv", ".tsv", ".txt", ".xls", ".xlsx"]
                style = {'display': 'none'}

            processing_type_store = {'type': prop_data['type'], 'extensions': file_extensions}

            return True, title, file_extensions, file_extensions, processing_type_store, style

        @app.callback(
            Output('selected-files-store', 'data', allow_duplicate=True),
            Output('processing-type-store', 'data', allow_duplicate=True),
            Output('processed-action-store', 'data', allow_duplicate=True),
            Output('table-data-store', 'data', allow_duplicate=True),
            Output('marked-for-removal', 'data', allow_duplicate=True),
            Output('is-at-root', 'data', allow_duplicate=True),

            Output('selected-files-tree', 'treeData', allow_duplicate=True),
            Output('selection-counter', 'children', allow_duplicate=True),
            Output('file-table', 'selectedRowKeys', allow_duplicate=True),
            Output('selected-files-tree', 'checkedKeys', allow_duplicate=True),
            Output('selected-files-tree', 'style', allow_duplicate=True),
            Output('files-selection-empty', 'style', allow_duplicate=True),

            Input('selection-modal', 'visible'),
            prevent_initial_call=True
        )
        def on_modal_close(modal_visible):
            if modal_visible:
                raise PreventUpdate
            return ([], {}, None, [], [], False, [], 'Total: 0 files selected', [], [], {'display': 'none'},
                    {'display': 'block'})

        @app.callback(
            Output('sm-processing-progress', 'percent', allow_duplicate=True),
            Output('ms-files-progress-stage', 'children', allow_duplicate=True),
            Output('ms-files-progress-detail', 'children', allow_duplicate=True),
            Input('selection-modal', 'visible'),
            prevent_initial_call=True,
        )
        def reset_processing_progress(modal_visible):
            if not modal_visible:
                raise PreventUpdate
            return 0, "", ""

        @app.callback(
            Output('explorer-processing-progress-container', 'style', allow_duplicate=True),
            Output('selection-container', 'style', allow_duplicate=True),
            Output('selection-modal', 'confirmLoading', allow_duplicate=True),
            Output('selection-modal', 'confirmAutoSpin', allow_duplicate=True),
            Output('selection-modal', 'cancelButtonProps', allow_duplicate=True),
            Input('selection-modal', 'visible'),
            prevent_initial_call=True,
        )
        def reset_modal_state(modal_visible):
            if not modal_visible:
                raise PreventUpdate
            progress_style = {'display': 'none', 'textAlign': 'center', 'padding': '20px'}
            selection_style = {'display': 'block'}
            return progress_style, selection_style, False, False, {'disabled': False}

        @app.callback(
            Output("current-path-modal", "items", allow_duplicate=True),
            Output("file-table", "data", allow_duplicate=True),
            Output('current-path-store', 'data', allow_duplicate=True),
            Output('table-data-store', 'data', allow_duplicate=True),

            Input('selection-modal', 'visible'),
            Input('current-path-modal', 'clickedItem'),
            Input('file-table', 'nClicksCell'),

            State('current-path-store', 'data'),
            State('processing-type-store', 'data'),
            State('file-table', 'recentlyCellClickRecord'),

            prevent_initial_call=True
        )
        def navigate_folders(modal_visible, bc_clicked_item, n_clicks_cell, current_path, processing_type, recentlyCellClickRecord):
            return _navigate_folders(self, modal_visible, bc_clicked_item, n_clicks_cell, recentlyCellClickRecord, current_path, processing_type)

        @app.callback(
            Output('selected-files-store', 'data', allow_duplicate=True),
            Output('selected-files-tree', 'treeData', allow_duplicate=True),
            Output('selected-files-tree', 'style', allow_duplicate=True),
            Output('files-selection-empty', 'style', allow_duplicate=True),
            Output('selection-counter', 'children', allow_duplicate=True),
            Output('file-table', 'selectedRowKeys', allow_duplicate=True),
            Output('marked-for-removal', 'data', allow_duplicate=True),
            Output('selected-files-tree', 'checkedKeys', allow_duplicate=True),

            Input('file-table', 'selectedRowKeys'),
            Input('file-table', 'nClicksCell'),
            Input('clear-selection-btn', 'nClicks'),
            Input('remove-marked-btn', 'nClicks'),
            Input('selected-files-tree', 'checkedKeys'),

            State('file-table', 'recentlyCellClickRecord'),
            State('selected-files-store', 'data'),
            State('processing-type-store', 'data'),
            State('table-data-store', 'data'),
            State('marked-for-removal', 'data'),

            prevent_initial_call=True
        )
        def update_selection(selectedRowKeys, nClicksCell, clear_clicks, remove_clicks, tree_checked_keys,
                             recentlyCellClickRecord, current_selection, processing_type, table_data, marked_for_removal):
            return _update_selection(self, selectedRowKeys, nClicksCell, clear_clicks, remove_clicks, tree_checked_keys,
                             recentlyCellClickRecord, current_selection, processing_type, table_data, marked_for_removal)

        @app.callback(
            Output('notifications-container', "children", allow_duplicate=True),
            Output('processed-action-store', 'data', allow_duplicate=True),
            Output('selection-modal', 'visible', allow_duplicate=True),
            Output('workspace-status', 'data', allow_duplicate=True),

            Input('selection-modal', 'okCounts'),
            State('processing-type-store', 'data'),
            State("selected-files-store", "data"),
            State('processing-cpu-input', 'value'),
            State("wdir", "data"),
            background=True,
            running=[
                (Output('explorer-processing-progress-container', 'style'),
                 {
                     "display": "flex",
                     "justifyContent": "center",
                     "alignItems": "center",
                     "flexDirection": "column",
                     "minWidth": "200px",
                     "maxWidth": "400px",
                     "margin": "auto",
                     'height': "60vh"
                 }, {'display': 'none'}),
                (Output("selection-container", "style"), {'display': 'none'}, {'display': 'block'}),

                (Output('selection-modal', 'confirmAutoSpin'), True, False),
                (Output('selection-modal', 'cancelButtonProps'), {'disabled': True},
                 {'disabled': False}),
                (Output('selection-modal', 'confirmLoading'), True, False),
            ],
            progress=[
                Output("sm-processing-progress", "percent"),
                Output("ms-files-progress-stage", "children"),
                Output("ms-files-progress-detail", "children"),
            ],
            cancel=[
                Input('cancel-ms-processing', 'nClicks')
            ],
            prevent_initial_call=True
        )
        def background_processing(set_progress, okCounts, processing_type, selected_files_list,
                                  cpu_input, wdir):
            return _background_processing(set_progress, okCounts, processing_type, selected_files_list, cpu_input, wdir)

        @app.callback(
            Output('selection-modal', 'visible', allow_duplicate=True),
            Input('cancel-ms-processing', 'nClicks'),
            prevent_initial_call=True,
        )
        def close_modal_on_cancel(cancel_clicks):
            if not cancel_clicks:
                raise PreventUpdate
            return False

        @app.callback(
            Output('explorer-processing-progress-container', 'style', allow_duplicate=True),
            Output('selection-container', 'style', allow_duplicate=True),
            Output('selection-modal', 'confirmLoading', allow_duplicate=True),
            Output('selection-modal', 'confirmAutoSpin', allow_duplicate=True),
            Output('selection-modal', 'cancelButtonProps', allow_duplicate=True),
            Input('clear-selection-btn', 'nClicks'),
            Input('remove-marked-btn', 'nClicks'),
            prevent_initial_call=True,
        )
        def restore_modal_body(clear_clicks, remove_clicks):
            progress_style = {'display': 'none', 'textAlign': 'center', 'padding': '20px'}
            selection_style = {'display': 'block'}
            return progress_style, selection_style, False, False, {'disabled': False}


def _navigate_folders(explorer_instance, modal_visible, bc_clicked_item, n_clicks_cell, recentlyCellClickRecord, current_path, processing_type):
    if not modal_visible:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    ctx = dash.callback_context
    # Check triggered for real callbacks, but allow manual calls for testing
    if ctx and ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    else:
        # Fallback for testing or non-contextual calls
        trigger_id = 'selection-modal'
    
    WIN_ROOT_KEY = "::WIN_ROOT::"
    show_drives = False
    new_path = None

    # Determine the new path
    if 'current-path-modal' in trigger_id and bc_clicked_item:
        key = bc_clicked_item.get('itemKey', current_path)
        if key == WIN_ROOT_KEY:
            show_drives = True
        else:
            new_path = Path(key)
    elif 'file-table' in trigger_id and recentlyCellClickRecord:
        # Navigate only if is a directory
        if recentlyCellClickRecord.get('is_dir'):
            new_path = Path(recentlyCellClickRecord['key'])
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif 'selection-modal' in trigger_id:
        if current_path == WIN_ROOT_KEY:
             show_drives = True
        else:
             new_path = Path(current_path or home_path)
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Handle Drive Listing View
    if show_drives:
        breadcrumb_items = [{'title': 'My Computer', 'key': WIN_ROOT_KEY}]
        extensions = processing_type.get('extensions', ['.csv'])
        new_table_data = explorer_instance.get_table_data(None, extensions, is_root=True)
        return breadcrumb_items, new_table_data, WIN_ROOT_KEY, new_table_data

    # Regular Construction of breadcrumb
    try:
        breadcrumb_items = []
        # Add My Computer node for Windows
        if platform.system() == 'Windows':
            breadcrumb_items.append({'title': 'My Computer', 'key': WIN_ROOT_KEY})
            
            root = Path(new_path.drive + '\\')
            all_paths = [root] + [p for p in new_path.parents if p != root][::-1]
            if new_path != root:
                all_paths.append(new_path)
        else:
            all_paths = [Path('/')] + [p for p in new_path.parents if p != Path('/')][::-1]
            if new_path != Path('/'):
                all_paths.append(new_path)

        for i, path in enumerate(all_paths):
            if i == 0 and platform.system() != 'Windows':
                breadcrumb_items.append(
                    {'title': 'root' if str(path) == '/' else str(path), 'key': str(path)})
            else:
                 breadcrumb_items.append({'title': path.name or str(path), 'key': str(path)})
                 
    except Exception as e:
        logger.error(f"Error constructing breadcrumb for {new_path}: {e}")
        breadcrumb_items = [{'title': str(new_path), 'key': str(new_path)}]

    # Generate table data
    extensions = processing_type.get('extensions', ['.csv'])
    new_table_data = explorer_instance.get_table_data(new_path, extensions)

    return breadcrumb_items, new_table_data, str(new_path), new_table_data


def _update_selection(explorer_instance, selectedRowKeys, nClicksCell, clear_clicks, remove_clicks, tree_checked_keys,
                      recentlyCellClickRecord, current_selection, processing_type, table_data, marked_for_removal):
    ctx = dash.callback_context
    if ctx and not ctx.triggered:
        raise PreventUpdate

    if ctx:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        trigger_action = ctx.triggered[0]['prop_id'].split('.')[-1]
    else:
        # Fallback for testing
        trigger_id = 'file-table'
        trigger_action = 'selectedRowKeys'

    selected_files = set(current_selection or [])
    marked = set(marked_for_removal or [])

    if trigger_id == 'clear-selection-btn':
        # Clear all selection
        selected_files = set()
        new_selected_keys = []
        marked = set()
        tree_checks = []

    elif trigger_id == 'file-table':
        # Update selection based on selected rows
        extensions = processing_type.get('extensions', [])
        if trigger_action == 'nClicksCell':
            # Cell was clicked - add the file if it's not a directory
            if recentlyCellClickRecord and not recentlyCellClickRecord.get('is_dir'):
                selected_files.add(recentlyCellClickRecord['key'])
        else:
            if selectedRowKeys:
                # Process new selections
                for row_key in selectedRowKeys:
                    # Search for the item in the table
                    for item in table_data:
                        if item['key'] == row_key:
                            if item['is_dir']:
                                # It's a folder, add all its files recursively
                                folder_path = Path(item['path'])
                                for ext in extensions:
                                    selected_files.update(str(f) for f in folder_path.rglob(f"*{ext}"))
                            else:
                                # It's a file
                                selected_files.add(item['path'])
                                break
        new_selected_keys = selectedRowKeys
        tree_checks = list(marked)

    elif trigger_id == 'selected-files-tree':
        # Update marked items for removal
        marked = set(tree_checked_keys or [])
        new_selected_keys = dash.no_update
        tree_checks = list(marked)

    elif trigger_id == 'remove-marked-btn':
        # Remove marked items - OPTIMIZED: pre-compute folder→files mapping once
        items_to_remove = set()
        selectedRowKeysToDelete = set()
        
        # Build folder→files mapping once (O(n) instead of O(n*m))
        folder_to_files = {}
        for f in selected_files:
            folder = Path(f).parent.as_posix()
            if folder not in folder_to_files:
                folder_to_files[folder] = set()
            folder_to_files[folder].add(f)
        
        # Known folders from the tree structure (no filesystem calls needed)
        known_folders = set(folder_to_files.keys())
        
        for item in marked:
            if item in (selectedRowKeys or []):
                selectedRowKeysToDelete.add(item)
            
            # Check if this is a folder (in our known folders) or a file
            if item in known_folders:
                # It's a folder - remove all its files directly from the mapping
                items_to_remove.update(folder_to_files[item])
            elif item in selected_files:
                # It's a file
                items_to_remove.add(item)

        selected_files -= items_to_remove
        marked = set()
        new_selected_keys = list(set(selectedRowKeys or []) - selectedRowKeysToDelete)
        tree_checks = []
    else:
        raise PreventUpdate

    selected_list = sorted(list(selected_files))

    # generate selected tree data
    tree_data = explorer_instance.get_selected_tree_data(selected_list)

    # styles
    if selected_list:
        tree_style = {'display': 'flex'}
        empty_style = {'display': 'none'}
    else:
        tree_style = {'display': 'none'}
        empty_style = {'display': 'block'}

    counter = f"Total: {len(selected_list)} files selected"
    return selected_list, tree_data, tree_style, empty_style, counter, new_selected_keys, list(
        marked), tree_checks


def _background_processing(set_progress, okCounts, processing_type, selected_files_list,
                           cpu_input, wdir):
    if not okCounts or not selected_files_list:
        raise PreventUpdate
    
    if wdir:
        activate_workspace_logging(wdir)

    # Convert list to dictionary grouped by folder for compatibility
    selected_files = {}
    for file_path in selected_files_list:
        folder = str(Path(file_path).parent)
        if folder not in selected_files:
            selected_files[folder] = []
        selected_files[folder].append(file_path)

    # defaults for branches
    failed_targets = []
    stats = {}
    duplicates_count = 0
    stage_label = "Processing"

    def progress_adapter(percent, detail=""):
        if set_progress:
            set_progress((percent, stage_label, detail or ""))

    if processing_type['type'] == "ms-files":
        stage_label = "MS Files"
        logger.info(f"Starting MS-Files processing. Selected: {len(selected_files_list)} files.")
        total_processed, failed_files, duplicates_count = process_ms_files(
            wdir, progress_adapter, selected_files, cpu_input
        )
        message = "MS Files processed"
    elif processing_type['type'] == "metadata":
        stage_label = "Metadata"
        logger.info("Starting Metadata processing.")
        total_processed, failed_files = process_metadata(wdir, progress_adapter, selected_files)
        message = "Metadata processed"
    else:
        stage_label = "Targets"
        logger.info("Starting Targets processing.")
        total_processed, failed_files, failed_targets, stats = process_targets(wdir, progress_adapter, selected_files)
        message = "Targets processed"

    duplicate_targets = stats.get("duplicate_peak_labels", 0) if processing_type['type'] == "targets" else 0
    failed_targets_count = len(failed_targets) if processing_type['type'] == "targets" else 0
    rt_adjusted_count = stats.get("rt_adjusted_count", 0) if processing_type['type'] == "targets" else 0

    # Log results
    logger.info(f"Processing finished. Processed: {total_processed}, Failed: {len(failed_files)}, Duplicates: {duplicates_count}")
    if failed_files:
        logger.warning(f"Failed files: {failed_files}")
    # Note: Individual target failures are already logged in tools.py, no need to log again here

    if total_processed:
        details = []
        if failed_files:
            details.append(f"{len(failed_files)} file(s) failed")
        if duplicates_count:
            details.append(f"{duplicates_count} duplicate file(s) skipped")
        if failed_targets_count:
            details.append(f"{failed_targets_count} target row(s) failed")
        if duplicate_targets:
            details.append(f"{duplicate_targets} duplicate target label(s) deduplicated")
        if rt_adjusted_count:
            details.append(f"{rt_adjusted_count} RT value(s) adjusted (outside span)")

        if details:
            description = f"Processed {total_processed} items. " + ", ".join(details)
            if failed_files or failed_targets_count:
                mss_type = "warning"
            elif duplicates_count or duplicate_targets:
                mss_type = "info"
            else:
                mss_type = "success"
        else:
            description = f"Processed {total_processed} items."
            mss_type = "success"
    elif duplicates_count:
        description = f"Skipped {duplicates_count} duplicates."
        mss_type = "info"
    elif failed_files or failed_targets_count:
        # Build informative error message showing actual validation errors
        if failed_files:
            # Get the first failed file's error message
            first_error = next(iter(failed_files.values()))
            # Extract key info from error message
            if "No valid targets found" in first_error:
                lines = first_error.strip().split('\n')
                description = f"Failed to process {len(failed_files)} file(s). {lines[0]}"
                # Add targets failed count if available  
                for line in lines:
                    if "Targets failed:" in line:
                        description += f" {line.strip()}"
                        break
            else:
                # Generic file error - show first 100 chars
                description = f"Failed to process {len(failed_files)} file(s). Error: {first_error[:100]}"
        else:
            description = f"Failed to process {failed_targets_count} target(s)."
        mss_type = "error"
    else:
        description = "No items processed."
        mss_type = "info"
    notification = fac.AntdNotification(message=message,
                                        description=description,
                                        type=mss_type,
                                        duration=3,
                                        placement='bottom',
                                        showProgress=True,
                                        style={"maxWidth": 420, "width": "420px"})
    processed_action_store = {'action': 'processing', 'status': 'success'}

    # Update workspace-status with current counts after processing
    workspace_status = {
        'ms_files_count': 0,
        'targets_count': 0,
        'chromatograms_count': 0,
        'selected_targets_count': 0,
        'optimization_samples_count': 0,
        'results_count': 0
    }
    if wdir:
        from ..duckdb_manager import duckdb_connection
        with duckdb_connection(wdir) as conn:
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
                    import psutil
                    from multiprocessing import cpu_count
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
                    logger.info(f"workspace-status updated after processing: {workspace_status}")

    return notification, processed_action_store, False, workspace_status

