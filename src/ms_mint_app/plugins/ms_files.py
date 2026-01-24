import logging
import os
import tempfile
from pathlib import Path as P, Path

import dash
import feffery_antd_components as fac
import feffery_utils_components as fuc
import pandas as pd
import polars as pl
import time
import math
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from plotly import colors

from .. import tools as T
from ..colors import make_palette_hsv
from ..duckdb_manager import duckdb_connection, build_where_and_params, build_order_by, compact_database
from ..plugin_interface import PluginInterface
from ..sample_metadata import GROUP_COLUMNS, GROUP_DESCRIPTIONS, GROUP_LABELS
from ..logging_setup import activate_workspace_logging

_label = "MS-Files"
MS_METADATA_TEMPLATE_COLUMNS = [
    'ms_file_label',
    'label',
    'color',
    'use_for_optimization',
    'use_for_processing',
    'use_for_analysis',
    'sample_type',
    *GROUP_COLUMNS,
    'polarity',
    'ms_type',
    'file_type',
]
MS_METADATA_TEMPLATE_DESCRIPTIONS = [
    'Unique file name; must match the MS file on disk',
    'Friendly label to display in plots and reports',
    'Hex color for visualizations (auto-generated if blank)',
    'True to include in optimization steps (COMPUTE CHROMATOGRAMS)',
    'True to include in processing (RUN MINT)',
    'True to include in analysis outputs',
    'Sample category (e.g.; Sample; QC; Blank; Standard)',
    *[GROUP_DESCRIPTIONS[col] for col in GROUP_COLUMNS],
    'Polarity (Positive or Negative)',
    'Acquisition type (ms1 or ms2)',
    'File format (e.g., mzML, mzXML)',
]
MS_METADATA_TEMPLATE_CSV = (
    ",".join(MS_METADATA_TEMPLATE_COLUMNS)
    + "\n"
    + ",".join(MS_METADATA_TEMPLATE_DESCRIPTIONS)
    + "\n"
)
MS_METADATA_DESCRIPTION_MAP = dict(zip(MS_METADATA_TEMPLATE_COLUMNS, MS_METADATA_TEMPLATE_DESCRIPTIONS))
NOTIFICATION_COMPACT_STYLE = {"maxWidth": 420, "width": "420px"}

home_path = Path.home()

logger = logging.getLogger(__name__)

# Patterns for detecting sample types from filenames
# Each entry: (patterns_list, color, sample_type_label)
# Patterns are case-insensitive and match substrings
SAMPLE_TYPE_PATTERNS = [
    # Blanks - quality control samples with no analyte (solvent/matrix only)
    (['blank', 'blk', 'solvent', 'diluent', 'mobilephase', 'reagent', 
      'matrix', 'background', 'zero', 'blanco', 'blanc'], '#2B2D42', 'Blank'),
    # Standards - reference samples for calibration
    (['std', 'standard', 'stnd', 'calibr', 'calibrant', 'cal', 'ref', 'reference',
      'stock', 'workingstandard', 'primarystandard', 'secondarystandard'], '#33BBEE', 'Standard'),
    # QC - quality control samples for system monitoring
    (['qc', 'qualitycontrol', 'control', 'check', 'pool', 'pqc', 
      'system suitability', 'sst', 'verification', 'crm'], '#CC3311', 'QC'),
]
# Default color for regular samples: #BBBBBB (set in database schema)



def detect_sample_color(filename: str) -> tuple[str | None, str | None]:
    """
    Detect sample type from filename and return appropriate color.
    
    Args:
        filename: The MS file label/name to analyze
    
    Returns:
        Tuple of (color, sample_type) if pattern matched, (None, None) otherwise
    """
    filename_lower = filename.lower()
    
    for patterns, color, sample_type in SAMPLE_TYPE_PATTERNS:
        for pattern in patterns:
            if pattern in filename_lower:
                return color, sample_type
    
    return None, None



class MsFilesPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 2
        logger.info(f"Initiated {_label} plugin")

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache, args):
        callbacks(self, app, fsc, cache, args)

    def outputs(self):
        return None


upload_root = os.getenv("MINT_DATA_DIR", tempfile.gettempdir())
upload_dir = str(P(upload_root) / "MINT-Uploads")
UPLOAD_FOLDER_ROOT = upload_dir

_layout = html.Div(
    [
        fac.AntdFlex(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'MS-Files', level=4, style={'margin': '0'}
                        ),
                        fac.AntdTooltip(
                            fac.AntdIcon(
                                id='ms-files-tour-icon',
                                icon='pi-info',
                                style={"cursor": "pointer", 'paddingLeft': '10px'},
                                **{'aria-label': 'Show tutorial'},
                            ),
                            title='Show tutorial'
                        ),
                        # Always visible: Load MS-Files button
                        fac.AntdTooltip(
                            fac.AntdButton(
                                'Load MS-Files',
                                id={
                                    'action': 'file-explorer',
                                    'type': 'ms-files',
                                },
                                style={'textTransform': 'uppercase', "margin": "0 10px"},
                            ),
                            title="Import MS files (mzML, mzXML) or ZIP archives containing them.",
                            placement="bottom"
                        ),
                        # Conditionally visible: Load Metadata button (only when files exist)
                        html.Div(
                            fac.AntdTooltip(
                                fac.AntdButton(
                                    'Load Metadata',
                                    id={
                                        'action': 'file-explorer',
                                        'type': 'metadata',
                                    },
                                    style={'textTransform': 'uppercase', "margin": "0 10px"},
                                ),
                                title="Import a metadata file (CSV) to annotate your MS files.",
                                placement="bottom"
                            ),
                            id='ms-files-load-metadata-wrapper',
                            style={'display': 'none'},
                        ),
                    ],
                    align='center',
                ),
                fac.AntdFlex(
                    [
                        fac.AntdTooltip(
                            fac.AntdButton(
                                'Download template',
                                id='download-ms-template-btn',
                                icon=fac.AntdIcon(icon='antd-download'),
                                iconPosition='end',
                                style={'textTransform': 'uppercase'},
                            ),
                            title="Download a blank CSV template for file metadata.",
                            placement="bottom"
                        ),
                        # Conditionally visible: Download MS-Files and Options (only when files exist)
                        html.Div(
                            [
                                fac.AntdTooltip(
                                    fac.AntdButton(
                                        'Download MS-files',
                                        id='download-ms-files-btn',
                                        icon=fac.AntdIcon(icon='antd-download'),
                                        iconPosition='end',
                                        style={'textTransform': 'uppercase', "margin": "0 10px"},
                                    ),
                                    title="Download the currently loaded MS-files list as a CSV.",
                                    placement="bottom"
                                ),
                                html.Div(
                                    fac.AntdDropdown(
                                        id='ms-options',
                                        title='Options',
                                        buttonMode=True,
                                        arrow=True,
                                        menuItems=[
                                            {'title': 'Generate colors', 'icon': 'antd-highlight', 'key': 'generate-colors'},
                                            {'isDivider': True},
                                            {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                                             'key': 'delete-selected'},
                                            {'title': fac.AntdText('Clear table', strong=True, type='danger'), 'key': 'delete-all'},
                                        ],
                                        buttonProps={'style': {'textTransform': 'uppercase'}},
                                    ),
                                    id='ms-options-wrapper',
                                ),
                            ],
                            id='ms-files-data-actions-wrapper',
                            style={'display': 'none', 'gap': '8px'},
                            className='ant-flex',
                        ),
                    ],
                    align='center',
                    gap='small',
                ),
            ],
            justify="space-between",
            align="center",
            gap="middle",
        ),
        fac.AntdModal(
            "Are you sure you want to delete the selected files?",
            title="Delete confirmation",
            id="delete-confirmation-modal",
            okButtonProps={"danger": True},
            renderFooter=True,
            locale='en-us',
        ),
        fac.AntdModal(
            [
                fac.AntdCenter(
                    fuc.FefferyHexColorPicker(
                        id='hex-color-picker', showAlpha=True
                    )
                ),
                fac.AntdFlex(
                    [
                        fac.AntdInput(
                            id='hex-color-input',
                            placeholder='#RRGGBB',
                            style={'width': '120px', 'textAlign': 'center'},
                            maxLength=7,
                        ),
                        html.Div(
                            id='hex-color-preview',
                            style={
                                'width': '30px',
                                'height': '30px',
                                'borderRadius': '4px',
                                'border': '1px solid #d9d9d9',
                            }
                        )
                    ],
                    gap='small',
                    justify='center',
                    align='center',
                    style={'marginTop': '10px'}
                ),
                html.Div(id='hex-color-error', style={'color': 'red', 'fontSize': '12px', 'textAlign': 'center', 'marginTop': '5px'})
            ],
            id='color-picker-modal',
            renderFooter=True,
            width=300,
            styles={
                'body': {
                    'height': 280,
                    'alignItems': 'center',
                    'alignContent': 'end',
                }
            },
            locale='en-us',
        ),
        # Empty state placeholder - shown when no MS files
        html.Div(
            fac.AntdFlex(
                [
                    fac.AntdEmpty(
                        description=fac.AntdFlex(
                            [
                                fac.AntdText('No MS files loaded', strong=True, style={'fontSize': '16px'}),
                                fac.AntdText('Click "Load MS-Files" to import your data', type='secondary'),
                            ],
                            vertical=True,
                            align='center',
                            gap='small',
                        ),
                        locale='en-us',
                    ),
                    fac.AntdTooltip(
                        fac.AntdButton(
                            'Load MS-Files',
                            id={
                                'action': 'file-explorer',
                                'type': 'ms-files-empty',
                            },
                            size='large',
                            style={'marginTop': '16px', 'textTransform': 'uppercase'},
                        ),
                        title="Import MS files (mzML, mzXML) or ZIP archives containing them.",
                        placement="bottom"
                    ),
                ],
                vertical=True,
                align='center',
                style={'marginTop': '100px'},
            ),
            id='ms-files-empty-state',
            style={'display': 'block'},
        ),
        # Conditionally visible: Table container (only when files exist)
        html.Div(
            [
                fac.AntdSpin(
                    fac.AntdTable(
                        id='ms-files-table',
                        containerId='ms-files-table-container',
                        columns=[
                            {
                                'title': 'MS-File Label',
                                'dataIndex': 'ms_file_label',
                                'width': '300px',
                                'fixed': 'left'
                            },
                            {
                                'title': 'Label',
                                'dataIndex': 'label',
                                'width': '300px',
                                'editable': True,
                                'editOptions': {
                                    'mode': 'text-area',
                                    'autoSize': {'minRows': 1, 'maxRows': 3},
                                },
                            },
                            {
                                'title': 'Color',
                                'dataIndex': 'color',
                                'width': '100px',
                                'renderOptions': {'renderType': 'button'},
                            },
                            {
                                'title': 'For Optimization',
                                'dataIndex': 'use_for_optimization',
                                'renderOptions': {'renderType': 'switch'},
                                'width': '170px',
                            },
                            {
                                'title': 'For Processing',
                                'dataIndex': 'use_for_processing',
                                'renderOptions': {'renderType': 'switch'},
                                'width': '170px',
                            },
                            {
                                'title': 'For Analysis',
                                'dataIndex': 'use_for_analysis',
                                'renderOptions': {'renderType': 'switch'},
                                'width': '150px',
                            },
                            {
                                'title': 'Sample Type',
                                'dataIndex': 'sample_type',
                                'width': '150px',
                                'editable': True,
                            },
                            *[
                                {
                                    'title': GROUP_LABELS[col],
                                    'dataIndex': col,
                                    'width': '130px',
                                    'editable': True,
                                }
                                for col in GROUP_COLUMNS
                            ],
                            {
                                'title': 'Polarity',
                                'dataIndex': 'polarity',
                                'width': '150px',
                            },
                            {
                                'title': 'MS Type',
                                'dataIndex': 'ms_type',
                                'width': '120px',
                            },
                            {
                                'title': 'File Type',
                                'dataIndex': 'file_type',
                                'width': '120px',
                            },
                        ],
                        titlePopoverInfo={
                            'ms_file_label': {
                                'title': 'ms_file_label',
                                'content': MS_METADATA_DESCRIPTION_MAP['ms_file_label'],
                            },
                            'label': {
                                'title': 'label',
                                'content': MS_METADATA_DESCRIPTION_MAP['label'],
                            },
                            'color': {
                                'title': 'color',
                                'content': MS_METADATA_DESCRIPTION_MAP['color'],
                            },
                            'use_for_optimization': {
                                'title': 'use_for_optimization',
                                'content': MS_METADATA_DESCRIPTION_MAP['use_for_optimization'],
                            },
                            'use_for_processing': {
                                'title': 'use_for_processing',
                                'content': MS_METADATA_DESCRIPTION_MAP['use_for_processing'],
                            },
                            'use_for_analysis': {
                                'title': 'use_for_analysis',
                                'content': MS_METADATA_DESCRIPTION_MAP['use_for_analysis'],
                            },
                            'sample_type': {
                                'title': 'sample_type',
                                'content': MS_METADATA_DESCRIPTION_MAP['sample_type'],
                            },
                            **{
                                col: {
                                    'title': GROUP_LABELS[col],
                                    'content': MS_METADATA_DESCRIPTION_MAP[col],
                                } for col in GROUP_COLUMNS
                            },
                            'polarity': {
                                'title': 'polarity',
                                'content': MS_METADATA_DESCRIPTION_MAP['polarity'],
                            },
                            'ms_type': {
                                'title': 'ms_type',
                                'content': MS_METADATA_DESCRIPTION_MAP['ms_type'],
                            },
                            'file_type': {
                                'title': 'file_type',
                                'content': MS_METADATA_DESCRIPTION_MAP['file_type'],
                            },
                        },
                        filterOptions={
                            'ms_file_label': {'filterMode': 'keyword'},
                            'label': {'filterMode': 'keyword'},
                            'color': {'filterMode': 'keyword'},
                            'use_for_optimization': {'filterMode': 'checkbox',
                                                      'filterCustomItems': ['True', 'False']},
                            'use_for_processing': {'filterMode': 'checkbox',
                                                   'filterCustomItems': ['True', 'False']},
                            'use_for_analysis': {'filterMode': 'checkbox',
                                                  'filterCustomItems': ['True', 'False']},
                            'sample_type': {'filterMode': 'checkbox'},
                            **{col: {'filterMode': 'keyword'} for col in GROUP_COLUMNS},
                            'polarity': {'filterMode': 'checkbox',
                                         'filterCustomItems': ['Positive', 'Negative']},
                            'ms_type': {'filterMode': 'checkbox',
                                        'filterCustomItems': ['ms1', 'ms2']},
                            'file_type': {'filterMode': 'checkbox',
                                          'filterCustomItems': ['mzXML', 'mzML']},
                        },
                        sortOptions={'sortDataIndexes': []},
                        pagination={
                            'position': 'bottomCenter',
                            'pageSize': 15,
                            'current': 1,
                            'showSizeChanger': True,
                            'pageSizeOptions': [5, 10, 15, 25, 50, 100],
                            'showQuickJumper': True,
                        },
                        tableLayout='fixed',
                        maxWidth="calc(100vw - 250px - 4rem)",
                        maxHeight="calc(100vh - 210px)",
                        locale='en-us',
                        rowSelectionType='checkbox',
                        size='small',
                        mode='server-side',
                    ),
                    id='ms-files-table-spin',
                    text='Updating table...',
                    size='small',
                    spinning=False,
                    listenPropsMode='exclude',
                    excludeProps=[
                        'ms-files-table.data',
                        'ms-files-table.pagination',
                        'ms-files-table.selectedRowKeys',
                        'ms-files-table.filterOptions',
                    ],
                )
            ],
            id='ms-files-table-container',
            style={'paddingTop': '1rem', 'display': 'none'},
        ),
        # Tour for empty workspace (no files loaded) - simplified
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'This tutorial shows how to get started with MS files.',
                },
                {
                    'title': 'Load raw files',
                    'description': 'Click "Load MS-Files" to browse and add raw data files (mzML, mzXML) to this workspace.',
                    'targetSelector': "[id='{\"action\":\"file-explorer\",\"type\":\"ms-files\"}']"
                },
                {
                    'title': 'Use the metadata template',
                    'description': 'Download the metadata template if you need the expected column names for annotating your files.',
                    'targetSelector': '#download-ms-template-btn'
                },
            ],
            id='ms-files-tour-empty',
            open=False,
            current=0,
        ),
        # Tour for populated workspace (files loaded) - full tour
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'This tutorial shows how to manage, review, and export your MS files.',
                },
                {
                    'title': 'Load more files',
                    'description': 'Click "Load MS-Files" to add more raw data files to this workspace.',
                    'targetSelector': "[id='{\"action\":\"file-explorer\",\"type\":\"ms-files\"}']"
                },
                {
                    'title': 'Add metadata',
                    'description': 'Use "Load Metadata" to import a CSV with sample info (labels, types, etc.).',
                    'targetSelector': "[id='{\"action\":\"file-explorer\",\"type\":\"metadata\"}']"
                },
                {
                    'title': 'Options',
                    'description': 'Generate colors or delete selected/all rows from the options menu.',
                    'targetSelector': '#ms-options-wrapper'
                },
                {
                    'title': 'Review and filter',
                    'description': 'Inspect, filter, and sort MS files here; select rows to delete or export.',
                    'targetSelector': '#ms-files-table-container'
                },
                {
                    'title': 'Export',
                    'description': 'Download the current table (with server-side filters) for backup or sharing.',
                    'targetSelector': '#download-ms-files-btn'
                },
                {
                    'title': 'Use the metadata template',
                    'description': 'Download the metadata template if you need the expected column names.',
                    'targetSelector': '#download-ms-template-btn'
                },
            ],
            id='ms-files-tour-full',
            open=False,
            current=0,
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Need help?',
                    'description': 'Click the info icon to open a quick tour of the MS-Files table.',
                    'targetSelector': '#ms-files-tour-icon',
                },
            ],
            mask=False,
            placement='rightTop',
            open=False,
            current=0,
            id='ms-files-tour-hint',
            className='targets-tour-hint',
            style={
                'background': '#ffffff',
                'border': '0.5px solid #1677ff',
                'boxShadow': '0 6px 16px rgba(0,0,0,0.15), 0 0 0 1px rgba(22,119,255,0.2)',
                'opacity': 1,
            },
        ),
        dcc.Store(id="ms-tour-hint-store", data={'open': False}, storage_type='local'),
        dcc.Download(id='download-ms-files-csv'),
        dcc.Store(id="ms-table-action-store", data={}),
    ]
)


def layout():
    return _layout

def generate_colors(wdir, regenerate=False):
    def _rgb_to_hex(rgb_str: str) -> str:
        if isinstance(rgb_str, str) and rgb_str.startswith("rgb"):
            nums = rgb_str.strip("rgb() ").split(",")
            if len(nums) == 3:
                try:
                    r, g, b = (int(n) for n in nums)
                    return f"#{r:02x}{g:02x}{b:02x}"
                except ValueError:
                    return rgb_str
        return rgb_str

    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate

        ms_colors = conn.execute(
            "SELECT ms_file_label, sample_type, color FROM samples"
        ).df()
        ms_colors["sample_key"] = ms_colors["sample_type"].fillna(ms_colors["ms_file_label"])

        if regenerate:
            assigned_colors = {}
        else:
            valid = ms_colors[
                ms_colors["color"].notna()
                & (ms_colors["color"].str.strip() != "")
                & (ms_colors["color"].str.strip() != "#BBBBBB")
            ].copy()
            valid["sample_key"] = valid["sample_type"].fillna(valid["ms_file_label"])
            assigned_colors = (
                valid.drop_duplicates(subset="sample_key")
                .set_index("sample_key")["color"]
                .to_dict()
            )

        sample_keys = ms_colors["sample_key"].drop_duplicates().to_list()

        if len(assigned_colors) != len(sample_keys):
            pastel_palette = colors.qualitative.Pastel
            pastel_palette_hex = [_rgb_to_hex(c) for c in pastel_palette]
            if len(sample_keys) <= len(pastel_palette):
                colors_map = assigned_colors.copy()
                palette_idx = 0
                for key in sample_keys:
                    if key in colors_map:
                        continue
                    colors_map[key] = pastel_palette_hex[palette_idx]
                    palette_idx += 1
            else:
                colors_map = make_palette_hsv(
                    sample_keys,
                    existing_map=assigned_colors,
                    s_range=(0.90, 0.95),
                    v_range=(0.90, 0.95),
                )

            colors_pd = ms_colors[["ms_file_label"]].copy()
            colors_pd["color"] = ms_colors["sample_key"].map(colors_map)
            conn.register("colors_pd", colors_pd)
            conn.execute(
                """
                UPDATE samples
                SET color = colors_pd.color
                FROM colors_pd
                WHERE samples.ms_file_label = colors_pd.ms_file_label
                """
            )

        return len(sample_keys) - len(assigned_colors)



def _save_color_to_db(okCounts, color, recentlyButtonClickedRow, wdir):
    """Background callback to persist color change to database."""
    if recentlyButtonClickedRow is None or not okCounts or not wdir:
        return dash.no_update

    previous_color = recentlyButtonClickedRow.get('color', {}).get('content', 'unknown')
    ms_file_label = recentlyButtonClickedRow.get('ms_file_label')
    
    if not ms_file_label:
        return dash.no_update
    
    try:
        with duckdb_connection(wdir) as conn:
            conn.execute("UPDATE samples SET color = ? WHERE ms_file_label = ?",
                         [color, ms_file_label])
        
        logger.info(f"Changed color for {ms_file_label} from {previous_color} to {color}")
        
        return fac.AntdNotification(
            message='Color saved',
            description=f'Color changed to {color}',
            type='success',
            duration=2,
            placement='bottom',
            showProgress=True,
            stack=True,
            style=NOTIFICATION_COMPACT_STYLE
        )
    except Exception as e:
        logger.error(f"DB error saving color: {e}")
        return fac.AntdNotification(
            message='Failed to save color',
            description=f'Error: {str(e)}',
            type='error',
            duration=3,
            placement='bottom',
            showProgress=True,
            stack=True,
            style=NOTIFICATION_COMPACT_STYLE
        )


def _genere_color_map(nClicks, clickedKey, wdir):
    ctx = dash.callback_context
    if (
            (ctx and (not ctx.triggered or not nClicks or clickedKey != 'generate-colors')) or
            (not ctx and clickedKey != 'generate-colors')
    ):
        raise PreventUpdate
    
    if wdir:
        activate_workspace_logging(wdir)

    # Single option: always generate colors by sample type, refreshing missing/placeholder values.
    n_colors = generate_colors(wdir, regenerate=True)

    if n_colors == 0:
        logger.info("Color generation requested but no new colors were needed.")
        notification = fac.AntdNotification(message='No new colors generated',
                                            type='warning',
                                            duration=3,
                                            placement='bottom',
                                            showProgress=True,
                                            stack=True,
                                            style=NOTIFICATION_COMPACT_STYLE
                                            )
    else:
        logger.info(f"Generated {n_colors} new colors.")
        notification = fac.AntdNotification(message='Colors generated',
                                 description=f'{n_colors} new colors generated',
                                 type='success',
                                 duration=3,
                                 placement='bottom',
                                 showProgress=True,
                                 stack=True,
                                 style=NOTIFICATION_COMPACT_STYLE
                                 )
    ms_table_action_store = {'action': 'color-changed', 'status': 'success'}
    return notification, ms_table_action_store, False  # spinning=False when done


def _save_switch_changes(recentlySwitchDataIndex, recentlySwitchStatus, recentlySwitchRow, wdir):
    if not wdir or not recentlySwitchDataIndex or recentlySwitchStatus is None or not recentlySwitchRow:
        raise PreventUpdate

    allowed_switch_columns = {
        "use_for_optimization",
        "use_for_processing",
        "use_for_analysis",
    }
    if recentlySwitchDataIndex not in allowed_switch_columns:
        raise PreventUpdate

    with duckdb_connection(wdir) as conn:
        if conn is None:
            raise PreventUpdate
        conn.execute(f"UPDATE samples SET {recentlySwitchDataIndex} = ? WHERE ms_file_label = ?",
                     (recentlySwitchStatus, recentlySwitchRow['ms_file_label']))
        logger.info(f"Updated {recentlySwitchDataIndex} for {recentlySwitchRow['ms_file_label']}: {recentlySwitchStatus}")

        # Enforce dependencies
        if recentlySwitchDataIndex == 'use_for_analysis' and recentlySwitchStatus is True:
            # Analysis = True => Processing must be True
            conn.execute("UPDATE samples SET use_for_processing = TRUE WHERE ms_file_label = ?",
                         (recentlySwitchRow['ms_file_label'],))
            logger.info(f"Auto-enabled use_for_processing for {recentlySwitchRow['ms_file_label']}")

        if recentlySwitchDataIndex == 'use_for_processing' and recentlySwitchStatus is False:
             # Processing = False => Analysis must be False
            conn.execute("UPDATE samples SET use_for_analysis = FALSE WHERE ms_file_label = ?",
                         (recentlySwitchRow['ms_file_label'],))
            logger.info(f"Auto-disabled use_for_analysis for {recentlySwitchRow['ms_file_label']}")

    return {'action': 'reload', 'timestamp': time.time()}


def _ms_files_table(section_context, processing_output, processed_action, pagination, filter_, sorter, filterOptions,
                   processing_type, wdir):
    if section_context and section_context['page'] != 'MS-Files':
        raise PreventUpdate
    if not wdir:
        raise PreventUpdate
    
    # Skip refresh if triggered by action store for certain actions (cell already updated visually)
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'ms-table-action-store':
            trigger_value = ctx.triggered[0].get('value')
            # Skip for empty/no_update (color change), or edit actions (cell updates handled locally)
            if not trigger_value or trigger_value == {} or trigger_value.get('action') == 'edit':
                raise PreventUpdate

    start_time = time.perf_counter()
    if pagination:
        page_size = pagination['pageSize']
        current = pagination['current']

        with duckdb_connection(wdir) as conn:
            if conn is None:
                # Database is locked (e.g., processing was just cancelled)
                raise PreventUpdate
            schema = conn.execute("DESCRIBE samples").pl()
        column_types = {r["column_name"]: r["column_type"] for r in schema.to_dicts()}
        where_sql, params = build_where_and_params(filter_, filterOptions)
        order_by_sql = build_order_by(sorter, column_types, tie=('ms_file_label', 'ASC'))  # '' if there is no valid sorter

        sql = f"""
        WITH filtered AS (
          SELECT *
          FROM samples
          {where_sql}
        ),
        paged AS (
          SELECT *, COUNT(*) OVER() AS __total__
          FROM filtered
          {(' ' + order_by_sql) if order_by_sql else ''}
          LIMIT ? OFFSET ?
        )
        SELECT * FROM paged;
        """

        params_paged = params + [page_size, (current - 1) * page_size]

        with duckdb_connection(wdir) as conn:
            dfpl = conn.execute(sql, params_paged).pl()

        # total rows:
        number_records = int(dfpl["__total__"][0]) if len(dfpl) else 0
        max_page = max(math.ceil(number_records / page_size), 1)
        current = min(max(current, 1), max_page)

        # If we just removed the page we were on, re-query for the new page index
        if params_paged[-1] != (current - 1) * page_size:
            params_paged = params + [page_size, (current - 1) * page_size]
            with duckdb_connection(wdir) as conn:
                dfpl = conn.execute(sql, params_paged).pl()
            number_records = int(dfpl["__total__"][0]) if len(dfpl) else 0

        with (duckdb_connection(wdir) as conn):
            st_custom_items = filterOptions['sample_type'].get('filterCustomItems')
            sample_type_filters = conn.execute("SELECT DISTINCT sample_type "
                                               "FROM samples "
                                               "ORDER BY sample_type ASC").df()['sample_type'].to_list()
            if st_custom_items != sample_type_filters:
                output_filterOptions = filterOptions.copy()
                output_filterOptions['sample_type']['filterCustomItems'] = (sample_type_filters
                                                                            if sample_type_filters != [None] else
                                                                            [])
            else:
                output_filterOptions = dash.no_update

        data = dfpl.with_columns(
            pl.col('color').map_elements(
                lambda value: {
                    'content': value,
                    'variant': 'filled',
                    'custom': {'color': value},
                    'style': {'background': value, 'width': '70px'}
                },
                return_dtype=pl.Struct({
                    'content': pl.String,
                    'variant': pl.String,
                    'custom': pl.Struct({'color': pl.String}),
                    'style': pl.Struct({'background': pl.String, 'width': pl.String})
                }),
                skip_nulls=False,
            ).alias('color'),
            pl.col('use_for_optimization').map_elements(
                lambda value: {'checked': value, 'checkedChildren': 'YES', 'unCheckedChildren': 'NO'},
                return_dtype=pl.Struct({
                    'checked': pl.Boolean,
                    'checkedChildren': pl.String,
                    'unCheckedChildren': pl.String
                })
            ).alias('use_for_optimization'),
            pl.col('use_for_processing').map_elements(
                lambda value: {'checked': value, 'checkedChildren': 'YES', 'unCheckedChildren': 'NO'},
                return_dtype=pl.Struct({
                    'checked': pl.Boolean,
                    'checkedChildren': pl.String,
                    'unCheckedChildren': pl.String
                })
            ).alias('use_for_processing'),
            pl.col('use_for_analysis').map_elements(
                lambda value: {'checked': value, 'checkedChildren': 'YES', 'unCheckedChildren': 'NO'},
                return_dtype=pl.Struct({
                    'checked': pl.Boolean,
                    'checkedChildren': pl.String,
                    'unCheckedChildren': pl.String
                })
            ).alias('use_for_analysis'),
            pl.col('ms_type').map_elements(
                lambda value: value.upper() if isinstance(value, str) else value,
                return_dtype=pl.String,
            ).alias('ms_type'),
        )
        end_time = time.perf_counter()
        time.sleep(max(0.0, min(len(data), 0.5 - (end_time - start_time))))

        return [
            data.to_dicts(),
            [],
            {**pagination,
             'total': number_records,
             'current': current,
             'pageSizeOptions': sorted(set([5, 10, 15, 25, 50, 100] + ([number_records] if number_records else [])))},
            output_filterOptions
        ]
    return dash.no_update


def _confirm_and_delete(okCounts, selectedRows, clickedKey, wdir, workspace_status):
    if okCounts is None:
        raise PreventUpdate
    if not wdir:
        raise PreventUpdate
    
    activate_workspace_logging(wdir)

    if clickedKey == "delete-selected" and not selectedRows:
        ms_table_action_store = {'action': 'delete', 'status': 'failed'}
        total_removed = 0
    elif clickedKey == "delete-selected":
        remove_ms_files = [
            row["ms_file_label"]
            for row in selectedRows
            if isinstance(row, dict) and row.get("ms_file_label")
        ]
        remove_ms_files = sorted(set(remove_ms_files))

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            try:
                conn.execute("BEGIN")
                if remove_ms_files:
                    conn.execute("DELETE FROM ms1_data WHERE ms_file_label IN ?", (remove_ms_files,))
                    conn.execute("DELETE FROM ms2_data WHERE ms_file_label IN ?", (remove_ms_files,))
                    conn.execute("DELETE FROM chromatograms WHERE ms_file_label IN ?", (remove_ms_files,))
                    conn.execute("DELETE FROM results WHERE ms_file_label IN ?", (remove_ms_files,))
                    conn.execute("DELETE FROM samples WHERE ms_file_label IN ?", (remove_ms_files,))
                conn.execute("COMMIT")
                conn.execute("CHECKPOINT")
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"Error deleting selected MS files: {e}", exc_info=True)
                return (fac.AntdNotification(
                            message="Failed to delete MS files",
                            description="No files were deleted.",
                            type="error",
                            duration=4,
                            placement='bottom',
                            showProgress=True,
                            stack=True,
                            style=NOTIFICATION_COMPACT_STYLE
                        ),
                        {'action': 'delete', 'status': 'failed'},
                        dash.no_update)

        total_removed = len(remove_ms_files)
        ms_table_action_store = {'action': 'delete', 'status': 'success'}
    else:
        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            total_removed_q = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
            ms_table_action_store = {'action': 'delete', 'status': 'failed'}
            total_removed = 0
            if total_removed_q:
                total_removed = total_removed_q[0]

                conn.execute("BEGIN")
                for t in ("ms1_data", "ms2_data", "chromatograms", "results", "samples"):
                    conn.execute(f"TRUNCATE {t}")
                conn.execute("COMMIT")
                conn.execute("CHECKPOINT")
                conn.execute("ANALYZE")

                ms_table_action_store = {'action': 'delete', 'status': 'success'}
        
        # Compact database only when clearing the entire table (outside 'with' block)
        if ms_table_action_store.get('status') == 'success':
            compact_success, compact_msg = compact_database(wdir)
            if compact_success:
                logger.info(f"Database compacted after clearing MS files: {compact_msg}")
            else:
                logger.warning(f"Failed to compact database: {compact_msg}")
    
    if total_removed > 0:
        logger.info(f"Deleted {total_removed} MS-Files.")
    
    # Get updated file count for workspace-status
    updated_count = 0
    with duckdb_connection(wdir) as conn:
        if conn is not None:
            result = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
            updated_count = result[0] if result else 0

    new_status = workspace_status or {}
    new_status['ms_files_count'] = updated_count

    return (fac.AntdNotification(message="MS files deleted" if total_removed > 0 else "Failed to delete MS files",
                                 description=f"Deleted {total_removed} files",
                                 type="success" if total_removed > 0 else "error",
                                 duration=3,
                                 placement='bottom',
                                 showProgress=True,
                                 stack=True,
                                 style=NOTIFICATION_COMPACT_STYLE
                                 ),
            ms_table_action_store,
            dash.no_update,
            new_status)


def _save_table_on_edit(row_edited, column_edited, wdir):
    if not row_edited or not column_edited or not wdir:
        raise PreventUpdate

    ms_file_label = row_edited.get('ms_file_label')
    if not ms_file_label:
        raise PreventUpdate

    new_value = row_edited.get(column_edited)

    allowed_columns = {
        "label",
        "sample_type",
        *GROUP_COLUMNS,
    }
    if column_edited not in allowed_columns:
        raise PreventUpdate

    try:
        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            conn.execute(f"UPDATE samples SET {column_edited} = ? WHERE ms_file_label = ?",
                         (new_value, ms_file_label))
            ms_table_action_store = {'action': 'edit', 'status': 'success'}
            logger.info(f"Updated metadata for {ms_file_label}: {column_edited} = {new_value}")
        return fac.AntdNotification(message="Changes saved",
                                    type="success",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True,
                                    style=NOTIFICATION_COMPACT_STYLE
                                    ), ms_table_action_store
    except Exception as e:
        logger.error(f"Error updating metadata: {e}", exc_info=True)
        ms_table_action_store = {'action': 'edit', 'status': 'failed'}
        return fac.AntdNotification(message="Failed to save changes",
                                    description=f"Error: {str(e)}",
                                    type="error",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True,
                                    style=NOTIFICATION_COMPACT_STYLE
                                    ), ms_table_action_store

def callbacks(cls, app, fsc, cache, args_namespace):
    # Clientside callback to toggle data-dependent UI visibility based on workspace-status
    # This runs in the browser for instant UI updates without server roundtrips
    app.clientside_callback(
        """(status) => {
            if (!status) {
                return [{'display': 'none'}, {'display': 'none'}, {'paddingTop': '1rem', 'display': 'none'}, {'display': 'block'}];
            }
            const hasFiles = (status.ms_files_count || 0) > 0;
            const showStyle = hasFiles ? 'block' : 'none';
            const hideStyle = hasFiles ? 'none' : 'block';
            const flexStyle = hasFiles ? 'flex' : 'none';
            return [
                {'display': showStyle},
                {'display': flexStyle, 'gap': '8px'},
                {'paddingTop': '1rem', 'display': showStyle},
                {'display': hideStyle}
            ];
        }""",
        [
            Output('ms-files-load-metadata-wrapper', 'style'),
            Output('ms-files-data-actions-wrapper', 'style'),
            Output('ms-files-table-container', 'style'),
            Output('ms-files-empty-state', 'style'),
        ],
        Input('workspace-status', 'data'),
    )

    @app.callback(
        Output('color-picker-modal', 'visible'),
        Output('hex-color-picker', 'color'),
        Output('hex-color-input', 'value'),
        Output('hex-color-preview', 'style'),
        Output('hex-color-error', 'children'),
        Input('ms-files-table', 'nClicksButton'),
        State('ms-files-table', 'clickedCustom'),
        prevent_initial_call=True
    )
    def open_color_picker(nClicksButton, clickedCustom):
        if not clickedCustom or 'color' not in clickedCustom:
            raise PreventUpdate
        color = clickedCustom['color']
        preview_style = {
            'width': '30px',
            'height': '30px',
            'borderRadius': '4px',
            'border': '1px solid #d9d9d9',
            'backgroundColor': color
        }
        return True, color, color, preview_style, ''

    @app.callback(
        Output('hex-color-input', 'value', allow_duplicate=True),
        Output('hex-color-preview', 'style', allow_duplicate=True),
        Input('hex-color-picker', 'color'),
        prevent_initial_call=True
    )
    def sync_picker_to_input(color):
        """Sync color picker selection to hex input field."""
        if not color:
            raise PreventUpdate
        preview_style = {
            'width': '30px',
            'height': '30px',
            'borderRadius': '4px',
            'border': '1px solid #d9d9d9',
            'backgroundColor': color
        }
        return color, preview_style

    @app.callback(
        Output('hex-color-picker', 'color', allow_duplicate=True),
        Output('hex-color-preview', 'style', allow_duplicate=True),
        Output('hex-color-error', 'children', allow_duplicate=True),
        Input('hex-color-input', 'value'),
        prevent_initial_call=True
    )
    def sync_input_to_picker(hex_value):
        """Sync hex input to color picker with validation."""
        import re
        if hex_value is None:
            raise PreventUpdate
        
        hex_value = hex_value.strip()
        
        # Add # if missing
        if hex_value and not hex_value.startswith('#'):
            hex_value = '#' + hex_value
        
        # Validate hex color format
        if not hex_value:
            raise PreventUpdate
        
        if not re.match(r'^#[0-9A-Fa-f]{6}$', hex_value):
            # Invalid hex code
            preview_style = {
                'width': '30px',
                'height': '30px',
                'borderRadius': '4px',
                'border': '1px solid #ff4d4f',
                'backgroundColor': '#f5f5f5'
            }
            return dash.no_update, preview_style, 'Invalid hex code (use #RRGGBB)'
        
        # Valid hex code
        preview_style = {
            'width': '30px',
            'height': '30px',
            'borderRadius': '4px',
            'border': '1px solid #d9d9d9',
            'backgroundColor': hex_value
        }
        return hex_value, preview_style, ''

    # Clientside callback for INSTANT UI update (no server round-trip)
    app.clientside_callback(
        """
        function(okCounts, color, recentlyButtonClickedRow, data) {
            if (!okCounts || !color || !recentlyButtonClickedRow || !data) {
                return window.dash_clientside.no_update;
            }
            
            // Find the row index
            const msFileLabel = recentlyButtonClickedRow.ms_file_label;
            let rowIndex = -1;
            for (let i = 0; i < data.length; i++) {
                if (data[i].ms_file_label === msFileLabel) {
                    rowIndex = i;
                    break;
                }
            }
            
            if (rowIndex === -1) {
                return window.dash_clientside.no_update;
            }
            
            // Create updated data with only the color cell changed
            const newData = [...data];
            newData[rowIndex] = {
                ...newData[rowIndex],
                color: {
                    content: color,
                    variant: 'filled',
                    custom: { color: color },
                    style: { background: color, width: '70px' }
                }
            };
            
            return newData;
        }
        """,
        Output('ms-files-table', 'data', allow_duplicate=True),
        Input('color-picker-modal', 'okCounts'),
        State('hex-color-picker', 'color'),
        State('ms-files-table', 'recentlyButtonClickedRow'),
        State('ms-files-table', 'data'),
        prevent_initial_call=True
    )

    # Server callback for database persistence (runs in background after UI update)
    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Input('color-picker-modal', 'okCounts'),
        State('hex-color-picker', 'color'),
        State('ms-files-table', 'recentlyButtonClickedRow'),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def save_color_to_db(okCounts, color, recentlyButtonClickedRow, wdir):
        return _save_color_to_db(okCounts, color, recentlyButtonClickedRow, wdir)

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('ms-table-action-store', 'data', allow_duplicate=True),
        Output('ms-files-table-spin', 'spinning', allow_duplicate=True),

        Input("ms-options", "nClicks"),
        State("ms-options", "clickedKey"),
        State('wdir', 'data'),
        background=True,
        running=[
            (Output("ms-files-table-spin", "spinning"), True, False),
        ],
        prevent_initial_call=True
    )
    def genere_color_map(nClicks, clickedKey, wdir):
        return _genere_color_map(nClicks, clickedKey, wdir)

    @app.callback(
        Output('ms-table-action-store', 'data', allow_duplicate=True),
        Input('ms-files-table', 'recentlySwitchDataIndex'),
        Input('ms-files-table', 'recentlySwitchStatus'),
        Input('ms-files-table', 'recentlySwitchRow'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def save_switch_changes(recentlySwitchDataIndex, recentlySwitchStatus, recentlySwitchRow, wdir):
        return _save_switch_changes(recentlySwitchDataIndex, recentlySwitchStatus, recentlySwitchRow, wdir)

    @app.callback(
        Output("download-ms-files-csv", "data"),
        Input("download-ms-template-btn", "nClicks"),
        Input("download-ms-files-btn", "nClicks"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def download_ms_files(template_clicks, list_clicks, wdir):
        from ..duckdb_manager import duckdb_connection_mint

        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        ws_name = "workspace"
        if wdir:
            try:
                ws_key = Path(wdir).stem
                with duckdb_connection_mint(Path(wdir).parent.parent) as mint_conn:
                    if mint_conn is not None:
                        ws_row = mint_conn.execute("SELECT name FROM workspaces WHERE key = ?", [ws_key]).fetchone()
                        if ws_row is not None:
                            ws_name = ws_row[0]
            except Exception:
                pass

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'download-ms-template-btn':
            filename = f"{T.today()}-MINT__{ws_name}-ms_files_template.csv"
            return dcc.send_string(MS_METADATA_TEMPLATE_CSV, filename)

        if trigger == 'download-ms-files-btn':
            if not wdir:
                raise PreventUpdate
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                df = conn.execute("SELECT * FROM samples").df()

            filename = f"{T.today()}-MINT__{ws_name}-ms_files.csv"
            return dcc.send_data_frame(df.to_csv, filename, index=False)

        raise PreventUpdate


    @app.callback(
        Output("ms-files-table", "data"),
        Output("ms-files-table", "selectedRowKeys"),
        Output("ms-files-table", "pagination"),
        Output("ms-files-table", "filterOptions"),

        Input('section-context', 'data'),
        Input("ms-table-action-store", "data"),
        Input("processed-action-store", "data"), # from explorer
        Input('ms-files-table', 'pagination'),
        Input('ms-files-table', 'filter'),
        Input('ms-files-table', 'sorter'),
        State('ms-files-table', 'filterOptions'),
        State("processing-type-store", "data"), # from explorer
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def ms_files_table(section_context, processing_output, processed_action, pagination, filter_, sorter, filterOptions,
                       processing_type, wdir):
        return _ms_files_table(section_context, processing_output, processed_action, pagination, filter_, sorter, filterOptions,
                       processing_type, wdir)

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input('section-context', 'data'),
        Input("wdir", "data"),
        prevent_initial_call=True,
    )
    def warn_missing_workspace(section_context, wdir):
        if not section_context or section_context.get('page') != 'MS-Files':
            raise PreventUpdate
        if wdir:
            raise PreventUpdate
        return fac.AntdNotification(
            message="Activate a workspace",
            description="Please select or create a workspace before working with MS files.",
            type="warning",
            duration=4,
            placement='bottom',
            showProgress=True,
            stack=True,
            style=NOTIFICATION_COMPACT_STYLE,
        )

    @app.callback(
        Output("delete-confirmation-modal", "visible"),
        Output("delete-confirmation-modal", "children"),

        Input("ms-options", "nClicks"),
        State("ms-options", "clickedKey"),
        State('ms-files-table', 'selectedRows'),
        prevent_initial_call=True
    )
    def toggle_modal(nClicks, clickedKey, selectedRows):
        ctx = dash.callback_context
        if (
                not ctx.triggered or
                not nClicks or
                not clickedKey or
                clickedKey not in ['delete-selected', 'delete-all']
        ):
            raise PreventUpdate

        if clickedKey == "delete-selected":
            if not selectedRows:
                raise PreventUpdate

        children = fac.AntdFlex(
            [
                fac.AntdText("This action will delete selected MS-files and cannot be undone?",
                             strong=True),
                fac.AntdText("Are you sure you want to delete the selected MS-files?")
            ],
            vertical=True,
        )
        if clickedKey == "delete-all":
            children = fac.AntdFlex(
                [
                    fac.AntdText("This action will delete ALL MS-files and cannot be undone?",
                                 strong=True, type="danger"),
                    fac.AntdText("Are you sure you want to delete the ALL MS-files?")
                ],
                vertical=True,
            )
        return True, children

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("ms-table-action-store", "data", allow_duplicate=True),
        Output("ms-files-table-spin", "spinning"),
        Output("workspace-status", "data", allow_duplicate=True),

        Input("delete-confirmation-modal", "okCounts"),
        State('ms-files-table', 'selectedRows'),
        State("ms-options", "clickedKey"),
        State("wdir", "data"),
        State("workspace-status", "data"),
        background=True,
        running=[
            (Output("ms-files-table-spin", "spinning"), True, False),
            (Output("delete-confirmation-modal", "confirmLoading"), True, False),
        ],
        prevent_initial_call=True,
    )
    def confirm_and_delete(okCounts, selectedRows, clickedKey, wdir, workspace_status):
        return _confirm_and_delete(okCounts, selectedRows, clickedKey, wdir, workspace_status)

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("ms-table-action-store", "data", allow_duplicate=True),

        Input("ms-files-table", "recentlyChangedRow"),
        State("ms-files-table", "recentlyChangedColumn"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def save_table_on_edit(row_edited, column_edited, wdir):
        return _save_table_on_edit(row_edited, column_edited, wdir)

    @app.callback(
        Output('ms-files-tour-empty', 'current'),
        Output('ms-files-tour-empty', 'open'),
        Output('ms-files-tour-full', 'current'),
        Output('ms-files-tour-full', 'open'),
        Input('ms-files-tour-icon', 'nClicks'),
        State('workspace-status', 'data'),
        prevent_initial_call=True,
    )
    def ms_files_tour(n_clicks, workspace_status):
        has_files = workspace_status and (workspace_status.get('ms_files_count', 0) or 0) > 0
        if has_files:
            # Open full tour, keep empty tour closed
            return 0, False, 0, True
        else:
            # Open empty tour, keep full tour closed
            return 0, True, 0, False

    @app.callback(
        Output('ms-files-tour-hint', 'open'),
        Output('ms-files-tour-hint', 'current'),
        Input('ms-tour-hint-store', 'data'),
    )
    def ms_hint_sync(store_data):
        if not store_data:
            raise PreventUpdate
        return store_data.get('open', True), 0

    @app.callback(
        Output('ms-tour-hint-store', 'data'),
        Input('ms-files-tour-hint', 'closeCounts'),
        Input('ms-files-tour-icon', 'nClicks'),
        State('ms-tour-hint-store', 'data'),
        prevent_initial_call=True,
    )
    def ms_hide_hint(close_counts, n_clicks, store_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'ms-files-tour-icon':
            return {'open': False}

        if close_counts:
            return {'open': False}

        return store_data or {'open': True}
