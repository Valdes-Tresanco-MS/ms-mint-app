import logging
import base64
import math

import dash
import feffery_antd_components as fac
import polars as pl
from multiprocessing import cpu_count
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .. import tools as T
from ..duckdb_manager import duckdb_connection, build_where_and_params, build_order_by, compact_database
from ..plugin_interface import PluginInterface
from ..logging_setup import activate_workspace_logging
from . import targets_asari
from .target_optimization import _get_cpu_help_text

_label = "Targets"
# Template column headers and descriptions for quick downloads
TARGET_TEMPLATE_COLUMNS = [
    'peak_label',
    'peak_selection',
    'bookmark',
    'mz_mean',
    'mz_width',
    'mz',
    'rt',
    'rt_min',
    'rt_max',
    'rt_unit',
    'intensity_threshold',
    'polarity',
    'filterLine',
    'ms_type',
    'category',
    'score',
    'notes',
    'source',
]
TARGET_TEMPLATE_DESCRIPTIONS = [
    'Unique metabolite/feature name',
    'True if selected for analysis',
    'True if bookmarked',
    'Mean m/z (centroid)',
    'm/z window or tolerance (ppm)',
    'Precursor m/z (MS2)',
    'Retention time (default: in seconds)',
    'Lower RT bound (default: in seconds)',
    'Upper RT bound (default: in seconds)',
    'RT unit (e.g. s or min; default: in seconds)',
    'Intensity cutoff (anything lower than this value is considered zero)',
    'Polarity (Positive or Negative)',
    'Filter ID for MS2 scans',
    'MS1 or MS2',
    'Category',
    'Score',
    'Free-form notes',
    'Data source or file',
]
TARGET_TEMPLATE_CSV = ",".join(TARGET_TEMPLATE_COLUMNS) + "\n" + ",".join(TARGET_TEMPLATE_DESCRIPTIONS) + "\n"
TARGET_DESCRIPTION_MAP = dict(zip(TARGET_TEMPLATE_COLUMNS, TARGET_TEMPLATE_DESCRIPTIONS))

logger = logging.getLogger(__name__)


class TargetsPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 4
        logger.info(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)

    def outputs(self):
        return None


_layout = html.Div(
    [
        fac.AntdFlex(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'Targets', level=4, style={'margin': '0'}
                        ),
                        fac.AntdTooltip(
                            fac.AntdIcon(
                                id='targets-tour-icon',
                                icon='pi-info',
                                style={"cursor": "pointer", 'paddingLeft': '10px'},
                                **{'aria-label': 'Show tutorial'},
                            ),
                            title='Show tutorial'
                        ),
                        fac.AntdTooltip(
                            fac.AntdButton(
                                'Load Targets',
                                id={
                                    'action': 'file-explorer',
                                    'type': 'targets',
                                },
                                style={'textTransform': 'uppercase', "margin": "0 10px"},
                            ),
                            title="Import a targets file (CSV) from your computer",
                            placement="bottom"
                        ),
                        fac.AntdTooltip(
                            fac.AntdButton(
                                'Untargeted Analysis',
                                id='asari-open-modal-btn',
                                style={'textTransform': 'uppercase', "marginLeft": "20px"},
                            ),
                            title="Automatically detect targets from processed files (using Asari)",
                            placement="bottom"
                        ),
                    ],
                    align='center',
                ),
                fac.AntdFlex(
                    [
                        # Always visible: Download Template
                        fac.AntdTooltip(
                            fac.AntdButton(
                                'Download template',
                                id='download-target-template-btn',
                                icon=fac.AntdIcon(icon='antd-download'),
                                iconPosition='end',
                                style={'textTransform': 'uppercase'},
                            ),
                            title="Download a blank CSV template for targets",
                            placement="bottom"
                        ),
                        # Conditionally visible: Download Targets and Options (only when targets exist)
                        html.Div(
                            [
                                fac.AntdTooltip(
                                    fac.AntdButton(
                                        'Download targets',
                                        id='download-target-list-btn',
                                        icon=fac.AntdIcon(icon='antd-download'),
                                        iconPosition='end',
                                        style={'textTransform': 'uppercase'},
                                    ),
                                    title="Download the current target list as a CSV file",
                                    placement="bottom"
                                ),
                                html.Div(
                                    fac.AntdDropdown(
                                        id='targets-options',
                                        title='Options',
                                        buttonMode=True,
                                        arrow=True,
                                        menuItems=[
                                            {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                                             'key': 'delete-selected'},
                                            {'title': fac.AntdText('Clear table', strong=True, type='danger'), 'key': 'delete-all'},
                                        ],
                                        buttonProps={'style': {'textTransform': 'uppercase'}},
                                    ),
                                    id='targets-options-wrapper',
                                ),
                            ],
                            id='targets-data-actions-wrapper',
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
        html.Div(
            [
                fac.AntdSpin(
                    fac.AntdTable(
                        id='targets-table',
                        containerId='targets-table-container',
                        columns=[
                            {
                                'title': 'Target',
                                'dataIndex': 'peak_label',
                                'width': '260px',
                                'fixed': 'left'
                            },
                            {
                                'title': 'Selection',
                                'dataIndex': 'peak_selection',
                                'width': '150px',
                                'renderOptions': {'renderType': 'switch'},
                            },
                            {
                                'title': 'Bookmark',
                                'dataIndex': 'bookmark',
                                'width': '150px',
                                'renderOptions': {'renderType': 'switch'},
                            },
                            {
                                'title': 'MZ-Mean',
                                'dataIndex': 'mz_mean',
                                'width': '150px',
                            },
                            {
                                'title': 'MZ-Width',
                                'dataIndex': 'mz_width',
                                'width': '150px',
                            },
                            {
                                'title': 'MZ',
                                'dataIndex': 'mz',
                                'width': '150px',
                            },
                            {
                                'title': 'RT',
                                'dataIndex': 'rt',
                                'width': '150px',
                                'editable': True,
                            },
                            {
                                'title': 'RT-min',
                                'dataIndex': 'rt_min',
                                'width': '150px',
                                'editable': True,
                            },
                            {
                                'title': 'RT-max',
                                'dataIndex': 'rt_max',
                                'width': '150px',
                                'editable': True,
                            },
                            {
                                'title': 'RT-Unit',
                                'dataIndex': 'rt_unit',
                                'width': '120px',
                                'editable': True,
                            },
                            {
                                'title': 'Intensity Threshold',
                                'dataIndex': 'intensity_threshold',
                                'width': '200px',
                                'editable': True,
                            },
                            {
                                'title': 'MS-Type',
                                'dataIndex': 'ms_type',
                                'width': '150px',
                            },
                            {
                                'title': 'Polarity',
                                'dataIndex': 'polarity',
                                'width': '120px',
                            },
                            {
                                'title': 'filterLine',
                                'dataIndex': 'filterLine',
                                'width': '300px',
                            },
                            {
                                'title': 'Proton',
                                'dataIndex': 'proton',
                                'width': '120px',
                            },
                            {
                                'title': 'Category',
                                'dataIndex': 'category',
                                'width': '150px',
                            },
                            {
                                'title': 'Score',
                                'dataIndex': 'score',
                                'width': '120px',
                            },
                            {
                                'title': 'Notes',
                                'dataIndex': 'notes',
                                'width': '300px',
                                'editable': True,
                            },
                            {
                                'title': 'Source',
                                'dataIndex': 'source',
                                'width': '200px',
                            },
                        ],
                        titlePopoverInfo={
                            'peak_label': {
                                'title': 'peak_label',
                                'content': TARGET_DESCRIPTION_MAP['peak_label'],
                            },
                            'peak_selection': {
                                'title': 'peak_selection',
                                'content': TARGET_DESCRIPTION_MAP['peak_selection'],
                            },
                            'bookmark': {
                                'title': 'bookmark',
                                'content': TARGET_DESCRIPTION_MAP['bookmark'],
                            },
                            'mz_mean': {
                                'title': 'mz_mean',
                                'content': TARGET_DESCRIPTION_MAP['mz_mean'],
                            },
                            'mz_width': {
                                'title': 'mz_width',
                                'content': TARGET_DESCRIPTION_MAP['mz_width'],
                            },
                            'mz': {
                                'title': 'mz',
                                'content': TARGET_DESCRIPTION_MAP['mz'],
                            },
                            'rt': {
                                'title': 'rt',
                                'content': TARGET_DESCRIPTION_MAP['rt'],
                            },
                            'rt_min': {
                                'title': 'rt_min',
                                'content': TARGET_DESCRIPTION_MAP['rt_min'],
                            },
                            'rt_max': {
                                'title': 'rt_max',
                                'content': TARGET_DESCRIPTION_MAP['rt_max'],
                            },
                            'rt_unit': {
                                'title': 'rt_unit',
                                'content': TARGET_DESCRIPTION_MAP['rt_unit'],
                            },
                            'intensity_threshold': {
                                'title': 'intensity_threshold',
                                'content': TARGET_DESCRIPTION_MAP['intensity_threshold'],
                            },
                            'ms_type': {
                                'title': 'ms_type',
                                'content': TARGET_DESCRIPTION_MAP['ms_type'],
                            },
                            'polarity': {
                                'title': 'polarity',
                                'content': TARGET_DESCRIPTION_MAP['polarity'],
                            },
                            'filterLine': {
                                'title': 'filterLine',
                                'content': TARGET_DESCRIPTION_MAP['filterLine'],
                            },
                            'proton': {
                                'title': 'proton',
                                'content': 'Ion charge state (Neutral, Positive, or Negative)',
                            },
                            'category': {
                                'title': 'category',
                                'content': TARGET_DESCRIPTION_MAP['category'],
                            },
                            'score': {
                                'title': 'score',
                                'content': TARGET_DESCRIPTION_MAP['score'],
                            },
                            'notes': {
                                'title': 'notes',
                                'content': TARGET_DESCRIPTION_MAP['notes'],
                            },
                            'source': {
                                'title': 'source',
                                'content': TARGET_DESCRIPTION_MAP['source'],
                            },
                        },
                        filterOptions={
                            'peak_label': {'filterMode': 'keyword'},
                            'peak_selection': {'filterMode': 'checkbox',
                                               'filterCustomItems': ['True', 'False']},
                            'bookmark': {'filterMode': 'checkbox',
                                         'filterCustomItems': ['True', 'False']},
                            'mz_mean': {'filterMode': 'keyword'},
                            'mz_width': {'filterMode': 'keyword'},
                            'mz': {'filterMode': 'keyword'},
                            'rt': {'filterMode': 'keyword'},
                            'rt_min': {'filterMode': 'keyword'},
                            'rt_max': {'filterMode': 'keyword'},
                            'rt_unit': {'filterMode': 'checkbox',
                                        'filterCustomItems': ['s', 'min']},
                            'intensity_threshold': {'filterMode': 'keyword'},
                            'ms_type': {'filterMode': 'checkbox',
                                        'filterCustomItems': ['ms1', 'ms2']
                                        },
                            'polarity': {'filterMode': 'checkbox',
                                         'filterCustomItems': ['Negative', 'Positive']
                                         },
                            'filterLine': {'filterMode': 'keyword'},
                            'proton': {'filterMode': 'checkbox',
                                       'filterCustomItems': ['Neutral', 'Positive', 'Negative']},
                            'category': {'filterMode': 'checkbox',
                                         # 'filterCustomItems': ['True', 'False']
                                         },
                            'score': {'filterMode': 'keyword'},
                            'notes': {'filterMode': 'keyword'},
                            'source': {'filterMode': 'keyword'},

                        },
                        sortOptions={
                            'sortDataIndexes': ['peak_label', 'peak_selection', 'bookmark', 'mz_mean', 'mz_width',
                                                'mz', 'rt', 'rt_min', 'rt_max',
                                                'intensity_threshold', 'polarity', 'filterLine', 'ms_type',
                                                'category', 'score']},
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
                    id='targets-table-spin',
                    text='Loading data...',
                    size='small',
                    listenPropsMode='exclude',
                    excludeProps=[
                        'targets-table.data',
                        'targets-table.pagination',
                        'targets-table.selectedRowKeys',
                        'targets-table.filterOptions',
                    ],
                )
            ],
            id='targets-table-container',
            style={'paddingTop': '1rem', 'display': 'none'},
        ),
        # Empty state placeholder - shown when no targets
        html.Div(
            fac.AntdFlex(
                [
                    fac.AntdEmpty(
                        description=fac.AntdFlex(
                            [
                                fac.AntdText('No Targets loaded', strong=True, style={'fontSize': '16px'}),
                                fac.AntdText('Click "Load Targets" to import your data or', type='secondary'),
                                fac.AntdText('"Untargeted Analysis" to populate the table with features detected using Asari', type='secondary'),
                            ],
                            vertical=True,
                            align='center',
                            gap='small',
                        ),
                        locale='en-us',
                    ),
                    fac.AntdTooltip(
                        fac.AntdButton(
                            'Load Targets',
                            id={
                                'action': 'file-explorer',
                                'type': 'targets-empty',
                            },
                            size='large',
                            style={'marginTop': '16px', 'textTransform': 'uppercase'},
                        ),
                        title="Import a targets file (CSV) from your computer",
                        placement="bottom"
                    ),
                ],
                vertical=True,
                align='center',
                style={'marginTop': '100px'},
            ),
            id='targets-empty-state',
            style={'display': 'block'},
        ),
        html.Div(id="targets-notifications-container"),

        fac.AntdModal(
            "Are you sure you want to delete the selected targets?",
            title="Delete target",
            id="delete-table-targets-modal",
            renderFooter=True,
            okButtonProps={"danger": True},
            locale='en-us',
        ),
        fac.AntdModal(
            [
                html.Div([
                    html.Div(id='asari-auto-mode-alert'),
                    fac.AntdDivider('Configuration'),
                    fac.AntdForm(
                        [
                            html.Div([
                                    fac.AntdFormItem(
                                        fac.AntdInputNumber(id='asari-multicores', value=cpu_count()//2, min=1, max=cpu_count(), style={'width': '100%'}),
                                        label="CPU",
                                        tooltip="Number of processor cores to use for parallel processing (multicores parameter in Asari)",
                                        hasFeedback=True,
                                        help=f"Selected {cpu_count()//2} / {cpu_count()} cpus",
                                        labelCol={'span': 13}, wrapperCol={'span': 11},
                                        id='asari-multicores-item'
                                    ),
                                fac.AntdFormItem(
                                    fac.AntdSelect(id='asari-mode', options=[{'label': 'Positive', 'value': 'pos'}, {'label': 'Negative', 'value': 'neg'}], value='pos', style={'width': '100%'}),
                                    label="Mode",
                                    tooltip="Ionization mode of the mass spectrometry data",
                                    labelCol={'span': 13}, wrapperCol={'span': 11}
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(id='asari-mz-tolerance', value=5, min=1, style={'width': '100%'}),
                                    label="MZ Width (ppm)",
                                    tooltip="Mass-to-charge ratio tolerance in parts per million (mz_tolerance_ppm parameter in Asari)",
                                    labelCol={'span': 13}, wrapperCol={'span': 11}
                                ),
                            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '10px'}),
                            
                            html.Div([
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(id='asari-snr', value=20, min=1, style={'width': '100%'}),
                                    label="Signal/Noise Ratio",
                                    tooltip="Peak height at least X fold over local noise",
                                    labelCol={'span': 13}, wrapperCol={'span': 11}
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(id='asari-min-peak-height', value=100000, min=0, style={'width': '100%'}),
                                    label="Min Peak Height",
                                    tooltip="Minimal peak height",
                                    labelCol={'span': 13}, wrapperCol={'span': 11}
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(id='asari-min-timepoints', value=6, min=1, style={'width': '100%'}),
                                    label="Min Timepoints",
                                    tooltip="Minimal number of data points in elution profile",
                                    labelCol={'span': 13}, wrapperCol={'span': 11}
                                ),
                            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '10px'}),
                            
                            html.Div([
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(id='asari-cselectivity', value=1, min=0, max=1, step=0.01, style={'width': '100%'}),
                                    label="cSelectivity",
                                    tooltip="How distinct chromatographic elution peaks are",
                                    labelCol={'span': 13}, wrapperCol={'span': 11}
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(id='asari-gaussian-shape', value=0.9, min=0, max=1, step=0.1, style={'width': '100%'}),
                                    label="Gaussian Shape",
                                    tooltip="Min cutoff of goodness of fitting to Gauss model",
                                    labelCol={'span': 13}, wrapperCol={'span': 11}
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(id='asari-detection-rate', value=90, min=0, max=100, step=1, style={'width': '100%'}),
                                    label="Detection Rate (%)",
                                    tooltip="Filter features detected in at least X% of samples",
                                    labelCol={'span': 13}, wrapperCol={'span': 11}
                                ),
                            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '10px'}),
                        ],
                        layout='horizontal'
                    ),
                    html.Div(id='asari-status-container', style={'marginTop': '10px'})
                ], id='asari-configuration-container'),
                
                html.Div([
                    html.H4("Processing Asari Workflow...", style={'marginBottom': '10px'}),
                    fac.AntdText(id='asari-progress-stage', style={'marginBottom': '0.5rem', 'fontWeight': 'bold'}),
                    fac.AntdProgress(id='asari-progress', percent=0, status='active', style={'width': '100%', 'marginBottom': '10px'}),
                    fac.AntdText(id='asari-progress-detail', type='secondary', style={'marginTop': '0.5rem', 'marginBottom': '0.75rem', 'display': 'block'}),
                    
                    html.Div(
                        id='asari-terminal-logs',
                        style={
                            'width': '100%',
                            'height': '200px',
                            'overflowY': 'auto',
                            'backgroundColor': '#f5f5f5',
                            'border': '1px solid #d9d9d9',
                            'borderRadius': '4px',
                            'padding': '10px',
                            'fontFamily': 'monospace',
                            'fontSize': '12px',
                            'whiteSpace': 'pre-wrap',
                            'marginBottom': '15px',
                            'color': '#333'
                        }
                    ),
                    
                    fac.AntdButton(
                        "Cancel",
                        id="cancel-asari-btn",
                        disabled=True,
                        style={
                            'alignText': 'center',
                            'marginTop': '0.25rem',
                        }
                    )
                ], id='asari-progress-container', style={'display': 'none'})
            ],
            title="Untargeted Analysis (via Asari)",
            id="asari-modal",
            visible=False,
            renderFooter=True,
            okText="Run Analysis",
            locale='en-us',
            confirmAutoSpin=True,
            loadingOkText="Processing...",
            maskClosable=False,
            okClickClose=False,
            width=900,
            centered=True,
            styles={'body': {'minHeight': '400px', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'}},
        ),
        # Tour for empty workspace (no targets loaded) - simplified
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'This tutorial shows how to get started with targets.',
                },
                {
                    'title': 'Load targets',
                    'description': 'Click "Load Targets" to import your target list (CSV).',
                    'targetSelector': "[id='{\"action\":\"file-explorer\",\"type\":\"targets\"}']"
                },
                {
                    'title': 'Untargeted Analysis',
                    'description': 'Or auto-detect targets from your MS data using the Asari algorithm.',
                    'targetSelector': '#asari-open-modal-btn'
                },
                {
                    'title': 'Use the template',
                    'description': 'Download the template if you need the expected columns and examples.',
                    'targetSelector': '#download-target-template-btn'
                },
            ],
            id='targets-tour-empty',
            open=False,
            current=0,
        ),
        # Tour for populated workspace (targets loaded) - full tour
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'This tutorial shows how to manage, review, and export targets.',
                },
                {
                    'title': 'Load more targets',
                    'description': 'Click "Load Targets" to add more targets from a CSV file.',
                    'targetSelector': "[id='{\"action\":\"file-explorer\",\"type\":\"targets\"}']"
                },
                {
                    'title': 'Untargeted Analysis',
                    'description': 'Auto-detect additional targets from your MS data using Asari.',
                    'targetSelector': '#asari-open-modal-btn'
                },
                {
                    'title': 'Review and edit',
                    'description': 'Inspect targets, filter/sort columns, and multi-select rows for bulk actions.',
                    'targetSelector': '#targets-table-container'
                },
                {
                    'title': 'Options',
                    'description': 'Delete selected targets or clear the table from the options menu.',
                    'targetSelector': '#targets-options-wrapper'
                },
                {
                    'title': 'Export',
                    'description': 'Download the current target table (server-side filters applied) for review or sharing.',
                    'targetSelector': '#download-target-list-btn'
                },
                {
                    'title': 'Get the template',
                    'description': 'Download the template if you need the expected columns and examples.',
                    'targetSelector': '#download-target-template-btn'
                },
            ],
            id='targets-tour-full',
            open=False,
            current=0,
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Need help?',
                    'description': 'Click the info icon to open a quick tour of the Targets table.',
                    'targetSelector': '#targets-tour-icon',
                },
            ],
            mask=False,
            placement='rightTop',
            open=False,
            current=0,
            id='targets-tour-hint',
            className='targets-tour-hint',
            style={
                'background': '#ffffff',
                'border': '0.5px solid #1677ff',
                'boxShadow': '0 6px 16px rgba(0,0,0,0.15), 0 0 0 1px rgba(22,119,255,0.2)',
                'opacity': 1,
            },
        ),
        dcc.Store(id="targets-action-store"),
        dcc.Store(id="targets-tour-hint-store", data={'open': False}, storage_type='local'),
        dcc.Download('download-targets-csv'),
    ],
)


def _targets_table(section_context, pagination, filter_, sorter, filterOptions, wdir):
    if section_context and section_context['page'] != 'Targets':
        logger.debug(f"_targets_table: PreventUpdate because current page is {section_context.get('page')}")
        raise PreventUpdate
    if not wdir:
        logger.debug("_targets_table: PreventUpdate because wdir is not set")
        raise PreventUpdate
    
    # Skip refresh if triggered by action store for edit actions (cell already updated visually)
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'targets-action-store':
            trigger_value = ctx.triggered[0].get('value')
            # Skip for edit actions (cell updates are handled locally by AntdTable)
            if trigger_value and trigger_value.get('action') == 'edit':
                raise PreventUpdate

    if pagination:
        page_size = pagination['pageSize']
        current = pagination['current']

        with duckdb_connection(wdir) as conn:
            if conn is None:
                raise PreventUpdate
            schema = conn.execute("DESCRIBE targets").pl()
        column_types = {r["column_name"]: r["column_type"] for r in schema.to_dicts()}
        where_sql, params = build_where_and_params(filter_, filterOptions)
        order_by_sql = build_order_by(sorter, column_types, tie=('peak_label', 'ASC'))
        if not order_by_sql:
            order_by_sql = 'ORDER BY "mz_mean" ASC, "peak_label" COLLATE NOCASE ASC'

        sql = f"""
                    WITH filtered AS (
                      SELECT *
                      FROM targets
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
            if conn is None:
                raise PreventUpdate
            dfpl = conn.execute(sql, params_paged).pl()

        data = (
            dfpl
            .with_columns(
                # If peak_selection is null, fall back to bookmark; default False.
                pl.when(pl.col('peak_selection').is_null())
                .then(pl.col('bookmark').fill_null(False))
                .otherwise(pl.col('peak_selection'))
                .cast(pl.Boolean)
                .alias('peak_selection_resolved'),
                pl.col('bookmark').fill_null(False).cast(pl.Boolean).alias('bookmark_resolved'),
                # Round rt, rt_min and rt_max to 1 decimal place for display
                pl.col('rt').round(1).alias('rt'),
                pl.col('rt_min').round(1).alias('rt_min'),
                pl.col('rt_max').round(1).alias('rt_max'),
                # Uppercase ms_type for display consistency (cast from cat to string first)
                pl.col('ms_type').cast(pl.String).str.to_uppercase().alias('ms_type'),
            )
            .drop(['peak_selection', 'bookmark'])
        )

        # total rows:
        number_records = int(data["__total__"][0]) if len(data) else 0
        
        # If result is empty but there are records, we may be on a page beyond the last
        if len(data) == 0 and number_records == 0:
            # Check if there are actually any records matching the filter
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                count_sql = f"SELECT COUNT(*) FROM targets {where_sql}"
                total_count = conn.execute(count_sql, params).fetchone()[0]
                if total_count > 0:
                    number_records = total_count
        
        # Calculate max page and adjust current if beyond it
        max_page = max(math.ceil(number_records / page_size), 1) if number_records else 1
        current = min(max(current, 1), max_page)

        # If we just removed the page we were on, re-query for the new page index
        if params_paged[-1] != (current - 1) * page_size:
            params_paged = params + [page_size, (current - 1) * page_size]
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                dfpl = conn.execute(sql, params_paged).pl()
            
            data = (
                dfpl
                .with_columns(
                    pl.when(pl.col('peak_selection').is_null())
                    .then(pl.col('bookmark').fill_null(False))
                    .otherwise(pl.col('peak_selection'))
                    .cast(pl.Boolean)
                    .alias('peak_selection_resolved'),
                    pl.col('bookmark').fill_null(False).cast(pl.Boolean).alias('bookmark_resolved'),
                    pl.col('rt').round(1).alias('rt'),
                    pl.col('rt_min').round(1).alias('rt_min'),
                    pl.col('rt_max').round(1).alias('rt_max'),
                    pl.col('ms_type').cast(pl.String).str.to_uppercase().alias('ms_type'),
                )
                .drop(['peak_selection', 'bookmark'])
            )
            number_records = int(data["__total__"][0]) if len(data) else 0

        with (duckdb_connection(wdir) as conn):
            if conn is None:
                raise PreventUpdate
            category_filters = conn.execute(
                "SELECT DISTINCT category FROM targets ORDER BY category ASC"
            ).df()['category'].to_list()

            if not isinstance(filterOptions, dict) or 'category' not in filterOptions:
                output_filterOptions = dash.no_update
            else:
                st_custom_items = filterOptions['category'].get('filterCustomItems')
                if st_custom_items != category_filters:
                    output_filterOptions = filterOptions.copy()
                    output_filterOptions['category']['filterCustomItems'] = (
                        category_filters if category_filters != [None] else []
                    )
                else:
                    output_filterOptions = dash.no_update

        # Convert to dicts and add the checkbox structure AFTER Polars processing
        # This avoids using pl.Object dtype which causes panic in frozen apps
        data_dicts = data.to_dicts()
        for row in data_dicts:
            row['peak_selection'] = {
                'checked': bool(row.pop('peak_selection_resolved', False)),
                'checkedChildren': 'YES',
                'unCheckedChildren': 'NO'
            }
            row['bookmark'] = {
                'checked': bool(row.pop('bookmark_resolved', False)),
                'checkedChildren': 'YES',
                'unCheckedChildren': 'NO'
            }

        return [
            data_dicts,
            [],
            {**pagination, 'total': number_records, 'current': current, 'pageSizeOptions': sorted([5, 10, 15, 25, 50,
            100, number_records])},
            output_filterOptions
        ]
    return dash.no_update


def _target_delete(okCounts, selectedRows, clickedKey, wdir, workspace_status):
    if okCounts is None or clickedKey not in ['delete-selected', 'delete-all']:
        logger.debug(f"_target_delete: PreventUpdate because okCounts={okCounts}, clickedKey={clickedKey}")
        raise PreventUpdate
    if not wdir:
        logger.debug("_target_delete: PreventUpdate because wdir is not set")
        raise PreventUpdate
    
    activate_workspace_logging(wdir)

    if clickedKey == "delete-selected" and not selectedRows:
        targets_action_store = {'action': 'delete', 'status': 'failed'}
        total_removed = 0
    elif clickedKey == "delete-selected":
        remove_targets = [row['peak_label'] for row in selectedRows]

        with duckdb_connection(wdir) as conn:
            if conn is None:
                logger.debug("_target_delete: PreventUpdate because database connection is None")
                raise PreventUpdate
            try:
                conn.execute("BEGIN")
                conn.execute("DELETE FROM targets WHERE peak_label IN ?", (remove_targets,))
                conn.execute("DELETE FROM chromatograms WHERE peak_label IN ?", (remove_targets,))
                conn.execute("DELETE FROM results WHERE peak_label IN ?", (remove_targets,))
                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"Error deleting selected targets: {e}", exc_info=True)
                return (fac.AntdNotification(
                            message="Failed to delete targets",
                            description=f"Error: {e}",
                            type="error",
                            duration=4,
                            placement='bottom',
                            showProgress=True,
                            stack=True
                        ),
                        {'action': 'delete', 'status': 'failed'},
                        dash.no_update)
        total_removed = len(remove_targets)
        targets_action_store = {'action': 'delete', 'status': 'success'}
    elif clickedKey == "delete-all":
        with duckdb_connection(wdir) as conn:
            if conn is None:
                logger.debug("_target_delete(delete-all): PreventUpdate because database connection is None")
                raise PreventUpdate
            total_removed_q = conn.execute("SELECT COUNT(*) FROM targets").fetchone()
            targets_action_store = {'action': 'delete', 'status': 'failed'}
            total_removed = 0
            should_compact = False
            if total_removed_q:
                total_removed = total_removed_q[0]

                try:
                    # Check for derived data (chromatograms/results) to decide if compaction is needed
                    # Only compact if we are deleting significant data to avoid "huge DB with only MS files"
                    derived_count = conn.execute("SELECT (SELECT COUNT(*) FROM chromatograms) + (SELECT COUNT(*) FROM results)").fetchone()[0]
                    should_compact = derived_count > 0
                    if should_compact:
                         logger.info(f"Targets delete-all: Found {derived_count} derived rows. Compaction will be triggered.")

                    conn.execute("BEGIN")
                    conn.execute("DELETE FROM targets")
                    conn.execute("DELETE FROM chromatograms")
                    conn.execute("DELETE FROM results")
                    conn.execute("COMMIT")
                    targets_action_store = {'action': 'delete', 'status': 'success'}
                except Exception as e:
                    conn.execute("ROLLBACK")
                    logger.error(f"Error deleting all targets: {e}", exc_info=True)
                    return (fac.AntdNotification(
                                message="Failed to delete targets",
                                description=f"Error: {e}",
                                type="error",
                                duration=4,
                                placement='bottom',
                                showProgress=True,
                                stack=True
                            ),
                            {'action': 'delete', 'status': 'failed'},
                            dash.no_update)
        
        # Compact database only when clearing the entire table (outside the 'with' block)
        if targets_action_store.get('status') == 'success' and should_compact:
            compact_success, compact_msg = compact_database(wdir)
            if compact_success:
                logger.info(f"Database compacted after clearing targets: {compact_msg}")
            else:
                logger.warning(f"Failed to compact database: {compact_msg}")


                logger.warning(f"Failed to compact database: {compact_msg}")

    if total_removed > 0:
        logger.info(f"Deleted {total_removed} targets.")

    # Update workspace status
    new_status = workspace_status or {}
    with duckdb_connection(wdir) as conn:
        if conn:
            count = conn.execute("SELECT COUNT(*) FROM targets").fetchone()
            new_status['targets_count'] = count[0] if count else 0

    return (fac.AntdNotification(message="Delete Targets",
                                 description=f"Deleted {total_removed} targets",
                                 type="success" if total_removed > 0 else "error",
                                 duration=3,
                                 placement='bottom',
                                 showProgress=True,
                                 stack=True
                                 ),
            targets_action_store,
            dash.no_update,
            new_status)


def _save_target_table_on_edit(row_edited, column_edited, wdir):
    if row_edited is None or column_edited is None:
        logger.debug("_save_target_table_on_edit: PreventUpdate because row_edited or column_edited is None")
        raise PreventUpdate

    if not wdir:
        logger.debug("_save_target_table_on_edit: PreventUpdate because wdir is not set")
        raise PreventUpdate

    allowed_columns = {
        "rt",
        "rt_min",
        "rt_max",
        "rt_unit",
        "intensity_threshold",
        "notes",
        "category",
        "score",
    }
    if column_edited not in allowed_columns:
        logger.debug(f"_save_target_table_on_edit: PreventUpdate because column '{column_edited}' is not editable")
        raise PreventUpdate
    try:
        with duckdb_connection(wdir) as conn:
            if conn is None:
                logger.debug("_save_target_table_on_edit: PreventUpdate because database connection is None")
                raise PreventUpdate

            query = f"UPDATE targets SET {column_edited} = ? WHERE peak_label = ?"
            if column_edited == 'sample_type' and row_edited[column_edited] in [None, '']:
                conn.execute(query, ['Unset', row_edited['peak_label']])
            else:
                conn.execute(query, [row_edited[column_edited], row_edited['peak_label']])
            targets_action_store = {'action': 'edit', 'status': 'success'}
            logger.info(f"Updated target {row_edited['peak_label']}: {column_edited} = {row_edited[column_edited]}")
        return fac.AntdNotification(message="Changes saved",
                                    description="Targets saved to disk.",
                                    type="success",
                                    duration=4,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    ), targets_action_store
    except Exception as e:
        logger.error(f"Error updating metadata: {e}", exc_info=True)
        targets_action_store = {'action': 'edit', 'status': 'failed'}
        return fac.AntdNotification(message="Failed to save changes",
                                    description=f"Error: {e}",
                                    type="error",
                                    duration=4,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    ), targets_action_store


def _save_switch_changes(recentlySwitchDataIndex, recentlySwitchStatus, recentlySwitchRow, wdir):
    if recentlySwitchDataIndex is None or recentlySwitchStatus is None or not recentlySwitchRow:
        logger.debug("_save_switch_changes: PreventUpdate because missing switch data")
        raise PreventUpdate

    if not wdir:
        logger.debug("_save_switch_changes: PreventUpdate because wdir is not set")
        raise PreventUpdate

    allowed_switch_columns = {"peak_selection", "bookmark"}
    if recentlySwitchDataIndex not in allowed_switch_columns:
        logger.debug(f"_save_switch_changes: PreventUpdate because column '{recentlySwitchDataIndex}' is not a switch column")
        raise PreventUpdate

    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.debug("_save_switch_changes: PreventUpdate because database connection is None")
            raise PreventUpdate
        conn.execute(f"UPDATE targets SET {recentlySwitchDataIndex} = ? WHERE peak_label = ?",
                     (recentlySwitchStatus, recentlySwitchRow['peak_label']))
        logger.info(f"Updated {recentlySwitchDataIndex} for {recentlySwitchRow['peak_label']}: {recentlySwitchStatus}")


def _run_asari_analysis(ok_counts, wdir, multicores, mz_tol, mode, snr, min_height, min_points, gaussian_shape, cselectivity, detection_rate, set_progress=None):
    if not ok_counts:
         logger.debug("_run_asari_analysis: PreventUpdate because ok_counts is None")
         raise PreventUpdate
         
    if not wdir:
        return dash.no_update, True, fac.AntdAlert(message="No workspace selected.", type="error"), dash.no_update, dash.no_update
    
    activate_workspace_logging(wdir)
        
    # Validate inputs
    if multicores is None or multicores < 1:
        return dash.no_update, True, fac.AntdAlert(message="Invalid CPU cores selected.", type="error"), dash.no_update, dash.no_update
    if mz_tol is None or mz_tol < 1:
        return dash.no_update, True, fac.AntdAlert(message="Invalid MZ Tolerance.", type="error"), dash.no_update, dash.no_update
    if snr is None or snr < 1:
        return dash.no_update, True, fac.AntdAlert(message="Invalid Signal/Noise Ratio.", type="error"), dash.no_update, dash.no_update
    if min_height is None or min_height < 0:
        return dash.no_update, True, fac.AntdAlert(message="Invalid Min Peak Height.", type="error"), dash.no_update, dash.no_update
    if min_points is None or min_points < 1:
         return dash.no_update, True, fac.AntdAlert(message="Invalid Min Timepoints.", type="error"), dash.no_update, dash.no_update
         
    # Gaussian Shape is optional
    if gaussian_shape is not None and (gaussian_shape < 0 or gaussian_shape > 1):
         return dash.no_update, True, fac.AntdAlert(message="Invalid Gaussian Shape. Must be between 0 and 1.", type="error"), dash.no_update, dash.no_update

    # cSelectivity is optional
    if cselectivity is not None and (cselectivity < 0 or cselectivity > 1):
         return dash.no_update, True, fac.AntdAlert(message="Invalid cSelectivity. Must be between 0 and 1.", type="error"), dash.no_update, dash.no_update

    # Detection Rate is optional
    if detection_rate is not None and (detection_rate < 0 or detection_rate > 100):
         return dash.no_update, True, fac.AntdAlert(message="Invalid Detection Rate. Must be between 0% and 100%.", type="error"), dash.no_update, dash.no_update
        
    params = {
        'multicores': multicores,
        'mz_tolerance_ppm': mz_tol,
        'mode': mode,
        'signal_noise_ratio': snr,
        'min_peak_height': min_height,
        'min_timepoints': min_points,
        'gaussian_shape': gaussian_shape,
        'cselectivity': cselectivity,
        'detection_rate': detection_rate
    }
    
    result = targets_asari.run_asari_workflow(wdir, params, set_progress=set_progress)
    
    status_alert = None
    if result['success']:
         status_alert = fac.AntdAlert(message=result['message'], type="success", showIcon=True, closable=True)
         import time
         
         # Get updated targets count for workspace-status refresh
         targets_count = 0
         try:
             with duckdb_connection(wdir) as conn:
                 if conn:
                     targets_count = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
         except Exception:
             pass
         
         # Return 5 values: notification, modal visible, status alert, targets-action-store, workspace-status
         workspace_status = {'targets_count': targets_count, 'timestamp': time.time()}
         return fac.AntdNotification(message="Asari Analysis", description=result['message'], type="success", placement="bottom"), False, status_alert, {'timestamp': time.time()}, workspace_status
    else:
         # Check if this is the "no features" case - use warning style and skip notification
         if result.get('no_features'):
             # Build properly formatted HTML for the alert message
             status_alert = fac.AntdAlert(
                 message=html.Div([
                     html.Strong("No features passed the current filter thresholds."),
                     html.Br(), html.Br(),
                     html.Span("Suggestions:"),
                     html.Ul([
                         html.Li("Lower Detection Rate to 50% or less"),
                         html.Li("Lower Signal/Noise Ratio to 5-10"),
                         html.Li("Lower Min Peak Height to 10,000-50,000"),
                         html.Li("Lower cSelectivity to 0.5-0.7"),
                     ], style={'marginTop': '5px', 'marginBottom': '5px', 'paddingLeft': '20px'}),
                     html.Span("This often happens when blanks or low-quality samples are included."),
                     html.Br(),
                     html.Span("Your mzML files have been kept so the next run will be faster.", style={'fontStyle': 'italic'}),
                 ]),
                 type="warning",
                 showIcon=True,
                 closable=True
             )
             # No notification popup - all info is in the modal
             return dash.no_update, True, status_alert, dash.no_update, dash.no_update
         else:
             status_alert = fac.AntdAlert(message=result['message'], type="error", showIcon=True, closable=True)
             return fac.AntdNotification(message="Analysis failed", description=result['message'], type="error", placement="bottom"), True, status_alert, dash.no_update, dash.no_update


def callbacks(app, fsc=None, cache=None):
    # Clientside callback to toggle targets UI visibility based on workspace-status
    # This runs in the browser for instant UI updates without server roundtrips
    app.clientside_callback(
        """(status) => {
            if (!status) {
                return [{'display': 'none'}, {'paddingTop': '1rem', 'display': 'none'}, {'display': 'block'}];
            }
            const hasTargets = (status.targets_count || 0) > 0;
            const showStyle = hasTargets ? 'block' : 'none';
            const hideStyle = hasTargets ? 'none' : 'block';
            const flexStyle = hasTargets ? 'flex' : 'none';
            return [
                {'display': flexStyle, 'gap': '8px'},
                {'paddingTop': '1rem', 'display': showStyle},
                {'display': hideStyle}
            ];
        }""",
        [
            Output('targets-data-actions-wrapper', 'style'),
            Output('targets-table-container', 'style'),
            Output('targets-empty-state', 'style'),
        ],
        Input('workspace-status', 'data'),
    )

    @app.callback(
        Output("targets-table", "data"),
        Output("targets-table", "selectedRowKeys"),
        Output("targets-table", "pagination"),
        Output("targets-table", "filterOptions"),

        Input('section-context', 'data'),
        Input("targets-action-store", "data"),
        Input("processed-action-store", "data"),  # from explorer
        Input('targets-table', 'pagination'),
        Input('targets-table', 'filter'),
        Input('targets-table', 'sorter'),
        State('targets-table', 'filterOptions'),
        State("processing-type-store", "data"),  # from explorer
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def targets_table(section_context, processing_output, processed_action, pagination, filter_, sorter, filterOptions,
                      processing_type, wdir):
        return _targets_table(section_context, pagination, filter_, sorter, filterOptions, wdir)

    @app.callback(
        Output("targets-notifications-container", "children"),
        Input('section-context', 'data'),
        Input("wdir", "data"),
    )
    def warn_missing_workspace(section_context, wdir):
        if not section_context or section_context.get('page') != 'Targets':
            return dash.no_update
        if wdir:
            return []
        return fac.AntdNotification(
            message="Activate a workspace",
            description="Select or create a workspace before working with Targets.",
            type="warning",
            duration=4,
            placement='bottom',
            showProgress=True,
            stack=True,
        )

    @app.callback(
        Output('delete-table-targets-modal', 'visible'),
        Output('delete-table-targets-modal', 'children'),
        Input("targets-options", "nClicks"),
        State("targets-options", "clickedKey"),
        State('targets-table', 'selectedRows'),
        prevent_initial_call=True
    )
    def show_delete_modal(nClicks, clickedKey, selectedRows):
        ctx = dash.callback_context
        if not ctx.triggered or clickedKey not in ['delete-selected', 'delete-all']:
            logger.debug(f"show_delete_modal: PreventUpdate because triggered={ctx.triggered}, clickedKey={clickedKey}")
            raise PreventUpdate

        if clickedKey == "delete-selected":
            if not bool(selectedRows):
                logger.debug("show_delete_modal: PreventUpdate because no rows selected for deletion")
                raise PreventUpdate

        children = fac.AntdFlex(
            [
                fac.AntdText("This action will delete selected targets and cannot be undone?",
                             strong=True),
                fac.AntdText("Are you sure you want to delete the selected targets?")
            ],
            vertical=True,
        )
        if clickedKey == "delete-all":
            children = fac.AntdFlex(
                [
                    fac.AntdText("This action will delete ALL targets and cannot be undone?",
                                 strong=True, type="danger"),
                    fac.AntdText("Are you sure you want to delete the ALL targets?")
                ],
                vertical=True,
            )
        return True, children
    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('targets-action-store', "data", allow_duplicate=True),
        Output("targets-table-spin", "spinning"),
        Output("workspace-status", "data", allow_duplicate=True),

        Input('delete-table-targets-modal', 'okCounts'),
        State('targets-table', 'selectedRows'),
        State("targets-options", "clickedKey"),
        State("wdir", "data"),
        State("workspace-status", "data"),
        background=True,
        running=[
            (Output("targets-table-spin", "spinning"), True, False),
            (Output("delete-table-targets-modal", "confirmLoading"), True, False),
        ],
        prevent_initial_call=True
    )
    def target_delete(okCounts, selectedRows, clickedKey, wdir, workspace_status):
        return _target_delete(okCounts, selectedRows, clickedKey, wdir, workspace_status)

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("targets-action-store", "data", allow_duplicate=True),

        Input("targets-table", "recentlyChangedRow"),
        State("targets-table", "recentlyChangedColumn"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def save_target_table_on_edit(row_edited, column_edited, wdir):
        return _save_target_table_on_edit(row_edited, column_edited, wdir)

    @app.callback(
        Input('targets-table', 'recentlySwitchDataIndex'),
        Input('targets-table', 'recentlySwitchStatus'),
        Input('targets-table', 'recentlySwitchRow'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def save_switch_changes(recentlySwitchDataIndex, recentlySwitchStatus, recentlySwitchRow, wdir):
        return _save_switch_changes(recentlySwitchDataIndex, recentlySwitchStatus, recentlySwitchRow, wdir)

    @app.callback(
        Output("download-targets-csv", "data"),

        Input("targets-options", "nClicks"),
        Input("download-target-template-btn", "nClicks"),
        Input("download-target-list-btn", "nClicks"),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def download_results(options_clicks, template_clicks, list_clicks, wdir):

        from pathlib import Path
        from ..duckdb_manager import duckdb_connection_mint

        ctx = dash.callback_context
        if not ctx.triggered:
            logger.debug("download_results: PreventUpdate because not triggered")
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

        if trigger == 'download-target-template-btn':
            filename = f"{T.today()}-MINT__{ws_name}-targets_template.csv"
            return dcc.send_string(TARGET_TEMPLATE_CSV, filename)

        if trigger == 'download-target-list-btn':
            if not wdir:
                logger.debug("download_results: PreventUpdate because wdir is not set for target list download")
                raise PreventUpdate
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    logger.debug("download_results: PreventUpdate because database connection is None")
                    raise PreventUpdate
                df = conn.execute("SELECT * FROM targets ORDER BY mz_mean ASC").df()
                # Reorder columns to match the template/export expectation
                cols = TARGET_TEMPLATE_COLUMNS
                df = df[[c for c in cols if c in df.columns]]
                filename = f"{T.today()}-MINT__{ws_name}-targets.csv"
        else:
            logger.debug("download_results: PreventUpdate (unexpected trigger or no action required)")
            raise PreventUpdate
        return dcc.send_data_frame(df.to_csv, filename, index=False)

    @app.callback(
        Output('targets-tour-empty', 'current'),
        Output('targets-tour-empty', 'open'),
        Output('targets-tour-full', 'current'),
        Output('targets-tour-full', 'open'),
        Input('targets-tour-icon', 'nClicks'),
        State('workspace-status', 'data'),
        prevent_initial_call=True,
    )
    def targets_tour(n_clicks, workspace_status):
        logger.debug(f"Tour clicked: {n_clicks}")
        has_targets = workspace_status and (workspace_status.get('targets_count', 0) or 0) > 0
        if has_targets:
            # Open full tour, keep empty tour closed
            return 0, False, 0, True
        else:
            # Open empty tour, keep full tour closed
            return 0, True, 0, False

    @app.callback(
        Output('targets-tour-hint', 'open'),
        Output('targets-tour-hint', 'current'),
        Input('targets-tour-hint-store', 'data'),
    )
    def sync_hint_store(store_data):
        if not store_data:
            logger.debug("sync_hint_store: PreventUpdate because store_data is empty")
            raise PreventUpdate
        return store_data.get('open', True), 0

    @app.callback(
        Output('targets-tour-hint-store', 'data'),
        Input('targets-tour-hint', 'closeCounts'),
        Input('targets-tour-icon', 'nClicks'),
        State('targets-tour-hint-store', 'data'),
        prevent_initial_call=True,
    )
    def hide_hint(close_counts, n_clicks, store_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            logger.debug("hide_hint: PreventUpdate because not triggered")
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'targets-tour-icon':
            return {'open': False}

        if close_counts:
            return {'open': False}

        return store_data or {'open': True}

    @app.callback(
        Output("asari-open-modal-btn", "disabled"),
        Output("asari-open-modal-btn", "title"),
        Input("wdir", "data"),
        Input("targets-action-store", "data"),  # Trigger refresh when targets change
    )
    def toggle_asari_button_for_ms_type(wdir, _action):
        """Disable Auto-Generate button when only MS2 files are loaded (Asari only works with MS1)."""
        if not wdir:
            return False, None  # Button enabled by default when no workspace
        
        try:
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    return False, None
                
                # Check if there are any MS1 files loaded
                result = conn.execute("""
                    SELECT COUNT(*) FROM ms1_data LIMIT 1
                """).fetchone()
                
                has_ms1_data = result and result[0] > 0
                
                if has_ms1_data:
                    return False, None  # Button enabled
                else:
                    return True, "Untargeted Analysis requires MS1 data (not available for MS2-only workflows)"
        except Exception:
            return False, None  # Default to enabled on error


    @app.callback(
        Output("asari-modal", "visible", allow_duplicate=True),
        Output("asari-mode", "value"),
        Output("asari-auto-mode-alert", "children"),
        Input("asari-open-modal-btn", "nClicks"),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def open_asari_modal(n_clicks, wdir):
        if not n_clicks:
             logger.debug("open_asari_modal: PreventUpdate because n_clicks is None")
             raise PreventUpdate
        
        # Check if MS1 data exists (Asari only works with MS1)
        if wdir:
            try:
                with duckdb_connection(wdir) as conn:
                    if conn:
                        result = conn.execute("SELECT COUNT(*) FROM ms1_data LIMIT 1").fetchone()
                        has_ms1_data = result and result[0] > 0
                        if not has_ms1_data:
                            # No MS1 data - don't open modal, show notification instead
                            logger.debug("open_asari_modal: No MS1 data available, Asari cannot run")
                            raise PreventUpdate
            except Exception as e:
                logger.debug(f"open_asari_modal: Error checking MS1 data: {e}")
        
        mode_val = 'pos'
        alert = None
        
        if wdir:
            try:
                with duckdb_connection(wdir) as conn:
                    # Check polarity of active samples
                    query = """
                        SELECT DISTINCT polarity 
                        FROM samples 
                        WHERE use_for_processing = TRUE OR use_for_optimization = TRUE
                    """
                    results = conn.execute(query).fetchall()
                    
                    found_pols = set()
                    for r in results:
                        if r and r[0]:
                            found_pols.add(r[0])
                            
                    if len(found_pols) == 1:
                        pol = list(found_pols)[0]
                        if 'Positive' in pol or 'positive' in pol:
                            mode_val = 'pos'
                            alert = fac.AntdAlert(message="Auto-detected Ionization Mode: Positive", type="success", showIcon=True)
                        elif 'Negative' in pol or 'negative' in pol:
                            mode_val = 'neg'
                            alert = fac.AntdAlert(message="Auto-detected Ionization Mode: Negative", type="success", showIcon=True)
                    elif len(found_pols) > 1:
                         mode_val = 'pos' # Default fallback
                         alert = fac.AntdAlert(message="Multiple polarities detected. Please verify Mode.", type="warning", showIcon=True)
                    else:
                        # No active samples or no polarity info
                        alert = fac.AntdAlert(message="No active samples found to detect polarity.", type="info", showIcon=True)
                        
            except Exception as e:
                logging.warning(f"Failed to auto-detect polarity: {e}")
                
        return True, mode_val, alert

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("asari-modal", "visible", allow_duplicate=True),
        Output("asari-status-container", "children"),
        Output("targets-action-store", "data", allow_duplicate=True),
        Output("workspace-status", "data", allow_duplicate=True),  # New: trigger UI refresh
        
        Input("asari-modal", "okCounts"),
        State("wdir", "data"),
        State("asari-multicores", "value"),
        State("asari-mz-tolerance", "value"),
        State("asari-mode", "value"),
        State("asari-snr", "value"),
        State("asari-min-peak-height", "value"),
        State("asari-min-timepoints", "value"),
        State("asari-gaussian-shape", "value"),
        State("asari-cselectivity", "value"),
        State("asari-detection-rate", "value"),
        
        background=True,
        cancel=[Input("cancel-asari-btn", "nClicks")],
        running=[
            (Output("asari-configuration-container", "style"), {'display': 'none'}, {'display': 'block'}),
            (Output("asari-progress-container", "style"), {
                "display": "flex",
                "justifyContent": "center",
                "justifyContent": "center",
                "alignItems": "center",
                "flexDirection": "column",
                "width": "80%",
                "margin": "auto",
                "height": "100%"
            }, {'display': 'none'}),
            (Output("asari-modal", "closable"), False, True),
            (Output("asari-modal", "maskClosable"), False, False),
            (Output("asari-modal", "okButtonProps"), {'disabled': True}, {'disabled': False}),
            (Output("asari-modal", "cancelButtonProps"), {'disabled': True}, {'disabled': False}),
            (Output("cancel-asari-btn", "disabled"), False, True),
            (Output("asari-modal", "confirmLoading"), True, False),
            (Output("asari-modal", "confirmAutoSpin"), True, False),
        ],
        progress=[
            Output("asari-progress", "percent"),
            Output("asari-progress-stage", "children"),
            Output("asari-progress-detail", "children"),
            Output("asari-terminal-logs", "children"),
        ],
        prevent_initial_call=True
    )
    def run_asari_analysis(set_progress, ok_counts, wdir, multicores, mz_tol, mode, snr, min_height, min_points, gaussian_shape, cselectivity, detection_rate):
        def progress_adapter(data):
            if set_progress:
                set_progress(data)
        return _run_asari_analysis(ok_counts, wdir, multicores, mz_tol, mode, snr, min_height, min_points, gaussian_shape, cselectivity, detection_rate, set_progress=progress_adapter)

    @app.callback(
        Output('asari-multicores-item', 'help'),
        Input('asari-multicores', 'value'),
        prevent_initial_call=True
    )
    def update_asari_cpu_help(cpu):
        return _get_cpu_help_text(cpu)
