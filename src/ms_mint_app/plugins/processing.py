import math
import base64
from os import cpu_count
from pathlib import Path
from io import BytesIO, StringIO

import dash
import feffery_antd_components as fac
import pandas as pd
import psutil
import time
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from .. import tools as T
from ..logging_setup import activate_workspace_logging
import logging
from .scalir import (
    intersect_peaks,
    fit_estimator,
    build_concentration_table,
    training_plot_frame,
    plot_standard_curve,
    slugify_label,
)
from ..duckdb_manager import (
    build_order_by,
    build_where_and_params,
    calculate_optimal_batch_size,
    calculate_optimal_params,
    compute_chromatograms_in_batches,
    compute_fitted_results,
    compute_results_in_batches,
    create_pivot,
    duckdb_connection,
    duckdb_connection_mint,
    get_physical_cores,
)
from ..plugin_interface import PluginInterface
from .target_optimization import (
    _get_cpu_help_text,
    _get_ram_help_text
)

_label = "Processing"

logger = logging.getLogger(__name__)

RESULTS_TABLE_COLUMNS = [
    {
        'title': 'Target',
        'dataIndex': 'peak_label',
        'width': '260px',
        'fixed': 'left'
    },
    {
        'title': 'MS-File Label',
        'dataIndex': 'ms_file_label',
        'width': '260px',
        'fixed': 'left'
    },
    {
        'title': 'peak_area',
        'dataIndex': 'peak_area',
        'width': '120px',
    },
    {
        'title': 'peak_area_top3',
        'dataIndex': 'peak_area_top3',
        'width': '150px',
    },    
    {
        'title': 'peak_max',
        'dataIndex': 'peak_max',
        'width': '120px',
    },
    {
        'title': 'peak_n_datapoints',
        'dataIndex': 'peak_n_datapoints',
        'width': '170px',
    },
    {
        'title': 'peak_rt_of_max',
        'dataIndex': 'peak_rt_of_max',
        'width': '150px',
    },
    # NOTE: EMG Peak Fitting columns are dynamically inserted here when fitting data exists
    # NOTE: SCALiR columns (Concentration, In Range, Unit) are dynamically inserted here
    # via the scalir_column_insert_index logic when concentrations.csv exists
    {
        'title': 'Intensity',
        'dataIndex': 'intensity',
        'width': '260px',
        'renderOptions': {'renderType': 'mini-area'},
    },
]

# Index where SCALiR columns should be inserted (before Intensity)
SCALIR_COLUMN_INSERT_INDEX = -1  # Insert before last column (Intensity)

class ProcessingPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 7
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
                            'Processing', level=4, style={'margin': '0'}
                        ),
                        fac.AntdTooltip(
                            fac.AntdIcon(
                                id='processing-tour-icon',
                                icon='pi-info',
                                style={"cursor": "pointer", 'paddingLeft': '10px'},
                                **{'aria-label': 'Show tutorial'},
                            ),
                            title='Show tutorial'
                        ),
                        fac.AntdSpace(
                            [
                                fac.AntdTooltip(
                                    fac.AntdButton(
                                        'Run MINT',
                                        id='processing-btn',
                                        style={'textTransform': 'uppercase', "margin": "0 10px"},
                                    ),
                                    id='processing-btn-tooltip',
                                    title="Calculate peak areas (integration) for all targets and MS-files.",
                                    placement="bottom"
                                ),
                                fac.AntdTooltip(
                                    fac.AntdButton(
                                        'SCALiR',
                                        id='scalir-modal-btn',
                                        disabled=True,
                                        style={'textTransform': 'uppercase'},
                                    ),
                                    id='scalir-btn-tooltip',
                                    title="Run SCALiR calibration. Requires computed results.",
                                    placement="bottom"
                                ),
                            ],
                            addSplitLine=False,
                            size="small",
                            style={"margin": "0 10px"},
                        ),
                    ],
                    align='center',
                ),
                # Conditionally visible: Download Results and Options (only when results exist)
                html.Div(
                    fac.AntdFlex(
                        [
                            fac.AntdTooltip(
                                fac.AntdButton(
                                    'Download Results',
                                    id='processing-download-btn',
                                    icon=fac.AntdIcon(icon='antd-download'),
                                    iconPosition='end',
                                    style={'textTransform': 'uppercase'},
                                ),
                                title="Download the complete results table as a CSV file.",
                                placement="bottom"
                            ),
                            fac.AntdDropdown(
                                id='processing-options',
                                title='Options',
                                buttonMode=True,
                                arrow=True,
                                menuItems=[
                                    {'title': fac.AntdText('Delete selected', strong=True, type='warning'),
                                     'key': 'processing-delete-selected'},
                                    {'title': fac.AntdText('Clear table', strong=True, type='danger'),
                                     'key': 'processing-delete-all'},
                                ],
                                buttonProps={'style': {'textTransform': 'uppercase'}},
                            ),
                        ],
                        id='processing-options-wrapper',
                        align='center',
                        gap='small',
                    ),
                    id='processing-data-actions-wrapper',
                    style={'display': 'none'},
                ),
            ],
            justify="space-between",
            align="center",
            gap="middle",
        ),
        html.Div(
            [
                fac.AntdFlex(
                    [
                        fac.AntdText('Targets to display', strong=True),
                        fac.AntdSelect(
                            id='processing-peak-select',
                            mode='multiple',
                            allowClear=True,
                            placeholder='Select one or more targets',
                            maxTagCount=4,
                            style={'minWidth': '320px', 'maxWidth': '540px'},
                            optionFilterProp='label',
                        ),
                        fac.AntdText(
                            'Use the dropdown to limit the table to specific compounds.',
                            type='secondary',
                        ),
                    ],
                    align='center',
                    gap='small',
                    wrap=True,
                    style={'paddingBottom': '0.75rem'},
                ),
                fac.AntdSpin(
                    fac.AntdTable(
                        id='results-table',
                        containerId='results-table-container',
                        columns=RESULTS_TABLE_COLUMNS,
                        titlePopoverInfo={
                            'ms_file_label': {
                                'title': 'MS-File Label',
                                'content': 'MS file identifier; matches the MS-Files table.',
                            },
                            'label': {
                                'title': 'Label',
                                'content': 'Friendly label from MS-Files.',
                            },
                            'dash_component': {
                                'title': 'Component',
                                'content': 'Internal dash component name.',
                            },
                            'use_for_optimization': {
                                'title': 'For Optimization',
                                'content': 'If true, file was included in optimization steps.',
                            },
                            'use_for_analysis': {
                                'title': 'For Analysis',
                                'content': 'If true, file was included in analysis outputs.',
                            },
                            'sample_type': {
                                'title': 'Sample Type',
                                'content': 'Sample category (Sample, QC, Blank, Standard).',
                            },
                            'polarity': {
                                'title': 'Polarity',
                                'content': 'Instrument polarity (Positive or Negative).',
                            },
                            'ms_type': {
                                'title': 'MS Type',
                                'content': 'Acquisition type (ms1 or ms2).',
                            },
                            'file_type': {
                                'title': 'File Type',
                                'content': 'File format (e.g., mzML, mzXML).',
                            },
                            'peak_label': {
                                'title': 'Target',
                                'content': 'Target name from the Targets table.',
                            },
                            'peak_area': {
                                'title': 'peak_area',
                                'content': 'Integrated area under the peak.',
                            },
                            'peak_area_top3': {
                                'title': 'peak_area_top3',
                                'content': 'Sum of the three highest intensity points.',
                            },
                            'peak_n_datapoints': {
                                'title': 'peak_n_datapoints',
                                'content': 'Number of datapoints spanning the peak.',
                            },
                            'peak_max': {
                                'title': 'peak_max',
                                'content': 'Maximum intensity within the peak.',
                            },
                            'peak_rt_of_max': {
                                'title': 'peak_rt_of_max',
                                'content': 'Retention time at maximum intensity.',
                            },
                            'scalir_conc': {
                                'title': 'Concentration',
                                'content': 'Predicted concentration from SCALiR calibration curve.',
                            },
                            'scalir_in_range': {
                                'title': 'In Range',
                                'content': 'Whether the value falls within the calibration range (1=yes, 0=no).',
                            },
                            'scalir_unit': {
                                'title': 'Unit',
                                'content': 'Concentration unit (e.g., μM, mM).',
                            },
                            'intensity': {
                                'title': 'Intensity',
                                'content': 'Mini-plot showing the chromatograms',
                            },
                            'peak_area_fitted': {
                                'title': 'peak_area_fitted',
                                'content': 'Peak area from EMG curve fitting (more accurate for tailing peaks)',
                            },
                            'fit_r_squared': {
                                'title': 'fit_r²',
                                'content': 'Goodness of fit (R², 0-1). Higher values indicate better model agreement.',
                            },
                            'fit_success': {
                                'title': 'fit_success',
                                'content': 'Whether EMG fitting converged successfully (✓ = success, ✗ = failed).',
                            },
                        },
                        filterOptions={
                            'peak_label': {'filterMode': 'keyword'},
                            'ms_file_label': {'filterMode': 'keyword'},
                            'sample_type': {'filterMode': 'checkbox'},
                            'ms_type': {'filterMode': 'checkbox',
                                        'filterCustomItems': ['MS1', 'MS2']},
                            'scalir_conc': {'filterMode': 'keyword'},
                            'scalir_in_range': {'filterMode': 'checkbox',
                                                 'filterCustomItems': ['0', '1']},
                            'scalir_unit': {'filterMode': 'checkbox'},
                        },
                        sortOptions={'sortDataIndexes': ['peak_label', 'peak_area', 'peak_area_top3',
                                                         'peak_n_datapoints', 'peak_max',
                                                         'peak_rt_of_max', 'scalir_conc', 'scalir_in_range']},
                        pagination={
                            'position': 'bottomCenter',
                            'pageSize': 15,
                            'showQuickJumper': True,
                            'showSizeChanger': True,
                            'pageSizeOptions': [5, 10, 15, 25, 50, 100],
                        },
                        tableLayout='fixed',
                        maxWidth="calc(100vw - 250px - 4rem)",
                        maxHeight="calc(100vh - 210px)",
                        locale='en-us',
                        showSorterTooltip=False,
                        rowSelectionType='checkbox',
                        size='small',
                        mode='server-side',
                    ),
                    id='results-table-spin',
                    text='Updating table...',
                    size='small',
                    spinning=False,
                )
            ],
            id='results-table-container',
            style={'paddingTop': '1rem', 'display': 'none'},
        ),
        # Empty state placeholder - shown when no results
        fac.AntdFlex(
            [
                fac.AntdEmpty(
                    description=fac.AntdFlex(
                        [
                            fac.AntdText('No Results available', strong=True, style={'fontSize': '16px'}),
                            fac.AntdText('Click "Run MINT" to process the MS files', type='secondary'),
                        ],
                        vertical=True,
                        align='center',
                        gap='small',
                    ),
                    locale='en-us',
                ),
                fac.AntdTooltip(
                    fac.AntdButton(
                        'Run MINT',
                        id='processing-empty-btn',
                        size='large',
                        style={'marginTop': '16px', 'textTransform': 'uppercase'},
                    ),
                    id='processing-empty-btn-tooltip',
                    title="Calculate peak areas (integration) for all targets and MS-files.",
                    placement="bottom"
                ),
            ],
            id='processing-empty-state',
            vertical=True,
            align='center',
            style={'display': 'flex', 'marginTop': '100px'},
        ),
        html.Div(id="processing-notifications-container"),
        fac.AntdModal(
            [
                fac.AntdFlex(
                    [
                        fac.AntdDivider('Options'),
                        fac.AntdForm(
                            [
                                fac.AntdFormItem(
                                    fac.AntdCheckbox(
                                        id='processing-targets-selection',
                                        label='Compute Bookmarked Targets only',
                                    ),
                                    tooltip='Check to compute only the targets that are bookmarked.'
                                ),
                            ]
                        ),
                        fac.AntdForm(
                            [
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='processing-chromatogram-compute-cpu',
                                        defaultValue=calculate_optimal_params()[0],
                                        min=1,
                                        max=cpu_count(),
                                    ),
                                    label='CPU:',
                                    hasFeedback=True,
                                    help=f"Selected {calculate_optimal_params()[0]} / {cpu_count()} cpus",
                                    id='processing-chromatogram-compute-cpu-item'
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='processing-chromatogram-compute-ram',
                                        value=calculate_optimal_params()[1],
                                        min=1,
                                        precision=1,
                                        step=0.1,
                                        suffix='GB'
                                    ),
                                    label='RAM:',
                                    hasFeedback=True,
                                    id='processing-chromatogram-compute-ram-item',
                                    help=f"Recommended {calculate_optimal_params()[1]}GB / "
                                         f"{round(psutil.virtual_memory().available / (1024 ** 3), 1)}GB available"
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='processing-chromatogram-compute-batch-size',
                                        value=4000,  # Default; updated by callback based on CPU/RAM
                                        min=500,
                                        max=8000,
                                        step=500,
                                    ),
                                    label='Batch Size:',
                                    tooltip='Pairs per batch. Larger = faster (up to 8000). '
                                            'Based on experimental benchmarks.',
                                ),
                            ],
                            layout='inline'
                        ),
                        fac.AntdDivider('Peak Fitting'),
                        fac.AntdSpace(
                            [
                                fac.AntdCheckbox(
                                    id='processing-enable-fitting',
                                    label='Enable EMG Peak Fitting',
                                ),
                                fac.AntdTooltip(
                                    fac.AntdIcon(icon='antd-question-circle', style={'color': '#8c8c8c', 'cursor': 'help'}),
                                    title='Fit Exponentially Modified Gaussian (EMG) to each peak for more accurate area quantification. '\
                                          'Adds peak_area_fitted, fit_r², and fit_success columns. '\
                                          'Best for peaks with tailing (asymmetric shape). '\
                                          'Runs after standard processing using multiprocessing.',
                                ),
                            ],
                            style={'marginBottom': '0.5rem'},
                        ),
                        fac.AntdDivider('Recompute'),
                        fac.AntdForm(
                            [
                                fac.AntdFormItem(
                                    fac.AntdCheckbox(
                                        id='processing-recompute',
                                        label='Recompute results'
                                    ),
                                    style={'marginBottom': '0.5rem'},
                                ),
                            ]
                        ),
                        fac.AntdAlert(
                            message='There are already computed results',
                            type='warning',
                            showIcon=True,
                            id='processing-warning',
                            style={'display': 'none', 'marginBottom': '2rem'},
                        )
                    ],
                    id='processing-options-container',
                    vertical=True
                ),

                html.Div(
                    [
                        html.H4("Running MINT..."),
                        fac.AntdText(
                            id='processing-progress-stage',
                            style={'marginBottom': '0.5rem'},
                        ),
                        fac.AntdProgress(
                            id='processing-progress',
                            percent=0,
                        ),
                        fac.AntdText(
                            id='processing-progress-detail',
                            type='secondary',
                            style={
                                'marginTop': '0.5rem',
                                'marginBottom': '0.75rem',
                            },
                        ),
                        fac.AntdButton(
                            'Cancel',
                            id='cancel-processing',
                            style={
                                'alignText': 'center',
                                'marginTop': '0.25rem',
                            },
                        ),
                    ],
                    id='processing-progress-container',
                    style={'display': 'none'},
                ),
            ],
            id='processing-modal',
            title='Run MINT',
            width=900,
            renderFooter=True,
            locale='en-us',
            confirmAutoSpin=True,
            loadingOkText='Running...',
            okClickClose=False,
            closable=False,
            maskClosable=False,
            destroyOnClose=True,
            okText="Run",
            centered=True,
            styles={'body': {'height': "50vh"}},
        ),
        fac.AntdModal(
            "Are you sure you want to delete the selected results?",
            title="Delete confirmation",
            id="processing-delete-confirmation-modal",
            okButtonProps={"danger": True},
            renderFooter=True,
            locale='en-us',
        ),
        fac.AntdModal(
            [
                fac.AntdFlex(
                    [
                        fac.AntdDivider(
                            'All results',
                            size='small'
                        ),

                        fac.AntdFlex(
                            [
                                fac.AntdForm(
                                    [
                                        fac.AntdFormItem(
                                            [
                                                fac.AntdSelect(
                                                    options=['peak_area', 'peak_area_fitted', 'peak_area_top3', 'peak_mean',
                                                             'peak_median', 'peak_n_datapoints', 'peak_min', 'peak_max',
                                                             'peak_rt_of_max', 'peak_sigma', 'peak_tau', 'peak_asymmetry',
                                                             'peak_rt_fitted', 'fit_r_squared', 'fit_success', 'total_intensity',
                                                             'rt_aligned', 'rt_shift', 'peak_mz_of_max', 'scan_time', 'intensity'],
                                                    mode="multiple",
                                                    value=['peak_area', 'peak_area_top3', 'peak_mean',
                                                           'peak_median', 'peak_n_datapoints', 'peak_min', 'peak_max',
                                                           'peak_rt_of_max', 'total_intensity'],
                                                    style={"width": "100%"},
                                                    locale="en-us",
                                                    id='download-options-all-results'
                                                ),
                                            ],
                                            label='Download all results',
                                            tooltip='Download all results in tabular format',
                                            hasFeedback=True,
                                            id='download-options-all-results-item'
                                        ),
                                    ],
                                    layout='vertical',
                                    style={'flex': 1}
                                ),
                                fac.AntdButton(
                                    'Download',
                                    id='download-all-results-btn',
                                    icon=fac.AntdIcon(icon='antd-download'),
                                    autoSpin=True,
                                    style={'minWidth': 110, 'textTransform': 'uppercase'},
                                )
                            ],
                            justify='center',
                            align='center',
                            gap=10
                        ),
                        fac.AntdDivider(
                            'Dense Matrix',
                            size='small'
                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdForm(
                                    [
                                        fac.AntdFormItem(
                                            [
                                                fac.AntdSelect(
                                                    options=['ms_file_label', 'peak_label', 'ms_type'],
                                                    value=['ms_file_label'],
                                                    style={"width": "100%"},
                                                    locale="en-us",
                                                    id='download-densematrix-rows',
                                                ),
                                            ],
                                            label='Rows:',
                                            tooltip='Rows information',
                                            hasFeedback=True,
                                            id='download-densematrix-rows-item',
                                        ),
                                        fac.AntdFormItem(
                                            [
                                                fac.AntdSelect(
                                                    options=['peak_label', 'ms_file_label', 'ms_type'],
                                                    value=['peak_label'],
                                                    style={"width": "100%"},
                                                    locale="en-us",
                                                    id='download-densematrix-cols'
                                                ),
                                            ],
                                            label='Columns:',
                                            tooltip='Columns information',
                                            hasFeedback=True,
                                            id='download-densematrix-cols-item',
                                        ),
                                        fac.AntdFormItem(
                                            [
                                                fac.AntdSelect(
                                                    options=['peak_area', 'peak_area_fitted', 'peak_area_top3', 'peak_mean', 'peak_median',
                                                             'peak_n_datapoints', 'peak_min', 'peak_max',
                                                             'peak_rt_of_max', 'peak_sigma', 'peak_tau', 'peak_asymmetry',
                                                             'peak_rt_fitted', 'fit_r_squared', 'fit_success', 'total_intensity'],
                                                    value=['peak_area'],
                                                    style={"width": "100%"},
                                                    locale="en-us",
                                                    id='download-densematrix-value'
                                                ),
                                            ],
                                            label='Value:',
                                            tooltip='Value information',
                                            hasFeedback=True,
                                            id='download-densematrix-value-item'
                                        ),
                                    ],
                                    layout='vertical',
                                    style={'flexGrow': 1}
                                ),
                                fac.AntdButton(
                                    'Download',
                                    id='download-densematrix-results-btn',
                                    icon=fac.AntdIcon(icon='antd-download'),
                                    autoSpin=True,
                                    style={'minWidth': 110, 'textTransform': 'uppercase'},
                                )
                            ],
                            justify='center',
                            align='center',
                            gap=10
                        ),

                    ],
                    vertical=True
                )
            ],
            title="Download results",
            id="download-results-modal",
            width=700,
            locale='en-us',
        ),
        # SCALiR Modal - Standard Curve Calibration
        fac.AntdModal(
            [
                fac.AntdSpace(
                    [
                        fac.AntdText("Params:", strong=True, style={'marginRight': 8}),
                        fac.AntdSelect(
                            id='scalir-intensity',
                            options=[
                                {'label': 'Peak Area', 'value': 'peak_area'},
                                {'label': 'Peak Max', 'value': 'peak_max'},
                            ],
                            value='peak_area',
                            style={'width': 160},
                        ),
                        fac.AntdSelect(
                            id='scalir-slope-mode',
                            options=[
                                {'label': 'Fixed slope', 'value': 'fixed'},
                                {'label': 'Constrained slope', 'value': 'interval'},
                                {'label': 'Free slope', 'value': 'wide'},
                            ],
                            value='fixed',
                            style={'width': 180},
                        ),
                        fac.AntdInputNumber(
                            id='scalir-slope-low',
                            value=0.85,
                            min=0.1,
                            max=5,
                            step=0.01,
                            placeholder='Slope low',
                            style={'width': 110},
                        ),
                        fac.AntdInputNumber(
                            id='scalir-slope-high',
                            value=1.15,
                            min=0.1,
                            max=5,
                            step=0.01,
                            placeholder='Slope high',
                            style={'width': 110},
                        ),
                        fac.AntdSwitch(
                            id='scalir-generate-plots',
                            checked=False,
                            checkedChildren='Save plots',
                            unCheckedChildren='Skip plots',
                        ),
                        fac.AntdTooltip(
                            fac.AntdIcon(
                                icon='antd-question-circle',
                                style={'color': '#555', 'cursor': 'help'}
                            ),
                            title='Skipping plot generation speeds up the process significantly for large datasets.'
                        ),
                    ],
                    wrap=True,
                    size='small',
                    style={'marginBottom': 12},
                ),
                html.Div(
                    [
                        fac.AntdSpace(
                            [
                                fac.AntdText(
                                    'Standards file (CSV):',
                                    strong=True,
                                    style={'margin': 0, 'lineHeight': '32px'},
                                ),
                                html.Div(
                                    [
                                        dcc.Upload(
                                            id='scalir-standards-upload',
                                            children=fac.AntdButton(
                                                'Select standards file',
                                                icon=fac.AntdIcon(icon='antd-upload'),
                                                type='default',
                                                block=True,
                                                style={
                                                    'width': '180px',
                                                    'height': '36px',
                                                    'display': 'flex',
                                                    'alignItems': 'center',
                                                    'justifyContent': 'center',
                                                    'gap': 6,
                                                },
                                            ),
                                            style={'padding': 0},
                                            multiple=False,
                                        ),
                                        fac.AntdText(
                                            id='scalir-standards-note',
                                            style={'color': '#555', 'fontSize': 12, 'lineHeight': '32px', 'marginLeft': 8},
                                        ),
                                    ],
                                    style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'},
                                ),
                            ],
                            align='center',
                            size='small',
                            style={'width': '100%', 'marginTop': 4},
                        ),
                        # Hidden placeholders for callbacks (content removed from UI but needed for callback outputs)
                        html.Div(id='scalir-conc-path', style={'display': 'none'}),
                        html.Div(id='scalir-params-path', style={'display': 'none'}),
                    ],
                    style={'marginBottom': 12, 'padding': '0 0px'},
                ),
                fac.AntdSpace(
                    [
                        fac.AntdButton(
                            'Run SCALiR',
                            id='scalir-run-btn',
                            type='primary',
                            style={'minWidth': 110},
                            autoSpin=True,
                        ),
                        fac.AntdButton(
                            'Clear',
                            id='scalir-reset-btn',
                            type='default',
                            danger=False,
                        ),
                        fac.AntdText(
                            id='scalir-status-text',
                            style={'marginLeft': 8},
                        ),
                    ],
                    size='small',
                    style={'marginBottom': 16},
                ),
                fac.AntdDivider('Results', style={'margin': '4px 0'}),
                fac.AntdFlex(
                    [
                        fac.AntdSpace(
                            [
                                fac.AntdText(
                                    'Dropdown to view specific compounds:',
                                    strong=True,
                                    style={'maxWidth': '280px', 'display': 'block'}
                                ),
                                fac.AntdSelect(
                                    id='scalir-metabolite-select',
                                    options=[],
                                    value=None,
                                    mode=None,  # Single-select mode
                                    allowClear=True,
                                    placeholder='Select a metabolite',
                                    style={'width': 280},
                                ),
                            ],
                            direction='vertical',
                            size=4,
                            style={'flexShrink': 0},
                        ),
                        html.Div(
                            id='scalir-plot-graphs',
                            style={
                                'display': 'none',
                                'flexGrow': 1,
                                'minWidth': '350px',
                            },
                        ),
                    ],
                    gap='middle',
                    align='flex-start',
                    style={'marginBottom': 12},
                ),
                fac.AntdText(id='scalir-plot-path', style={'fontSize': 12, 'color': '#444'}),
                dcc.Store(id='scalir-results-store'),
            ],
            id='scalir-modal',
            title='SCALiR - Standard Curve Calibration',
            width=950,
            renderFooter=False,
            locale='en-us',
            centered=True,
            styles={'body': {'minHeight': '520px'}},
        ),
        # Tour for empty state (no results yet) - simplified
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'This tutorial shows how to get started with processing.',
                },
                {
                    'title': 'Run MINT',
                    'description': 'Click "Run MINT" to compute chromatograms and results for your MS files and targets.',
                    'targetSelector': '#processing-btn'
                },
                {
                    'title': 'Run SCALiR',
                    'description': 'Click "SCALiR" to calibrate concentrations using standard curves.',
                    'targetSelector': '#scalir-modal-btn'
                },
            ],
            id='processing-tour-empty',
            open=False,
            current=0,
        ),
        # Tour for populated state (results exist) - full tour
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'This tutorial shows how to review and export your results.',
                },
                {
                    'title': 'Run more processing',
                    'description': 'Click "Run MINT" to recompute or compute additional targets.',
                    'targetSelector': '#processing-btn'
                },
                {
                    'title': 'Run SCALiR',
                    'description': 'Click "SCALiR" to calibrate concentrations using standard curves.',
                    'targetSelector': '#scalir-modal-btn'
                },
                {
                    'title': 'Pick targets',
                    'description': 'Choose one or more targets to show in the results table.',
                    'targetSelector': '#processing-peak-select'
                },
                {
                    'title': 'Review results',
                    'description': 'Filter and sort results in the table.',
                    'targetSelector': '#results-table-container'
                },
                {
                    'title': 'Export or clean up',
                    'description': 'Download results or delete rows from the options.',
                    'targetSelector': '#processing-options-wrapper'
                },
            ],
            id='processing-tour-full',
            open=False,
            current=0,
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Need help?',
                    'description': 'Click the info icon to open a quick tour of Processing.',
                    'targetSelector': '#processing-tour-icon',
                },
            ],
            mask=False,
            placement='rightTop',
            open=False,
            current=0,
            id='processing-tour-hint',
            className='targets-tour-hint',
            style={
                'background': '#ffffff',
                'border': '0.5px solid #1677ff',
                'boxShadow': '0 6px 16px rgba(0,0,0,0.15), 0 0 0 1px rgba(22,119,255,0.2)',
                'opacity': 1,
            },
        ),
        dcc.Store('results-action-store'),
        dcc.Store(id='processing-tour-hint-store', data={'open': False}, storage_type='local'),
        dcc.Download('download-csv'),
    ]
)


def layout():
    return _layout


# SCALiR constants and helper functions
SCALIR_ALLOWED_METRICS = {
    'peak_area',
    'peak_area_top3',
    'peak_max',
    'peak_mean',
    'peak_median',
}


def _parse_uploaded_standards(contents, filename):
    """Parse an uploaded standards file (CSV or Excel)."""
    if not contents or not filename:
        raise ValueError("No standards file provided.")
    if len(contents) > 15_000_000:
        raise ValueError("Standards file is too large (limit: 15MB).")
    content_type, content_string = contents.split(",", 1)
    if content_type and "csv" not in content_type and "excel" not in content_type:
        raise ValueError("Unsupported file type. Use CSV or Excel.")
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(BytesIO(decoded))
    try:
        return pd.read_csv(StringIO(decoded.decode("utf-8")))
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(decoded))


def _build_delete_modal_content(clickedKey: str, selectedRows: list) -> tuple[bool, fac.AntdFlex]:
    """
    Build the delete confirmation modal content based on the clicked action.
    
    Args:
        clickedKey: The key of the clicked option ('processing-delete-selected' or 'processing-delete-all')
        selectedRows: List of selected rows from the table
        
    Returns:
        Tuple of (modal_visible, modal_children)
        
    Raises:
        PreventUpdate: If validation fails (no rows selected for delete-selected)
    """
    if clickedKey == "processing-delete-selected":
        if not selectedRows:
            logger.debug("_build_delete_modal_content: PreventUpdate because no rows selected for delete-selected")
            raise PreventUpdate
        
        children = fac.AntdFlex(
            [
                fac.AntdText("This action will delete selected results and cannot be undone?",
                             strong=True),
                fac.AntdText("Are you sure you want to delete the selected results?")
            ],
            vertical=True,
        )
    else:  # processing-delete-all
        children = fac.AntdFlex(
            [
                fac.AntdText("This action will delete ALL results and cannot be undone?",
                             strong=True, type="danger"),
                fac.AntdText("Are you sure you want to delete the ALL results?")
            ],
            vertical=True,
        )
    
    return True, children


def _load_peaks_from_results(wdir: str, current_value: list) -> tuple[list, list]:
    """
    Load available peak labels from results table and manage selection.
    
    Args:
        wdir: Workspace directory path
        current_value: Currently selected peak labels
        
    Returns:
        Tuple of (options, selected_values) for the peak selector dropdown
        
    Raises:
        PreventUpdate: If workspace is invalid or database unavailable
    """
    if not wdir:
        logger.debug("_load_peaks_from_results: PreventUpdate because wdir is not set")
        raise PreventUpdate
        
    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.debug("_load_peaks_from_results: PreventUpdate because database connection is None")
            raise PreventUpdate
        peaks = conn.execute(
            "SELECT DISTINCT peak_label FROM results ORDER BY peak_label"
        ).fetchall()

    options = [
        {'label': peak[0], 'value': peak[0]}
        for peak in peaks
        if peak and peak[0] is not None
    ]

    if not options:
        return [], []
    
    if not current_value:
        return options, [options[0]['value']]
    
    valid_values = {opt['value'] for opt in options}
    filtered_value = [v for v in current_value if v in valid_values]
    
    # If current selection is no longer valid (deleted), auto-select first available
    if not filtered_value and options:
        return options, [options[0]['value']]
    
    return options, filtered_value


def _download_all_results(wdir: str, ws_name: str, selected_columns: list) -> tuple:
    """
    Download all results with selected columns as CSV.
    
    Args:
        wdir: Workspace directory path
        ws_name: Workspace name for filename
        selected_columns: List of column names to include in download
        
    Returns:
        Tuple of (download_data, notification) where download_data is for dcc.Download
        and notification is AntdNotification or None
    """
    # All columns that can be downloaded (matching the dropdown options in the modal)
    allowed_cols = {
        # Core result columns
        'peak_area', 'peak_area_top3', 'peak_mean', 'peak_median',
        'peak_n_datapoints', 'peak_min', 'peak_max', 'peak_rt_of_max', 'total_intensity',
        # EMG Peak Fitting columns
        'peak_area_fitted', 'peak_sigma', 'peak_tau', 'peak_asymmetry',
        'peak_rt_fitted', 'fit_r_squared', 'fit_success',
        # Raw data arrays (optional - for advanced users)
        'scan_time', 'intensity',
        # RT alignment columns
        'rt_aligned', 'rt_shift',
        # m/z of max
        'peak_mz_of_max',
    }
    
    if not selected_columns or not isinstance(selected_columns, list):
        return dash.no_update, fac.AntdNotification(
            message="Download Results",
            description="Select at least one column.",
            type="warning",
            duration=4,
            placement="bottom",
            showProgress=True,
        )
    
    safe_cols = [c for c in selected_columns if c in allowed_cols]
    if not safe_cols:
        return dash.no_update, fac.AntdNotification(
            message="Download Results",
            description="No valid columns selected.",
            type="warning",
            duration=4,
            placement="bottom",
            showProgress=True,
        )
    
    with duckdb_connection(wdir) as conn:
        if conn is None:
            return dash.no_update, fac.AntdNotification(
                message="Download Results",
                description="Database unavailable.",
                type="error",
                duration=4,
                placement="bottom",
                showProgress=True,
            )
        
        # Check for existing backup file (generated when results table loads)
        backup_path = Path(wdir) / "results" / "results_backup.csv"
        
        filename = f"{T.today()}-MINT__{ws_name}-all_results.csv"
        tmp_path = None
        import os
        
        if backup_path.exists():
            # 1. Use Polars to clean/filter the data first (respecting user column selection)
            logger.info(f"Download request: {filename} (filtering columns with Polars)")
            import polars as pl
            import tempfile
            
            # Read schema to find available columns
            lf = pl.scan_csv(backup_path, infer_schema_length=100)
            available_cols = lf.collect_schema().names()
            
            # Filter to requested columns (always include keys)
            cols_to_select = ['peak_label', 'ms_file_label'] + [
                c for c in safe_cols 
                if c in available_cols and c not in ('peak_label', 'ms_file_label')
            ]
            
            # Write filtered data to temp file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
                tmp_path = tmp.name
            
            lf.select(cols_to_select).collect().write_csv(tmp_path)
            
        else:
            # Fallback: generate from database if backup doesn't exist
            logger.info(f"Backup missing for {filename}, generating from DB...")
            tmp_path = _generate_csv_from_db(wdir, ws_name, safe_cols)
            
            if tmp_path is None:
                return dash.no_update, fac.AntdNotification(
                    message="Download Results",
                    description="Database unavailable.",
                    type="error",
                    duration=4,
                    placement="bottom",
                    showProgress=True,
                )
        
        # 2. Check size of the filtered file

        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        
        # 3. If file is large (>50MB), serve via Flask Direct Download + Modal
        # This prevents browser freezing/crashing with base64 data
        if file_size_mb > 50:
            logger.info(f"File is large ({file_size_mb:.1f} MB). Using Direct Download via Modal.")
            import base64
            
            # Encode path for Flask route
            encoded_path = base64.urlsafe_b64encode(tmp_path.encode()).decode()
            download_url = f"/download-results-direct/{encoded_path}?filename={filename}"
            
            return dash.no_update, fac.AntdModal(
                title="Download Large Results File",
                visible=True,
                width=500,
                children=[
                    html.P(f"The filtered results file is large ({file_size_mb:.1f} MB)."),
                    html.P("Click the button below to download it directly."),
                    html.Div(
                        fac.AntdButton(
                            "DOWNLOAD RESULTS",
                            href=download_url,
                            target="_blank",
                            icon=fac.AntdIcon(icon="antd-download"),
                            style={
                                "marginTop": "10px",
                                "textTransform": "uppercase"
                            }
                        ),
                        style={"textAlign": "center", "marginTop": "20px"}
                    )
                ],
                okButtonProps={'style': {'display': 'none'}},
                cancelButtonProps={'style': {'display': 'none'}}
            )
            
        # 4. If file is small, standard dcc.Download is fine
        return dcc.send_file(tmp_path, filename=filename), dash.no_update


def _generate_csv_from_db(wdir: str, ws_name: str, safe_cols: list) -> str:
    """Fallback: generate CSV from database when backup doesn't exist. Returns path to temp file."""
    with duckdb_connection(wdir) as conn:
        if conn is None:
            return None
        
        # Build column list, converting arrays to comma-separated strings
        col_list = []
        for c in safe_cols:
            if c in ('scan_time', 'intensity'):
                col_list.append(f"array_to_string(r.{c}, ',') AS {c}")
            elif c not in ('peak_label', 'ms_file_label', 'ms_type'): # Avoid dupes
                col_list.append(f"r.{c}")
        cols = ', '.join(col_list)
        
        filename = f"{T.today()}-MINT__{ws_name}-all_results.csv"
        logger.info(f"Generating temporary CSV: {filename} (from database)")
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
            tmp_path = tmp.name
        
        conn.execute(f"""
            COPY (
                SELECT 
                    r.peak_label, 
                    r.ms_file_label, 
                    s.ms_type,
                    {cols} 
                FROM results r 
                JOIN samples s ON s.ms_file_label = r.ms_file_label 
                ORDER BY s.ms_type, r.peak_label, r.ms_file_label
            ) TO ? (HEADER, DELIMITER ',')
        """, (tmp_path,))
        
    return tmp_path


def _download_dense_matrix(wdir: str, ws_name: str, rows: list, cols: list, value: list) -> tuple:
    """
    Download dense matrix (pivot table) as CSV.
    
    Args:
        wdir: Workspace directory path
        ws_name: Workspace name for filename
        rows: List with row column name
        cols: List with column column name
        value: List with value column name
        
    Returns:
        Tuple of (download_data, notification) where download_data is for dcc.Download
        and notification is AntdNotification or None
    """
    allowed_rows_cols = {'ms_file_label', 'peak_label', 'ms_type'}
    allowed_values = {
        'peak_area', 'peak_area_top3', 'peak_mean', 'peak_median',
        'peak_n_datapoints', 'peak_min', 'peak_max', 'peak_rt_of_max', 'total_intensity',
    }
    
    if not rows or not cols or not value:
        return dash.no_update, fac.AntdNotification(
            message="Download Results",
            description="Missing row, column, or value selection.",
            type="warning",
            duration=4,
            placement="bottom",
            showProgress=True,
        )
    
    row_col = rows[0]
    col_col = cols[0]
    val_col = value[0]
    
    if row_col not in allowed_rows_cols or col_col not in allowed_rows_cols:
        return dash.no_update, fac.AntdNotification(
            message="Download Results",
            description="Invalid row/column selection.",
            type="warning",
            duration=4,
            placement="bottom",
            showProgress=True,
        )
    
    if val_col not in allowed_values:
        return dash.no_update, fac.AntdNotification(
            message="Download Results",
            description="Invalid value selection.",
            type="warning",
            duration=4,
            placement="bottom",
            showProgress=True,
        )
    
    with duckdb_connection(wdir) as conn:
        if conn is None:
            return dash.no_update, fac.AntdNotification(
                message="Download Results",
                description="Database unavailable.",
                type="error",
                duration=4,
                placement="bottom",
                showProgress=True,
            )
        
        df = create_pivot(conn, rows[0], cols[0], value[0], table='results')
        filename = f"{T.today()}-MINT__{ws_name}-{value[0]}_results.csv"
        logger.info(f"Download request (dense matrix): {filename}")
    
    return dcc.send_data_frame(df.to_csv, filename, index=False), dash.no_update


def _delete_selected_results(wdir: str, selected_rows: list) -> tuple:
    """
    Delete selected results from the database.
    
    Args:
        wdir: Workspace directory path
        selected_rows: List of selected row dictionaries with peak_label and ms_file_label
        
    Returns:
        Tuple of (notification, results_action_store, total_removed)
        where total_removed is [unique_peaks, unique_files]
    """
    if not selected_rows:
        return (
            None,  # No notification on validation failure
            {'action': 'delete', 'status': 'failed'},
            [0, 0]
        )
    
    # Extract unique pairs
    remove_pairs = list({
        (row["peak_label"], row["ms_file_label"]) for row in selected_rows
        if row.get("peak_label") and row.get("ms_file_label")
    })
    unique_peaks = len({p for p, _ in remove_pairs})
    unique_files = len({m for _, m in remove_pairs})
    
    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.debug("_delete_selected_results: PreventUpdate because database connection is None")
            raise PreventUpdate
        
        try:
            conn.execute("BEGIN")
            total_removed = [unique_peaks, unique_files]
            if remove_pairs:
                placeholders = ", ".join(["(?, ?)"] * len(remove_pairs))
                params = [v for pair in remove_pairs for v in pair]
                conn.execute(
                    f"DELETE FROM results WHERE (peak_label, ms_file_label) IN ({placeholders})",
                    params
                )
                results_action_store = {'action': 'delete', 'status': 'success'}
                # Invalidate backup file so it regenerates fresh on next download
                (Path(wdir) / "results" / "results_backup.csv").unlink(missing_ok=True)
            else:
                results_action_store = {'action': 'delete', 'status': 'failed'}
                total_removed = [0, 0]
            conn.execute("COMMIT")
            return (None, results_action_store, total_removed)
        except Exception as e:
            conn.execute("ROLLBACK")
            return (
                fac.AntdNotification(
                    message="Failed to delete results",
                    description=f"Error: {e}",
                    type="error",
                    duration=4,
                    placement='bottom',
                    showProgress=True,
                    stack=True
                ),
                {'action': 'delete', 'status': 'failed'},
                [0, 0]
            )


def _delete_all_results(wdir: str) -> tuple:
    """
    Delete all results from the database.
    
    Args:
        wdir: Workspace directory path
        
    Returns:
        Tuple of (notification, results_action_store, total_removed)
        where total_removed is [unique_peaks, unique_files]
    """
    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.debug("_delete_all_results: PreventUpdate because database connection is None")
            raise PreventUpdate
        
        try:
            conn.execute("BEGIN")
            total_removed_q = conn.execute("""
                SELECT COUNT(DISTINCT peak_label)    as unique_peaks,
                       COUNT(DISTINCT ms_file_label) as unique_files
                FROM results
            """).fetchone()
            
            results_action_store = {'action': 'delete', 'status': 'failed'}
            total_removed = [0, 0]
            
            if total_removed_q:
                total_removed = list(total_removed_q)
                conn.execute("DELETE FROM results")
                results_action_store = {'action': 'delete', 'status': 'success'}
                # Invalidate backup file so it regenerates fresh on next download
                (Path(wdir) / "results" / "results_backup.csv").unlink(missing_ok=True)
                logger.info(f"Deleted {total_removed[0]} targets and {total_removed[1]} samples from results.")
            
            conn.execute("COMMIT")
            return (None, results_action_store, total_removed)
        except Exception as e:
            logger.error("Failed to delete results.", exc_info=True)
            conn.execute("ROLLBACK")
            return (
                fac.AntdNotification(
                    message="Failed to delete results",
                    description=f"Error: {e}",
                    type="error",
                    duration=4,
                    placement='bottom',
                    showProgress=True,
                    stack=True
                ),
                {'action': 'delete', 'status': 'failed'},
                [0, 0]
            )


def callbacks(app, fsc, cache):
    # Flask route for direct file downloads (bypasses Dash's slow base64 encoding)
    from flask import send_file as flask_send_file, request
    
    @app.server.route('/download-results-direct/<path:filepath>')
    def download_results_direct(filepath):
        """Serve large results files directly via Flask - much faster than dcc.send_file"""
        # Decode the path (it's base64 encoded for safety)
        import base64
        try:
            decoded_path = base64.urlsafe_b64decode(filepath.encode()).decode()
            file_path = Path(decoded_path)
            
            # Get desired filename from query param, else use actual filename
            download_name = request.args.get('filename', file_path.name)
            
            if file_path.exists() and file_path.suffix == '.csv':
                return flask_send_file(
                    str(file_path),
                    as_attachment=True,
                    download_name=download_name
                )
        except Exception as e:
            logger.error(f"Direct download failed: {e}")
        return "File not found", 404
    
    # Clientside callback to toggle processing UI visibility based on workspace-status
    # This runs in the browser for instant UI updates without server roundtrips
    app.clientside_callback(
        """(status) => {
            if (!status) {
                return [{'display': 'none'}, {'paddingTop': '1rem', 'display': 'none'}, {'display': 'flex'}];
            }
            const hasResults = (status.results_count || 0) > 0;
            const showStyle = hasResults ? 'block' : 'none';
            const hideStyle = hasResults ? 'none' : 'flex';
            return [
                {'display': showStyle},
                {'paddingTop': '1rem', 'display': showStyle},
                {'display': hideStyle, 'marginTop': '100px'}
            ];
        }""",
        [
            Output('processing-data-actions-wrapper', 'style'),
            Output('results-table-container', 'style'),
            Output('processing-empty-state', 'style'),
        ],
        Input('workspace-status', 'data'),
    )

    # Disable Run MINT buttons when no targets or ms-files exist
    app.clientside_callback(
        """(status) => {
            if (!status) return [true, true, "Load MS-Files and Targets first", "Load MS-Files and Targets first"];
            const hasFiles = (status.ms_files_count || 0) > 0;
            const hasTargets = (status.targets_count || 0) > 0;
            const disabled = !(hasFiles && hasTargets);
            const tooltip = disabled ? "Load MS-Files and Targets first" : "Calculate peak areas (integration) for all targets and MS-files.";
            return [disabled, disabled, tooltip, tooltip];
        }""",
        Output('processing-btn', 'disabled'),
        Output('processing-empty-btn', 'disabled'),
        Output('processing-btn-tooltip', 'title'),
        Output('processing-empty-btn-tooltip', 'title'),
        Input('workspace-status', 'data'),
        prevent_initial_call=False,
    )

    @app.callback(
        Output("processing-notifications-container", "children"),
        Input('section-context', 'data'),
        Input("wdir", "data"),
        prevent_initial_call=True,
    )
    def warn_missing_workspace(section_context, wdir):
        if not section_context or section_context.get('page') != 'Processing':
            return dash.no_update
        if wdir:
            return []
        return fac.AntdNotification(
            message="Activate a workspace",
            description="Please select or create a workspace first.",
            type="warning",
            duration=4,
            placement='bottom',
            showProgress=True,
            stack=True,
        )
    @app.callback(
        Output("results-table", "data"),
        Output("results-table", "selectedRowKeys"),
        Output("results-table", "pagination"),
        Output("results-table", "columns"),

        Input('section-context', 'data'),
        Input("results-action-store", "data"),
        Input('scalir-results-store', 'data'),
        Input('processing-peak-select', 'value'),
        Input('results-table', 'pagination'),
        Input('results-table', 'filter'),
        Input('results-table', 'sorter'),
        State('results-table', 'filterOptions'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def results_table(section_context, results_actions, scalir_data, selected_peaks,
                      pagination, filter_, sorter, filterOptions, wdir):
        if section_context and section_context['page'] != 'Processing':
            logger.debug(f"results_table: PreventUpdate because current page is {section_context.get('page')}")
            raise PreventUpdate

        if not wdir:
            logger.debug("results_table: PreventUpdate because wdir is not set")
            raise PreventUpdate

        # Connect to database using context manager
        with duckdb_connection(wdir) as conn:
            if conn is None:
                logger.debug("results_table: PreventUpdate because database connection is None")
                raise PreventUpdate

            # Create alias for compatibility
            conn.execute("PRAGMA enable_profiling='json'")
            conn.execute("PRAGMA profile_output='duckdb_profile.json'")

        pagination = pagination or {
            'position': 'bottomCenter',
            'pageSize': 15,
            'showQuickJumper': True,
            'showSizeChanger': True,
            'pageSizeOptions': [5, 10, 15, 25, 50, 100],
        }
        base_page_size_options = [5, 10, 15, 25, 50, 100]
        page_size_options = base_page_size_options.copy()
        try:
            page_size = int(pagination.get('pageSize') or 15)
        except (TypeError, ValueError):
            page_size = 25
        if page_size <= 0:
            page_size = 25
        current = pagination.get('current') or 1

        filterOptions = filterOptions or {}
        selected_peaks = selected_peaks or []
        if not isinstance(selected_peaks, list):
            selected_peaks = [selected_peaks]

        if not selected_peaks:
            return (
                [],
                [],
                {
                    **pagination,
                    'total': 0,
                    'current': 1,
                    'pageSize': page_size,
                    'pageSizeOptions': page_size_options,
                },
                RESULTS_TABLE_COLUMNS,
            )

        with duckdb_connection(wdir) as conn:
            if conn is None:
                logger.debug("results_table: PreventUpdate because database connection is None")
                raise PreventUpdate
            results_schema = conn.execute("DESCRIBE results").fetchall()
            samples_schema = conn.execute("DESCRIBE samples").fetchall()
            column_types = {row[0]: row[1] for row in results_schema}
            column_types.update({row[0]: row[1] for row in samples_schema})

            where_sql, params = build_where_and_params(filter_, filterOptions)
            where_sql = f"{where_sql} {'AND' if where_sql else 'WHERE'} r.peak_label IN ? AND s.use_for_processing = TRUE"
            params.append(selected_peaks)

            order_params: list = []
            order_values = ""
            if selected_peaks:
                order_values = ", ".join(["(?, ?)"] * len(selected_peaks))
                for idx, peak in enumerate(selected_peaks):
                    order_params.extend([idx, peak])

            # Build sorter SQL (even when peaks selected, use as secondary sort)
            sorter_sql = build_order_by(
                sorter,
                column_types,
                tie=('peak_label', 'ASC') if not selected_peaks else None,
                nocase_text=True
            )
            
            if selected_peaks:
                # When peaks are selected, preserve selection order but allow column sorting as secondary
                if sorter_sql:
                    # Extract just the ORDER BY columns (without "ORDER BY" prefix)
                    sorter_cols = sorter_sql.replace("ORDER BY ", "")
                    order_by_sql = f"ORDER BY __peak_order__, {sorter_cols}"
                else:
                    order_by_sql = "ORDER BY __peak_order__, ms_file_label"
            else:
                order_by_sql = sorter_sql

            # Check for SCALiR concentrations and dynamically add columns
            scalir_join = ""
            scalir_select = ""
            fitting_select = ""
            columns = RESULTS_TABLE_COLUMNS.copy()
            
            # Check if fitting data exists (any fit_success = TRUE)
            has_fitting_data = False
            try:
                fitting_check = conn.execute(
                    "SELECT EXISTS(SELECT 1 FROM results WHERE fit_success = TRUE LIMIT 1)"
                ).fetchone()
                has_fitting_data = fitting_check and fitting_check[0]
            except Exception:
                pass
            
            if has_fitting_data:
                # Add fitting columns before Intensity (last column)
                fitting_cols = [
                    {'title': 'peak_area_fitted', 'dataIndex': 'peak_area_fitted', 'width': '175px', 'sorter': True},
                    {'title': 'fit_r²', 'dataIndex': 'fit_r_squared', 'width': '100px', 'sorter': True},
                    {
                        'title': 'fit_success', 
                        'dataIndex': 'fit_success', 
                        'width': '110px',
                        'filterOptions': [
                            {'label': '✓ Success', 'value': '✓'},
                            {'label': '✗ Failed', 'value': '✗'},
                        ]
                    },
                ]
                insert_idx = len(columns) - 1  # Before Intensity
                for col in fitting_cols:
                    columns.insert(insert_idx, col)
                    insert_idx += 1
                fitting_select = ", r.peak_area_fitted, r.fit_r_squared, r.fit_success"
            
            conc_file = Path(wdir) / "results" / "scalir" / "concentrations.csv"
            if conc_file.exists():
                try:
                    conn.execute(f"CREATE OR REPLACE VIEW scalir_concentrations AS SELECT * FROM read_csv_auto('{conc_file}')")
                    
                    # Get available columns in the SCALiR results
                    sc_desc = conn.execute("DESCRIBE scalir_concentrations").fetchall()
                    sc_cols = [row[0] for row in sc_desc]
                    
                    # Build join condition
                    join_conds = []
                    # SCALiR stores ms_file_label in the 'ms_file' column (aliased in run_scalir)
                    if 'ms_file' in sc_cols:
                        join_conds.append("TRIM(CAST(sc.ms_file AS VARCHAR)) = TRIM(CAST(r.ms_file_label AS VARCHAR))")
                    if 'peak_label' in sc_cols:
                        join_conds.append("CAST(sc.peak_label AS VARCHAR) = CAST(r.peak_label AS VARCHAR)")

                    if join_conds:
                        scalir_join = f"LEFT JOIN scalir_concentrations sc ON ({' AND '.join(join_conds)})"
                        
                        fields_map = [
                            ('pred_conc', 'Concentration', None, '175px', 'scalir_conc'),
                            ('in_range', 'In Range', None, '150px', 'scalir_in_range'),
                            ('unit', 'Unit', None, '100px', 'scalir_unit')
                        ]
                        
                        select_fields = []
                        scalir_cols_to_insert = []
                        for field_name, title, render_type, width, alias in fields_map:
                            if field_name in sc_cols:
                                select_fields.append(f"sc.{field_name} AS {alias}")
                                col_def = {
                                    'title': title,
                                    'dataIndex': alias,
                                    'width': width
                                }
                                if render_type:
                                    col_def['renderOptions'] = {'renderType': render_type}
                                scalir_cols_to_insert.append(col_def)
                        
                        # Insert SCALiR columns before Intensity (last column)
                        if scalir_cols_to_insert:
                            insert_idx = len(columns) - 1  # Before the last column
                            for col in scalir_cols_to_insert:
                                columns.insert(insert_idx, col)
                                insert_idx += 1
                        
                        if select_fields:
                            scalir_select = ", " + ", ".join(select_fields)


                except Exception as e:
                    logger.warning(f"Failed to join SCALiR concentrations: {e}")

            # Only fetch columns that are actually displayed in the table
            # This avoids sending large unused arrays (scan_time, etc.) to the client
            filtered_sql = f"""
            WITH
            {f"target_order AS (SELECT * FROM (VALUES {order_values}) AS t(ord, target_peak_label))," if order_values else ""}
            filtered AS (
              SELECT 
                r.peak_label,
                r.ms_file_label,
                r.peak_area,
                r.peak_area_top3,
                r.peak_max,
                r.peak_n_datapoints,
                r.peak_rt_of_max
                {fitting_select}
                {scalir_select},
                -- Convert intensity array to comma-separated string for lighter JSON payload
                array_to_string(r.intensity, ',') AS intensity,
                UPPER(s.ms_type) AS ms_type,
                s.sample_type
              {", COALESCE(tord.ord, 1e9) AS __peak_order__" if order_values else ""}
              FROM results r
              LEFT JOIN samples s USING (ms_file_label)
              {f"LEFT JOIN target_order tord ON tord.target_peak_label = r.peak_label" if order_values else ""}
              {scalir_join}
              {where_sql}
            )
            """

            count_sql = filtered_sql + "SELECT COUNT(*) AS __total__ FROM filtered;"
            number_records = conn.execute(count_sql, order_params + params).fetchone()[0] or 0

            page_size_options = sorted(
                set(base_page_size_options + ([number_records] if number_records else []))
            )
            if page_size not in page_size_options:
                page_size = min(25, number_records) if number_records else 25
            if number_records and page_size > number_records:
                page_size = number_records
            effective_page_size = page_size if page_size > 0 else (number_records or 1)
            max_page = max(math.ceil(number_records / effective_page_size), 1) if number_records else 1
            current = min(max(current, 1), max_page)
            offset = (current - 1) * effective_page_size if number_records else 0

            sql = filtered_sql + f"""
            , paged AS (
              SELECT *
              FROM filtered
              {(' ' + order_by_sql) if order_by_sql else ''}
              LIMIT ? OFFSET ?
            )
            SELECT * FROM paged;
            """

            params_paged = order_params + params + [effective_page_size, offset]
            df = conn.execute(sql, params_paged).df()

        if number_records and params_paged[-1] != (current - 1) * effective_page_size:
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    logger.debug("results_table: PreventUpdate because database connection is None (repaginate)")
                    raise PreventUpdate
                params_paged = order_params + params + [effective_page_size, (current - 1) * effective_page_size]
                df = conn.execute(sql, params_paged).df()

        # Generate key column and round values for display
        df["key"] = df["peak_label"].astype(str) + "-" + df["ms_file_label"].astype(str)
        if 'scalir_conc' in df.columns:
            df['scalir_conc'] = df['scalir_conc'].round(4)
        # Round fitting columns for cleaner display
        if 'peak_area_fitted' in df.columns:
            df['peak_area_fitted'] = df['peak_area_fitted'].round(0).astype('Int64')
        if 'fit_r_squared' in df.columns:
            df['fit_r_squared'] = df['fit_r_squared'].round(2)
        # Convert fit_success boolean to display-friendly format
        if 'fit_success' in df.columns:
            df['fit_success'] = df['fit_success'].map({True: '✓', False: '✗', None: ''})
        # Convert intensity string back to list for sparkline, downsampled to max 50 points
        if 'intensity' in df.columns:
            def parse_and_downsample(s, max_points=50):
                if not s or pd.isna(s):
                    return []
                try:
                    vals = [float(x) for x in str(s).split(',') if x.strip()]
                    # Downsample if too many points
                    if len(vals) > max_points:
                        step = len(vals) / max_points
                        vals = [vals[int(i * step)] for i in range(max_points)]
                    return vals
                except Exception:
                    return []
            df['intensity'] = df['intensity'].apply(parse_and_downsample)
        # Replace NaN with None for clean JSON serialization
        df = df.where(pd.notnull(df), None)
        data = df.to_dict('records')

        # If current page is empty but there are records, navigate to the last valid page
        if len(data) == 0 and number_records > 0:
            max_page = max(math.ceil(number_records / effective_page_size), 1)
            current = max_page
            # Re-query with the adjusted page
            offset = (current - 1) * effective_page_size
            params_paged = order_params + params + [effective_page_size, offset]
            df = conn.execute(sql, params_paged).df()
            df["key"] = df["peak_label"].astype(str) + "-" + df["ms_file_label"].astype(str)
            if 'scalir_conc' in df.columns:
                df['scalir_conc'] = df['scalir_conc'].round(4)
            if 'peak_area_fitted' in df.columns:
                df['peak_area_fitted'] = df['peak_area_fitted'].round(0).astype('Int64')
            if 'fit_r_squared' in df.columns:
                df['fit_r_squared'] = df['fit_r_squared'].round(2)
            if 'fit_success' in df.columns:
                df['fit_success'] = df['fit_success'].map({True: '✓', False: '✗', None: ''})
            if 'intensity' in df.columns:
                df['intensity'] = df['intensity'].apply(parse_and_downsample)
            df = df.where(pd.notnull(df), None)
            data = df.to_dict('records')

        return [
            data,
            [],
            {
                **pagination,
                'total': number_records,
                'current': current,
                'pageSize': effective_page_size,
                'pageSizeOptions': page_size_options,
            },
            columns
        ]

    @app.callback(
        Output('processing-peak-select', 'options'),
        Output('processing-peak-select', 'value', allow_duplicate=True),
        Input('section-context', 'data'),
        Input('wdir', 'data'),
        Input('results-action-store', 'data'),
        State('processing-peak-select', 'value'),
        prevent_initial_call=True,
    )
    def load_available_peaks(section_context, wdir, results_actions, current_value):
        if section_context and section_context['page'] != 'Processing':
            logger.debug(f"load_available_peaks: PreventUpdate because current page is {section_context.get('page')}")
            raise PreventUpdate
        
        return _load_peaks_from_results(wdir, current_value)

    @app.callback(
        Output("processing-delete-confirmation-modal", "visible"),
        Output("processing-delete-confirmation-modal", "children"),

        Input("processing-options", "nClicks"),
        State("processing-options", "clickedKey"),
        State('results-table', 'selectedRows'),
        prevent_initial_call=True
    )
    def toggle_modal(nClicks, clickedKey, selectedRows):
        ctx = dash.callback_context
        if (
                not ctx.triggered or
                not nClicks or
                not clickedKey or
                clickedKey not in ['processing-delete-selected', 'processing-delete-all']
        ):
            logger.debug(f"toggle_modal: PreventUpdate because triggered={ctx.triggered}, nClicks={nClicks}, clickedKey={clickedKey}")
            raise PreventUpdate

        return _build_delete_modal_content(clickedKey, selectedRows)

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("results-action-store", "data", allow_duplicate=True),

        Input("processing-delete-confirmation-modal", "okCounts"),
        State('results-table', 'selectedRows'),
        State("processing-options", "clickedKey"),
        State("wdir", "data"),
        background=True,
        running=[
            (Output("results-table-spin", "spinning"), True, False),
            (Output("processing-delete-confirmation-modal", "confirmLoading"), True, False),
        ],
        prevent_initial_call=True,
    )
    def confirm_and_delete(okCounts, selectedRows, clickedKey, wdir):
        if okCounts is None:
            logger.debug("confirm_and_delete: PreventUpdate because okCounts is None")
            raise PreventUpdate
        if not wdir:
            logger.debug("confirm_and_delete: PreventUpdate because wdir is not set")
            raise PreventUpdate
        
        # Delegate to appropriate standalone function
        if clickedKey == "processing-delete-selected":
            error_notif, results_action_store, total_removed = _delete_selected_results(wdir, selectedRows)
        else:  # processing-delete-all
            error_notif, results_action_store, total_removed = _delete_all_results(wdir)
        
        # If there was an error, return it immediately
        if error_notif is not None:
            return (error_notif, results_action_store)
        
        # Return success notification
        return (
            fac.AntdNotification(
                message="Results deleted",
                description=f"Deleted {total_removed[0]} targets and {total_removed[1]} samples.",
                type="success" if total_removed != [0, 0] else "error",
                duration=3,
                placement='bottom',
                showProgress=True,
                stack=True
            ),
            results_action_store
        )

    @app.callback(
        Output("download-results-modal", "visible"),

        Input("processing-download-btn", "nClicks"),
        prevent_initial_call=True,
    )
    def open_download_results(n_clicks):
        if not n_clicks:
            logger.debug("open_download_results: PreventUpdate because n_clicks is None")
            raise PreventUpdate
        return True

    @app.callback(
        Output('download-all-results-btn', 'disabled'),
        Output('download-densematrix-results-btn', 'disabled'),
        Output('download-options-all-results-item', 'validateStatus'),
        Output('download-options-all-results-item', 'help'),
        Output('download-densematrix-rows-item', 'validateStatus'),
        Output('download-densematrix-rows-item', 'help'),
        Output('download-densematrix-cols-item', 'validateStatus'),
        Output('download-densematrix-cols-item', 'help'),
        Output('download-densematrix-value-item', 'validateStatus'),
        Output('download-densematrix-value-item', 'help'),

        Input('download-options-all-results', 'value'),
        Input('download-densematrix-rows', 'value'),
        Input('download-densematrix-cols', 'value'),
        Input('download-densematrix-value', 'value'),
        prevent_initial_call=True
    )
    def validate_download_input(all_results, rows, cols, value):

        if not all_results:
            all_results_status = 'error'
            all_results_help = 'Select at least one column'
            all_results_download = True
        else:
            all_results_status = 'success'
            all_results_help = None
            all_results_download = False

        if not rows:
            rows_status = 'error'
            rows_help = 'Select a valid column'
        else:
            rows_status = 'success'
            rows_help = None

        if not cols:
            cols_status = 'error'
            cols_help = 'Select a valid column'
        else:
            cols_status = 'success'
            cols_help = None

        if not value:
            value_status = 'error'
            value_help = 'Select a valid column'
        else:
            value_status = 'success'
            value_help = None

        densematrix_download = False
        if not rows or not cols or not value:
            densematrix_download = True

        if rows == cols:
            densematrix_download = True
            cols_status = 'error'
            cols_help = 'Select a different column'

        return (
            all_results_download,
            densematrix_download,
            all_results_status,
            all_results_help,
            rows_status,
            rows_help,
            cols_status,
            cols_help,
            value_status,
            value_help
        )

    @app.callback(
        Output("download-csv", "data"),
        Output("notifications-container", "children", allow_duplicate=True),
        Output("download-all-results-btn", "loading"),
        Output("download-densematrix-results-btn", "loading"),

        Input("download-all-results-btn", "nClicks"),
        Input("download-densematrix-results-btn", "nClicks"),
        State("download-options-all-results", "value"),
        State('download-densematrix-rows', 'value'),
        State('download-densematrix-cols', 'value'),
        State('download-densematrix-value', 'value'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def download_results(d_all_clicks, d_dm_clicks, d_options_value, d_dm_rows, d_dm_cols, d_dm_value, wdir):

        ctx = dash.callback_context
        if not ctx.triggered:
            logger.debug("download_results: PreventUpdate because not triggered")
            raise PreventUpdate
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if prop_id == 'download-all-results-btn' and not d_all_clicks:
            logger.debug("download_results: PreventUpdate because download-all-results-btn clicked but no count")
            raise PreventUpdate
        if prop_id == 'download-densematrix-results-btn' and not d_dm_clicks:
            logger.debug("download_results: PreventUpdate because download-densematrix-results-btn clicked but no count")
            raise PreventUpdate

        if not wdir:
            return dash.no_update, fac.AntdNotification(
                message="Download Results",
                description="Workspace not available. Please select or create a workspace.",
                type="error",
                duration=4,
                placement="bottom",
                showProgress=True,
            ), False, False

        ws_key = Path(wdir).stem
        with duckdb_connection_mint(Path(wdir).parent.parent) as mint_conn:
            if mint_conn is None:
                logger.debug("download_results: PreventUpdate because mint database connection is None")
                raise PreventUpdate
            ws_row = mint_conn.execute("SELECT name FROM workspaces WHERE key = ?", [ws_key]).fetchone()
            if ws_row is None:
                logger.debug("download_results: PreventUpdate because workspace row not found")
                raise PreventUpdate
            ws_name = ws_row[0]

        # Delegate to appropriate standalone function (returns tuple of download_data, notification)
        if prop_id == 'download-all-results-btn':
            download_data, notification = _download_all_results(wdir, ws_name, d_options_value)
            return download_data, notification, False, False
        else:
            download_data, notification = _download_dense_matrix(wdir, ws_name, d_dm_rows, d_dm_cols, d_dm_value)
            return download_data, notification, False, False

    @app.callback(
        Output('processing-tour-empty', 'current'),
        Output('processing-tour-empty', 'open'),
        Output('processing-tour-full', 'current'),
        Output('processing-tour-full', 'open'),
        Input('processing-tour-icon', 'nClicks'),
        State('workspace-status', 'data'),
        prevent_initial_call=True,
    )
    def processing_tour_open(n_clicks, workspace_status):
        has_results = workspace_status and (workspace_status.get('chromatograms_count', 0) or 0) > 0
        if has_results:
            # Open full tour, keep empty tour closed
            return 0, False, 0, True
        else:
            # Open empty tour, keep full tour closed
            return 0, True, 0, False

    @app.callback(
        Output('processing-tour-hint', 'open'),
        Output('processing-tour-hint', 'current'),
        Input('processing-tour-hint-store', 'data'),
    )
    def processing_hint_sync(store_data):
        if not store_data:
            logger.debug("processing_hint_sync: PreventUpdate because store_data is empty")
            raise PreventUpdate
        return store_data.get('open', True), 0

    @app.callback(
        Output('processing-tour-hint-store', 'data'),
        Input('processing-tour-hint', 'closeCounts'),
        Input('processing-tour-icon', 'nClicks'),
        State('processing-tour-hint-store', 'data'),
        prevent_initial_call=True,
    )
    def processing_hide_hint(close_counts, n_clicks, store_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            logger.debug("processing_hide_hint: PreventUpdate because not triggered")
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'processing-tour-icon':
            return {'open': False}

        if close_counts:
            return {'open': False}

        return store_data or {'open': True}

    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("processing-modal", "visible"),
        Output("processing-warning", "style"),
        Output("processing-progress", "percent", allow_duplicate=True),
        Output("processing-progress-stage", "children", allow_duplicate=True),
        Output("processing-progress-detail", "children", allow_duplicate=True),
        Output("processing-recompute", "checked"),
        Output("processing-chromatogram-compute-cpu", "value"),
        Output("processing-chromatogram-compute-ram", "value"),
        Output("processing-chromatogram-compute-cpu-item", "help", allow_duplicate=True),
        Output("processing-chromatogram-compute-ram-item", "help", allow_duplicate=True),

        Input("processing-btn", "nClicks"),
        Input("processing-empty-btn", "nClicks"),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def open_run_mint_modal(nClicks, nClicks_empty, wdir):
        if not nClicks and not nClicks_empty:
            logger.debug("open_run_mint_modal: PreventUpdate because nClicks is None")
            raise PreventUpdate


        computed_results = 0
        # check if some results was computed
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return (
                    fac.AntdNotification(
                        message="Run MINT",
                        description="Workspace not available. Please select or create a workspace.",
                        type="error",
                        duration=4,
                        placement="bottom",
                        showProgress=True,
                    ),
                    False,
                    dash.no_update,
                    0,
                    "",
                    "",
                    False,
                    dash.no_update,
                    dash.no_update,
                )

            ms_files = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
            targets = conn.execute("SELECT COUNT(*) FROM targets").fetchone()

            if not ms_files or ms_files[0] == 0 or not targets or targets[0] == 0:
                return (
                    fac.AntdNotification(
                        message="Requirements not met",
                        description="At least one MS-file and one target are required.",
                        type="warning",
                        duration=4,
                        placement="bottom",
                        showProgress=True,
                    ),
                    False,
                    dash.no_update,
                    0,
                    "",
                    "",
                    False,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )

            results = conn.execute("SELECT COUNT(*) FROM results").fetchone()
            if results:
                computed_results = results[0]

        style = {'display': 'block', 'marginBottom': '2rem'} if computed_results else {'display': 'none'}

        recompute = bool(computed_results)

        # Smart Default CPU/RAM
        from os import cpu_count
        n_cpus_total = cpu_count()
        default_cpus = max(1, n_cpus_total // 2)
        
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        default_ram = round(min(float(default_cpus), available_ram_gb), 1)
        available_ram_gb_rounded = round(available_ram_gb, 1)

        help_cpu = f"Selected {default_cpus} / {n_cpus_total} cpus"
        help_ram = f"Selected {default_ram}GB / {available_ram_gb_rounded}GB available RAM"

        return (
            dash.no_update, 
            True, 
            style, 
            0, 
            "", 
            "", 
            recompute,
            default_cpus,
            default_ram,
            help_cpu,
            help_ram
        )

    @app.callback(
        Output('results-action-store', 'data'),
        Output('processing-modal', 'visible', allow_duplicate=True),
        Output('workspace-status', 'data', allow_duplicate=True),

        Input('processing-modal', 'okCounts'),
        State('processing-recompute', 'checked'),
        State('processing-enable-fitting', 'checked'),
        State("processing-chromatogram-compute-cpu", "value"),
        State("processing-chromatogram-compute-ram", "value"),
        State('processing-chromatogram-compute-batch-size', "value"),
        State('processing-targets-selection', 'checked'),
        State('wdir', 'data'),
        background=True,
        running=[
            (Output('processing-progress-container', 'style'), {
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "flexDirection": "column",
                "minWidth": "200px",
                "maxWidth": "400px",
                "margin": "auto",
                'height': "60vh"
            }, {'display': 'none'}),
            (Output("processing-options-container", "style"), {'display': 'none'}, {'display': 'flex'}),

            (Output('processing-modal', 'confirmAutoSpin'), True, False),
            (Output('processing-modal', 'cancelButtonProps'), {'disabled': True},
             {'disabled': False}),
            (Output('processing-modal', 'confirmLoading'), True, False),
        ],
        progress=[
            Output("processing-progress", "percent", allow_duplicate=True),
            Output("processing-progress-stage", "children", allow_duplicate=True),
            Output("processing-progress-detail", "children", allow_duplicate=True),
        ],
        cancel=[
            Input('cancel-processing', 'nClicks')
        ],
        prevent_initial_call=True
    )
    def compute_results(set_progress, okCounts, recompute, enable_fitting, n_cpus, ram, batch_size, bookmarked, wdir):

        if not okCounts:
            logger.debug("compute_results: PreventUpdate because okCounts is None")
            raise PreventUpdate

        activate_workspace_logging(wdir)
        start = time.perf_counter()
        
        def progress_adapter(percent, stage="", detail=""):
            if set_progress:
                set_progress((percent, stage or "", detail or ""))

        try:
            logger.info('Starting full processing run.')
            logger.info('Computing chromatograms...')
            progress_adapter(0, "Chromatograms", "Preparing batches...")
            compute_chromatograms_in_batches(wdir, use_for_optimization=False, batch_size=batch_size,
                                             set_progress=progress_adapter, recompute_ms1=False,
                                             recompute_ms2=False, n_cpus=n_cpus, ram=ram, use_bookmarked=bookmarked)
            
            logger.info('Computing results...')
            progress_adapter(0, "Results", "Preparing batches...")
            compute_results_in_batches(wdir=wdir,
                               use_bookmarked= bookmarked,
                               recompute = recompute,
                               batch_size = batch_size,
                               checkpoint_every = 10,
                               set_progress=progress_adapter,
                               n_cpus=n_cpus,
                               ram=ram)

            # Run peak fitting if enabled
            if enable_fitting:
                logger.info('Computing peak fitting (EMG)...')
                progress_adapter(0, "Peak Fitting", "Starting EMG fitting...")
                fit_stats = compute_fitted_results(
                    wdir=wdir,
                    use_bookmarked=bookmarked,
                    recompute=recompute,
                    n_workers=n_cpus or 8,
                    set_progress=progress_adapter,
                    n_cpus=n_cpus,
                    ram=ram
                )
                logger.info(f"Peak fitting complete: {fit_stats}")

            # Persist the results table to a workspace folder for resilience
            progress_adapter(100, "Finalizing", "Backing up results...")
            try:
                results_dir = Path(wdir) / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                backup_path = results_dir / "results_backup.csv"
                
                # Use DuckDB's native COPY for much faster CSV export
                with duckdb_connection(wdir) as conn:
                    conn.execute(
                        "COPY (SELECT * FROM results) TO ? (HEADER, DELIMITER ',')",
                        (str(backup_path),)
                    )
                logger.info(f"Backed up results to {backup_path}")
            except Exception:
                logger.warning("Failed to backup results to CSV.", exc_info=True)


            logger.info(f"Results computed in {time.perf_counter() - start:.2f} seconds")
        except Exception:
             logger.error("Processing failed during computation", exc_info=True)
             raise

        # Update workspace-status with current counts including results
        workspace_status = {
            'ms_files_count': 0,
            'targets_count': 0,
            'chromatograms_count': 0,
            'selected_targets_count': 0,
            'optimization_samples_count': 0,
            'results_count': 0
        }
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
                    n_cpus_count = cpu_count()
                    default_cpus = max(1, n_cpus_count // 2)
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
                        'n_cpus': n_cpus_count,
                        'default_cpus': default_cpus,
                        'ram_avail': round(ram_avail, 1),
                        'default_ram': default_ram
                    }
        logger.info(f"workspace-status updated after processing: {workspace_status}")

        return {'action': 'processing', 'status': 'completed', 'timestamp': time.time()}, False, workspace_status

    @app.callback(
        Output('results-action-store', 'data', allow_duplicate=True),
        Output('processing-progress-container', 'style', allow_duplicate=True),
        Output('processing-options-container', 'style', allow_duplicate=True),
        Output('processing-modal', 'visible', allow_duplicate=True),
        Output('processing-progress', 'percent', allow_duplicate=True),
        Output('processing-progress-stage', 'children', allow_duplicate=True),
        Output('processing-progress-detail', 'children', allow_duplicate=True),
        Input('cancel-processing', 'nClicks'),
        prevent_initial_call=True
    )
    def cancel_results_processing(cancel_clicks):
        if not cancel_clicks:
            logger.debug("cancel_results_processing: PreventUpdate because cancel_clicks is None")
            raise PreventUpdate
        logger.info("MINT processing cancelled by user.")
        return (
            {'action': 'processing', 'status': 'cancelled', 'timestamp': time.time()},
            {'display': 'none'},
            {'display': 'flex'},
            False,
            0,
            "",
            "",
        )

    @app.callback(
        Output("processing-chromatogram-compute-cpu-item", "help", allow_duplicate=True),
        Output("processing-chromatogram-compute-ram-item", "help", allow_duplicate=True),
        Output("processing-chromatogram-compute-batch-size", "value"),
        Input("processing-chromatogram-compute-cpu", "value"),
        Input("processing-chromatogram-compute-ram", "value"),
        prevent_initial_call=True
    )
    def update_processing_resource_help(cpu, ram):
        help_cpu = _get_cpu_help_text(cpu)
        help_ram = _get_ram_help_text(ram)
        # Auto-calculate optimal batch size based on current CPU and RAM
        optimal_batch = calculate_optimal_batch_size(
            round(ram) if ram else 8,
            100000,  # Estimate for total pairs
            int(cpu) if cpu else max(1, cpu_count() // 2)
        )
        return help_cpu, help_ram, optimal_batch

    # ====================
    # SCALiR Callbacks
    # ====================

    # Toggle SCALiR button disabled state based on results existence
    app.clientside_callback(
        """(status) => {
            if (!status) return true;
            return (status.results_count || 0) === 0;
        }""",
        Output('scalir-modal-btn', 'disabled'),
        Input('workspace-status', 'data'),
    )

    # Open SCALiR modal and load existing results if available
    app.callback(
        Output('scalir-modal', 'visible'),
        Output('scalir-status-text', 'children'),
        Output('scalir-conc-path', 'children'),
        Output('scalir-metabolite-select', 'options'),
        Output('scalir-metabolite-select', 'value'),
        Output('scalir-plot-graphs', 'children'),
        Output('scalir-plot-graphs', 'style'),
        Output('scalir-results-store', 'data'),
        Input('scalir-modal-btn', 'nClicks'),
        State('wdir', 'data'),
        prevent_initial_call=True,
    )(open_scalir_modal)

    # Show standards filename
    app.callback(
        Output('scalir-standards-note', 'children'),
        Input('scalir-standards-upload', 'filename'),
        prevent_initial_call=True,
    )(show_standards_filename)

    # Run SCALiR
    app.callback(
        Output('scalir-status-text', 'children', allow_duplicate=True),
        Output('scalir-conc-path', 'children', allow_duplicate=True),
        Output('scalir-metabolite-select', 'options', allow_duplicate=True),
        Output('scalir-metabolite-select', 'value', allow_duplicate=True),
        Output('scalir-plot-graphs', 'children', allow_duplicate=True),
        Output('scalir-plot-graphs', 'style', allow_duplicate=True),
        Output('scalir-results-store', 'data', allow_duplicate=True),
        Output('scalir-run-btn', 'loading', allow_duplicate=True),
        Input('scalir-run-btn', 'nClicks'),
        State('scalir-standards-upload', 'contents'),
        State('scalir-standards-upload', 'filename'),
        State('scalir-intensity', 'value'),
        State('scalir-slope-mode', 'value'),
        State('scalir-slope-low', 'value'),
        State('scalir-slope-high', 'value'),
        State('scalir-generate-plots', 'checked'),
        State('wdir', 'data'),
        State('section-context', 'data'),
        prevent_initial_call=True,
    )(run_scalir)

    # Reset SCALiR modal
    app.callback(
        Output('scalir-status-text', 'children', allow_duplicate=True),
        Output('scalir-conc-path', 'children', allow_duplicate=True),
        Output('scalir-metabolite-select', 'options', allow_duplicate=True),
        Output('scalir-metabolite-select', 'value', allow_duplicate=True),
        Output('scalir-plot-graphs', 'children', allow_duplicate=True),
        Output('scalir-plot-graphs', 'style', allow_duplicate=True),
        Output('scalir-results-store', 'data', allow_duplicate=True),
        Output('scalir-standards-upload', 'contents', allow_duplicate=True),
        Output('scalir-standards-upload', 'filename', allow_duplicate=True),
        Output('scalir-standards-note', 'children', allow_duplicate=True),
        Input('scalir-reset-btn', 'nClicks'),
        prevent_initial_call=True,
    )(reset_scalir)

    # Update SCALiR plots based on metabolite selection
    app.callback(
        Output('scalir-plot-graphs', 'children', allow_duplicate=True),
        Output('scalir-plot-graphs', 'style', allow_duplicate=True),
        Output('scalir-plot-path', 'children'),
        Input('scalir-metabolite-select', 'value'),
        State('scalir-results-store', 'data'),
        prevent_initial_call=True,
    )(update_scalir_plot)



def open_scalir_modal(n_clicks, wdir):
    if not n_clicks:
        raise PreventUpdate
    
    # Default empty return (modal visible, everything else empty)
    empty_ret = (
        True, "", "", [], None, [], 
        {'display': 'none', 'flexGrow': 1, 'minWidth': '350px'}, 
        None
    )

    if not wdir:
        return empty_ret

    output_dir = Path(wdir) / "results" / "scalir"
    train_frame_path = output_dir / "train_frame.csv"
    params_path = output_dir / "standard_curve_parameters.csv"
    
    # If results don't exist, just open empty modal
    if not train_frame_path.exists() or not params_path.exists():
        return empty_ret

    try:
        train_frame = pd.read_csv(train_frame_path)
        params = pd.read_csv(params_path)
        
        units_path = output_dir / "units.csv"
        units_filtered = pd.read_csv(units_path) if units_path.exists() else None
        concentrations_path = output_dir / "concentrations.csv"
        
        common = sorted(train_frame['peak_label'].unique())
        metabolite_options = [{'label': label, 'value': label} for label in common]
        first_label = common[0] if common else None

        PLOTLY_HIGH_RES_CONFIG = {
            'displayModeBar': False,
            'displaylogo': False,
        }

        # Generate initial plot
        plots = []
        if first_label and not train_frame.empty:
            fig = _plot_curve_fig(train_frame, first_label, units_filtered, params)
            plots.append(
                dcc.Graph(
                    figure=fig,
                    style={'width': '100%', 'height': '450px'},
                    config=PLOTLY_HIGH_RES_CONFIG,
                )
            )
        
        plot_style = {
            'display': 'block', 
            'flexGrow': 1, 
            'minWidth': '350px', 
            'maxHeight': '400px', 
            'overflowY': 'auto'
        } if plots else {'display': 'none'}

        store_data = {
            "train_frame": train_frame.to_json(orient="split"),
            "units": units_filtered.to_json(orient="split") if units_filtered is not None else None,
            "params": params.to_json(orient="split"),
            "plot_dir": str(output_dir / "plots"),
            "common": common,
            "generated_all_plots": True
        }
        
        status_text = f"Loaded results for {len(common)} metabolites."
        conc_text = f"Concentrations: {concentrations_path}" if concentrations_path.exists() else ""
        
        return (True, status_text, conc_text, metabolite_options, first_label, plots, plot_style, store_data)

    except Exception as e:
        logger.error(f"SCALiR: Failed to load existing results: {e}", exc_info=True)
        return empty_ret


def show_standards_filename(filename):
    if filename:
        return filename
    return "No standards file selected."


def run_scalir(n_clicks, standards_contents, standards_filename, intensity, slope_mode,
               slope_low, slope_high, generate_plots, wdir, section_context):
    if not n_clicks:
        raise PreventUpdate
    if not section_context or section_context.get('page') != 'Processing':
        raise PreventUpdate

    import plotly.graph_objects as go
    PLOTLY_HIGH_RES_CONFIG = {
        'toImageButtonOptions': {
            'format': 'png',
            'scale': 4,
            'height': None,
            'width': None,
        },
        'displayModeBar': True,
        'displaylogo': False,
    }

    hidden_style = {
        'display': 'none',
        'flexWrap': 'wrap',
        'gap': '16px',
        'paddingTop': '8px',
        'justifyContent': 'flex-start',
    }
    if not wdir:
        return ("No active workspace.", "", [], None, [], hidden_style, None, False)
    try:
        logger.info(f"SCALiR: Parsing standards file {standards_filename}...")
        standards_df = _parse_uploaded_standards(standards_contents, standards_filename)
    except Exception as exc:
        logger.error(f"SCALiR: Failed to parse standards file: {exc}")
        return (f"Upload a standards table (CSV). Error: {exc}", "", [], None, [], hidden_style, None, False)

    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.error("SCALiR: Failed to connect to database.")
            return ("Database connection failed.", "", [], None, [], hidden_style, None, False)
        if intensity not in SCALIR_ALLOWED_METRICS:
            intensity = 'peak_area'
        try:
            mint_df = conn.execute(f"""
                SELECT ms_file_label AS ms_file, peak_label, {intensity}
                FROM results
                WHERE {intensity} IS NOT NULL
            """).df()
        except Exception as exc:
            logger.error(f"SCALiR: Could not load results from database: {exc}")
            return (f"Could not load results: {exc}", "", [], None, [], hidden_style, None, False)

    if mint_df.empty:
        return ("No results found for calibration.", "", [], None, [], hidden_style, None, False)

    units_df = None
    if "unit" in standards_df.columns:
        units_df = standards_df[["peak_label", "unit"]].copy()
        standards_df = standards_df.drop(columns=["unit"])

    try:
        mint_filtered, standards_filtered, units_filtered, common = intersect_peaks(
            mint_df, standards_df, units_df
        )
    except Exception as exc:
        logger.error(f"SCALiR: Alignment failed: {exc}", exc_info=True)
        return (f"Could not align standards with results: {exc}", "", [], None, [], hidden_style, None, False)

    if not common:
        return ("No overlapping peak_label values between results and standards.", "", [], None, [], hidden_style, None, False)

    low = slope_low or 0.85
    high = slope_high or 1.15
    slope_interval = (min(low, high), max(low, high))

    try:
        logger.info(f"SCALiR: Before fit - mint_filtered: {len(mint_filtered)} rows, cols: {list(mint_filtered.columns)}")
        logger.info(f"SCALiR: Before fit - standards_filtered: {len(standards_filtered)} rows, cols: {list(standards_filtered.columns)}")
        estimator, std_results, x_train, y_train, params = fit_estimator(
            mint_filtered, standards_filtered, intensity, slope_mode or "fixed", slope_interval
        )
        logger.info(f"SCALiR: After fit - std_results: {len(std_results)} rows, x_train: {len(x_train)} rows")
        logger.info(f"SCALiR: Fitting completed. Metabolites: {len(common)}")
        concentrations = build_concentration_table(
            estimator, mint_filtered, intensity, units_filtered
        )
    except Exception as exc:
        logger.error(f"SCALiR: Fitting failed: {exc}", exc_info=True)
        return (f"Error fitting calibration: {exc}", "", [], None, [], hidden_style, None, False)

    output_dir = Path(wdir) / "results" / "scalir"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    if generate_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    concentrations_path = output_dir / "concentrations.csv"
    params_path = output_dir / "standard_curve_parameters.csv"
    concentrations.to_csv(concentrations_path, index=False)
    params.to_csv(params_path, index=False)

    # Save train_frame and units for persistence
    try:
        logger.info(f"SCALiR: x_train has {len(x_train)} rows, columns: {list(x_train.columns)}")
        if 'value' in x_train.columns:
            logger.info(f"SCALiR: x_train.value stats - min: {x_train['value'].min()}, max: {x_train['value'].max()}, values > 0: {(x_train['value'] > 0).sum()}")
        train_frame = training_plot_frame(estimator, x_train, y_train, params)
        train_frame_path = output_dir / "train_frame.csv"
        train_frame.to_csv(train_frame_path, index=False)
        logger.info(f"SCALiR: train_frame has {len(train_frame)} rows, columns: {list(train_frame.columns)}")
    except Exception as exc:
        logger.error(f"SCALiR: Failed to build train_frame: {exc}", exc_info=True)
        train_frame = pd.DataFrame()

    if units_filtered is not None:
        units_path = output_dir / "units.csv"
        units_filtered.to_csv(units_path, index=False)

    if generate_plots and not train_frame.empty:
        for label in common:
            plot_standard_curve(train_frame, label, units_filtered, plots_dir)

    sorted_common = sorted(common)
    metabolite_options = [{'label': label, 'value': label} for label in sorted_common]
    first_label = sorted_common[0] if sorted_common else None
    initial_selection = first_label  # Single value for single-select mode

    PLOTLY_HIGH_RES_CONFIG = {
        'displayModeBar': False,
        'displaylogo': False,
    }

    # Generate initial plot for the first selected metabolite
    plots = []
    if first_label and not train_frame.empty:
        fig = _plot_curve_fig(train_frame, first_label, units_filtered, params)
        plots.append(
            dcc.Graph(
                figure=fig,
                style={
                    'width': '100%',
                    'height': '450px',
                },
                config=PLOTLY_HIGH_RES_CONFIG,
            )
        )

    plot_style = {
        'display': 'block' if plots else 'none',
        'flexGrow': 1,
        'minWidth': '350px',
        'maxHeight': '400px',
        'overflowY': 'auto',
    }

    store_data = {
        "train_frame": train_frame.to_json(orient="split") if not train_frame.empty else None,
        "units": units_filtered.to_json(orient="split") if units_filtered is not None else None,
        "params": params.to_json(orient="split") if not params.empty else None,
        "plot_dir": str(plots_dir),
        "common": common,
        "generated_all_plots": bool(generate_plots),
    }

    status_text = f"Fitted {len(common)} metabolites with intensity '{intensity}'."
    return (
        status_text,
        f"Concentrations: {concentrations_path}",
        metabolite_options,
        initial_selection,
        plots,
        plot_style,
        store_data,
        False,
    )


def reset_scalir(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return (
        "",
        "",
        [],
        None,  # Single value for single-select
        [],
        {
            'display': 'none',
            'flexGrow': 1,
            'minWidth': '350px',
        },
        None,
        None,
        None,
        "No standards file selected.",
    )


def update_scalir_plot(selected_label, store_data):
    import plotly.graph_objects as go
    PLOTLY_HIGH_RES_CONFIG = {
        'displayModeBar': False,
        'displaylogo': False,
    }

    # If no selection, hide plots
    if not selected_label:
        return [], {'display': 'none'}, ""

    if not store_data:
        raise PreventUpdate

    train_frame_json = store_data.get("train_frame")
    if not train_frame_json:
        raise PreventUpdate

    train_frame = pd.read_json(StringIO(train_frame_json), orient="split")
    units_json = store_data.get("units")
    units_df = pd.read_json(StringIO(units_json), orient="split") if units_json else None
    params_json = store_data.get("params")
    params_df = pd.read_json(StringIO(params_json), orient="split") if params_json else None

    # Single metabolite plot
    fig = _plot_curve_fig(train_frame, selected_label, units_df, params_df)
    plots = [
        dcc.Graph(
            figure=fig,
            style={
                'width': '100%',
                'maxWidth': '500px',
                'height': '400px',
            },
            config=PLOTLY_HIGH_RES_CONFIG,
        )
    ]

    plot_dir = Path(store_data.get("plot_dir", ""))
    plot_path = ""
    if selected_label and store_data.get("generated_all_plots") and plot_dir:
        candidate = plot_dir / f"{slugify_label(selected_label)}_curve.png"
        if candidate.exists():
            plot_path = f"Plot saved at: {candidate}"

    return plots, {
        'display': 'block',
        'flexGrow': 1,
        'minWidth': '350px',
    }, plot_path


def _plot_curve_fig(frame: pd.DataFrame, peak_label: str, units: pd.DataFrame = None, params_df: pd.DataFrame = None):
    """Generate a calibration curve plot for a given peak_label."""
    import plotly.graph_objects as go

    subset = frame[frame.peak_label == peak_label]
    if subset.empty:
        return go.Figure()

    unit = None
    if units is not None and "unit" in units.columns:
        match = units[units.peak_label == peak_label]
        if not match.empty:
            unit = match.unit.iloc[0]

    in_range = subset[subset.in_range == 1]
    out_range = subset[subset.in_range != 1]

    fig = go.Figure()

    if not out_range.empty:
        fig.add_trace(
            go.Scatter(
                x=out_range.true_conc,
                y=out_range.value,
                mode="markers",
                marker=dict(color="#888888", size=10),
                name="Outside range",
            )
        )
    if not in_range.empty:
        fig.add_trace(
            go.Scatter(
                x=in_range.true_conc,
                y=in_range.value,
                mode="markers",
                marker=dict(color="#111111", size=10),
                name="In range",
            )
        )
        line_data = in_range.sort_values("pred_conc")
        fig.add_trace(
            go.Scatter(
                x=line_data.pred_conc,
                y=line_data.value,
                mode="lines",
                line=dict(color="#111111", width=2),
                name="Fit",
            )
        )

    xlabel = f"{peak_label} concentration"
    if unit:
        xlabel = f"{xlabel} ({unit})"
    params_text = None
    if params_df is not None and not params_df.empty:
        try:
            row = params_df[params_df.peak_label == peak_label].iloc[0]
            slope = row.get('slope', None)
            intercept = row.get('intercept', None)
            lloq = row.get('LLOQ', None)
            uloq = row.get('ULOQ', None)
            params_text = (
                f"slope={slope:.1f}<br>"
                f"intercept={intercept:.0f}<br>"
                f"LLOQ={lloq:.2g}<br>"
                f"ULOQ={uloq:.2g}"
            )
        except Exception:
            params_text = None
    fig.update_layout(
        xaxis=dict(
            title=xlabel,
            type="log",
            ticks="outside",
        ),
        yaxis=dict(
            title=f"Intensity (AU)",
            type="log",
            tickformat="~s",
            tickmode="auto",
            nticks=6,
            ticks="outside",
        ),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=60, r=20, t=50, b=60),
        template="plotly_white",
    )
    if params_text:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.03,
            y=0.97,
            xanchor="left",
            yanchor="top",
            text=params_text,
            showarrow=False,
            font=dict(size=12, color="#444"),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderpad=4,
        )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig
