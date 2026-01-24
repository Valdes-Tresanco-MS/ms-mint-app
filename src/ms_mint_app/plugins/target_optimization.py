import json
import logging
from os import cpu_count

import dash
import feffery_antd_components as fac
import math
import numpy as np
import plotly.graph_objects as go
import psutil
import time
from pathlib import Path
from dash import html, dcc, Patch
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from ..duckdb_manager import (
    duckdb_connection,
    compute_chromatograms_in_batches,
    calculate_optimal_batch_size,
    populate_full_range_downsampled_chromatograms_for_target,
)
from ..plugin_interface import PluginInterface
from ..tools import sparsify_chrom, proportional_min1_selection
from ..plugins.analysis_tools.trace_helper import (
    generate_chromatogram_traces,
    calculate_rt_alignment,
    calculate_shifts_per_sample_type,
    apply_savgol_smoothing,
    apply_lttb_downsampling,
)
from ..rt_span_optimizer import optimize_rt_spans_batch
from .workspaces import activate_workspace_logging

_label = "Optimization"

logger = logging.getLogger(__name__)
LTTB_TARGET_POINTS = 100
FULL_RANGE_DOWNSAMPLE_POINTS = 1000
SAVGOL_WINDOW = 10
SAVGOL_ORDER = 2


class TargetOptimizationPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 6
        logger.info(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)

    def outputs(self):
        return None


def _get_cpu_help_text(cpu):
    n_cpus_total = cpu_count()
    return f"Selected {cpu} / {n_cpus_total} cpus"


def _get_ram_help_text(ram):
    ram_max = round(psutil.virtual_memory().available / (1024 ** 3), 1)
    return f"Selected {ram}GB / {ram_max}GB available RAM"


def downsample_for_preview(scan_time, intensity, max_points=100):
    """Reduce puntos manteniendo la forma general"""
    if len(scan_time) <= max_points:
        return scan_time, intensity

    indices = np.linspace(0, len(scan_time) - 1, max_points, dtype=int)
    return scan_time[indices], intensity[indices]



def get_chromatogram_dataframe(conn, target_label, full_range=False, wdir=None):
    """
    Fetches chromatogram data for a specific target.
    If full_range is True, queries the raw ms1/ms2_data tables (slower but complete).
    If False, queries the cached chromatograms table (faster but sliced to RT window).
    """
    if full_range:
        # 1. Get target metadata
        t_info = conn.execute("""
            SELECT ms_type, mz_mean, mz_width, rt_min, rt_max 
            FROM targets 
            WHERE peak_label = ?
        """, [target_label]).fetchone()
        
        if not t_info:
            return None
            
        ms_type, mz_mean, mz_width_ppm, rt_min, rt_max = t_info
        
        if ms_type not in ['ms1', 'ms2']:
             return None

        if ms_type == 'ms1':
            has_full_ds = conn.execute(
                """
                SELECT COUNT(*)
                FROM chromatograms
                WHERE peak_label = ?
                  AND ms_type = 'ms1'
                  AND scan_time_full_ds IS NOT NULL
                """,
                [target_label],
            ).fetchone()[0]
            if not has_full_ds and wdir:
                populate_full_range_downsampled_chromatograms_for_target(
                    wdir,
                    target_label,
                    n_out=FULL_RANGE_DOWNSAMPLE_POINTS,
                    conn=conn,
                )
                has_full_ds = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM chromatograms
                    WHERE peak_label = ?
                      AND ms_type = 'ms1'
                      AND scan_time_full_ds IS NOT NULL
                    """,
                    [target_label],
                ).fetchone()[0]

            if has_full_ds:
                query = """
                    WITH picked_samples AS (
                        SELECT ms_file_label, color, label, sample_type
                        FROM samples
                        WHERE use_for_optimization = TRUE
                    ),
                    base AS (
                        SELECT
                            c.ms_file_label,
                            s.color,
                            s.label,
                            s.sample_type,
                            c.scan_time_full_ds AS scan_time,
                            c.intensity_full_ds AS intensity
                        FROM chromatograms c
                        JOIN picked_samples s USING (ms_file_label)
                        WHERE c.peak_label = ?
                          AND c.ms_type = 'ms1'
                          AND c.scan_time_full_ds IS NOT NULL
                          AND c.intensity_full_ds IS NOT NULL
                    ),
                    zipped AS (
                        SELECT
                            ms_file_label,
                            color,
                            label,
                            sample_type,
                            list_transform(
                                range(1, len(scan_time) + 1),
                                i -> struct_pack(
                                    t := list_extract(scan_time, i),
                                    i := list_extract(intensity, i)
                                )
                            ) AS pairs
                        FROM base
                    ),
                    final AS (
                        SELECT
                            ms_file_label,
                            color,
                            label,
                            sample_type,
                            list_transform(pairs, p -> p.t) AS scan_time_sliced,
                            list_transform(pairs, p -> p.i) AS intensity_sliced,
                            CASE
                                WHEN len(pairs) = 0 THEN NULL
                                ELSE list_max(list_transform(pairs, p -> p.i)) * 1.10
                            END AS intensity_max_in_range,
                            CASE
                                WHEN len(pairs) = 0 THEN NULL
                                ELSE list_min(list_transform(pairs, p -> p.i))
                            END AS intensity_min_in_range,
                            CASE
                                WHEN len(pairs) = 0 THEN NULL
                                ELSE list_max(list_transform(pairs, p -> p.t))
                            END AS scan_time_max_in_range,
                            CASE
                                WHEN len(pairs) = 0 THEN NULL
                                ELSE list_min(list_transform(pairs, p -> p.t))
                            END AS scan_time_min_in_range
                        FROM zipped
                    )
                    SELECT *
                    FROM final
                    ORDER BY ms_file_label
                """
                return conn.execute(query, [target_label]).pl()

        # Calculate m/z window
        delta_mz = mz_mean * mz_width_ppm / 1e6
        mz_lower = mz_mean - delta_mz
        mz_upper = mz_mean + delta_mz
        
        table_name = "ms1_data" if ms_type == "ms1" else "ms2_data"
        
        # 2. Query raw data
        # Note: We group by file and aggregate directly to match the format of the 'chromatograms' table query
        # We assume ms_file_scans has the timing info.
        query = f"""
        WITH picked_samples AS (
            SELECT ms_file_label, color, label, sample_type
            FROM samples 
            WHERE use_for_optimization = TRUE
        ),
        raw_scans AS (
            SELECT d.ms_file_label, d.scan_id, d.intensity, s.scan_time
            FROM {table_name} d
            JOIN ms_file_scans s ON d.ms_file_label = s.ms_file_label AND d.scan_id = s.scan_id
            WHERE d.mz BETWEEN ? AND ?
              AND s.ms_type = ?
              AND d.ms_file_label IN (SELECT ms_file_label FROM picked_samples)
        )
        SELECT 
            r.ms_file_label, 
            p.color, 
            p.label, 
            p.sample_type, 
            LIST(r.scan_time ORDER BY r.scan_time) as scan_time_sliced,
            LIST(r.intensity ORDER BY r.scan_time) as intensity_sliced,
            MAX(r.intensity) * 1.10 as intensity_max_in_range,
            MIN(r.intensity) as intensity_min_in_range,
            MAX(r.scan_time) as scan_time_max_in_range,
            MIN(r.scan_time) as scan_time_min_in_range
        FROM raw_scans r
        JOIN picked_samples p ON r.ms_file_label = p.ms_file_label
        GROUP BY r.ms_file_label, p.color, p.label, p.sample_type
        ORDER BY r.ms_file_label
        """
        
        return conn.execute(query, [mz_lower, mz_upper, ms_type]).pl()

    else:
        # specific standard query for cached chromatograms
        query = """
            WITH picked_samples AS (SELECT ms_file_label, color, label, sample_type
                                    FROM samples
                                    WHERE use_for_optimization = TRUE
            ),
             picked_target AS (SELECT peak_label,
                                      intensity_threshold
                               FROM targets
                               WHERE peak_label = ?),
             base AS (SELECT c.*,
                             s.color,
                             s.label,
                             s.sample_type,
                             t.intensity_threshold
                      FROM chromatograms c
                               JOIN picked_samples s USING (ms_file_label)
                               JOIN picked_target t USING (peak_label)),
             zipped AS (SELECT ms_file_label,
                               color,
                               label,
                               sample_type,
                               intensity_threshold,
                               list_transform(
                                       range(1, len(scan_time) + 1),
                                       i -> struct_pack(
                                               t := list_extract(scan_time, i),
                                               i := list_extract(intensity,  i)
                                            )
                               ) AS pairs
                        FROM base),

             sliced AS (SELECT ms_file_label,
                               color,
                               label,
                               sample_type,
                               pairs,
                               list_filter(pairs, p -> p.i >= COALESCE(intensity_threshold, 0)) AS pairs_in
                        FROM zipped),
             final AS (SELECT ms_file_label,
                              color,
                              label,
                              sample_type,
                              list_transform(pairs_in, p -> p.t)                            AS scan_time_sliced,
                              list_transform(pairs_in, p -> p.i)                            AS intensity_sliced,
                              CASE
                                  WHEN len(pairs) = 0 THEN NULL
                                  ELSE list_max(list_transform(pairs, p -> p.i)) * 1.10 END AS
                                                                                               intensity_max_in_range,
                              CASE
                                  WHEN len(pairs) = 0 THEN NULL
                                  ELSE list_min(list_transform(pairs, p -> p.i)) END        AS intensity_min_in_range,
                              CASE
                                  WHEN len(pairs) = 0 THEN NULL
                                  ELSE list_max(list_transform(pairs, p -> p.t)) END        AS scan_time_max_in_range,
                              CASE
                                  WHEN len(pairs) = 0 THEN NULL
                                  ELSE list_min(list_transform(pairs, p -> p.t)) END        AS scan_time_min_in_range

                       FROM sliced)
        SELECT *
        FROM final
        ORDER BY ms_file_label;
            """
        return conn.execute(query, [target_label]).pl()


MAX_NUM_CARDS = 20  # Support up to 20 cards/page while keeping load time reasonable
DEFAULT_GRAPH_WIDTH = 250
DEFAULT_GRAPH_HEIGHT = 180

# Use a valid empty Plotly figure (not `{}`) so container resizes/redraws don't
# trigger Plotly's "_doPlot" warning for graphs that haven't been plotted yet.
EMPTY_PLOTLY_FIGURE = {"data": [], "layout": {"template": "plotly_white"}}

# High-resolution export configuration for Plotly graphs
PLOTLY_HIGH_RES_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'scale': 4,  # 4x scale ≈ 300 DPI
        'height': None,
        'width': None,
    },
    'displayModeBar': True,
    'displaylogo': False,
}

_layout = fac.AntdLayout(
    [
        fac.AntdHeader(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'Optimization', level=4, style={'margin': '0', 'whiteSpace': 'nowrap'}
                        ),
                        fac.AntdIcon(
                            id='optimization-tour-icon',
                            icon='pi-info',
                            style={"cursor": "pointer", 'paddingLeft': '10px'},
                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdTooltip(
                                    fac.AntdButton(
                                        'Compute Chromatograms',
                                        id='compute-chromatograms-btn',
                                        style={'textTransform': 'uppercase', "margin": "0 10px"},
                                    ),
                                    id='compute-chromatograms-btn-tooltip',
                                    title="Calculate chromatograms from the MS files and Targets.",
                                    placement="bottom"
                                ),
                                fac.AntdSelect(
                                    id='targets-select',
                                    options=[],
                                    mode="multiple",
                                    autoSpin=True,
                                    maxTagCount="responsive",
                                    style={"width": "450px"},
                                    locale="en-us",
                                )
                            ],
                            justify='space-between',
                            style={"margin": "0 40px 0 10px", 'width': '100%'},
                        ),
                    ],
                    align='center',
                ),

            ],
            style={'background': 'white', 'padding': '0px', 'lineHeight': '32px', 'height': '32px'}
        ),
        fac.AntdLayout(
            [
                fac.AntdSider(
                    [
                        fac.AntdButton(
                            id='optimization-sidebar-collapse',
                            type='text',
                            icon=fac.AntdIcon(
                                id='optimization-sidebar-collapse-icon',
                                icon='antd-right',  # Start with right arrow (collapsed)
                                style={'fontSize': '14px'}, ),
                            shape='default',
                            style={
                                'position': 'absolute',
                                'zIndex': 1,
                                # 'top': 0,
                                'right': -8,
                                'boxShadow': '2px 2px 5px 1px rgba(0,0,0,0.5)',
                                'background': 'white',
                            },

                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdFlex(
                                    [
                                        html.Div(
                                            [
                                                fac.AntdTitle(
                                                    'Sample Type',
                                                    level=5,
                                                    style={'margin': '0'}
                                                ),
                                                fac.AntdTooltip(
                                                    fac.AntdIcon(
                                                        icon='antd-question-circle',
                                                        style={'marginLeft': '5px', 'color': 'gray', 'fontSize': '14px'}
                                                    ),
                                                    title='All samples in this tree were chosen to optimize chromatogram parameters. A default number (50) of samples equally distributed accross sample types is selected for preview. Check boxes to include more samples in the preview.',
                                                    placement='right'
                                                )
                                            ],
                                            style={'display': 'flex', 'alignItems': 'center'}
                                        ),
                                        fac.AntdCompact(
                                            [
                                                fac.AntdTooltip(
                                                    fac.AntdIcon(
                                                        icon='pi-crosshair',
                                                        className='expand-icon',
                                                        id='mark-tree-action'
                                                    ),
                                                    title='Mark all Sample Types'
                                                ),
                                                fac.AntdTooltip(
                                                    fac.AntdIcon(
                                                        icon='antd-up',
                                                        className='expand-icon',
                                                        id='collapse-tree-action'
                                                    ),
                                                    title='Collapse Sample Type Tree'
                                                ),
                                                fac.AntdTooltip(
                                                    fac.AntdIcon(
                                                        icon='antd-down',
                                                        className='expand-icon',
                                                        id='expand-tree-action'
                                                    ),
                                                    title='Expand Sample Type Tree'
                                                ),
                                            ],
                                        )
                                    ],
                                    justify='space-between',
                                    align='center',
                                    style={'marginRight': 30, 'height': 32, 'overflow': 'hidden'}
                                ),
                                html.Div(
                                    fac.AntdSpin(
                                        [
                                            fac.AntdTree(
                                                id='sample-type-tree',
                                                treeData=[],
                                                multiple=True,
                                                checkable=True,
                                                defaultExpandAll=False,
                                                showIcon=True,
                                                style={'display': 'none'}
                                            ),
                                            fac.AntdEmpty(
                                                id='sample-type-tree-empty',
                                                description='No samples marked for optimization',
                                                locale='en-us',
                                                image='simple',
                                                styles={'root': {'height': '100%', 'alignContent': 'center'}}
                                            )
                                        ],
                                        style={'height': '100%'}
                                    ),
                                    style={
                                        'flex': '1',
                                        'overflow': 'auto',
                                        'minHeight': '0'
                                    },
                                    id='sample-selection'
                                ),

                                html.Div(
                                    [
                                        fac.AntdDivider(
                                            'Options',
                                            size='small'
                                        ),
                                        html.Div(
                                            [
                                                fac.AntdFormItem(
                                                    fac.AntdSelect(
                                                        id='chromatogram-preview-filter-ms-type',
                                                        options=['All', 'ms1', 'ms2'],
                                                        value='All',
                                                        placeholder='Select ms_type',
                                                        style={'width': '100%'},
                                                        allowClear=False,
                                                        locale="en-us",
                                                    ),
                                                    label='MS-Type:',
                                                    tooltip='Filter chromatograms by ms_type',
                                                    style={'marginBottom': '1rem'}
                                                ),
                                                fac.AntdFormItem(
                                                    fac.AntdSelect(
                                                        id='chromatogram-preview-filter-bookmark',
                                                        options=['All', 'Bookmarked', 'Unmarked'],
                                                        value='All',
                                                        placeholder='Select filter',
                                                        style={'width': '100%'},
                                                        allowClear=False,
                                                        locale="en-us",
                                                    ),
                                                    label='Selection:',
                                                    tooltip='Filter chromatograms by bookmark status',
                                                    style={'marginBottom': '1rem'}
                                                ),
                                                fac.AntdFormItem(
                                                    fac.AntdSelect(
                                                        id='chromatogram-preview-order',
                                                        options=[{'label': 'By Peak Label', 'value': 'peak_label'},
                                                                 {'label': 'By MZ-Mean', 'value': 'mz_mean'}],
                                                        value='mz_mean',
                                                        placeholder='Select filter',
                                                        style={'width': '100%'},
                                                        allowClear=False,
                                                        locale="en-us",
                                                    ),
                                                    label='Order by:',
                                                    tooltip='Ascended order chromatograms by peak label or mz mean',
                                                    style={'marginBottom': '1rem'}
                                                ),
                                                fac.AntdFormItem(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-preview-log-y',
                                                        checked=False
                                                    ),
                                                    label='Intensity Log Scale',
                                                    tooltip='Apply log scale to intensity axis',
                                                    style={'marginBottom': '0'}
                                                )
                                            ],
                                            style={'padding': 10}
                                        ),
                                        fac.AntdForm(
                                            [
                                                fac.AntdFormItem(
                                                    fac.AntdCompact(
                                                        [
                                                            fac.AntdInputNumber(
                                                                id='chromatogram-graph-width',
                                                                value=350,  # Matches defaultPageSize=9
                                                                defaultValue=350,
                                                                min=180,
                                                                max=1400
                                                            ),
                                                            fac.AntdInputNumber(
                                                                id='chromatogram-graph-height',
                                                                value=220,  # Matches defaultPageSize=9
                                                                defaultValue=220,
                                                                min=100,
                                                                max=700
                                                            ),
                                                        ],
                                                        style={'width': '160px'}
                                                    ),
                                                    label='WxH:',
                                                    tooltip='Set preview plot width and height'
                                                ),
                                                fac.AntdFormItem(
                                                fac.AntdTooltip(
                                                    fac.AntdButton(
                                                        # 'Apply',
                                                        id='chromatogram-graph-button',
                                                        icon=fac.AntdIcon(icon='pi-broom', style={'fontSize': 20}),
                                                        # type='primary'
                                                    ),
                                                    title='Update graph size and clean plots',
                                                    placement="bottom"
                                                ),
                                                    style={"marginInlineEnd": 0}
                                                )
                                            ],
                                            layout='inline',
                                            style={'padding': 10, 'justifyContent': 'center'}
                                        )
                                    ],
                                    style={'overflow': 'visible', 'flexShrink': 0, 'minHeight': '280px'},
                                    id='sidebar-options'
                                ),
                            ],
                            vertical=True,
                            justify='space-between',
                            style={'height': '100%'}
                        )
                    ],
                    id='optimization-sidebar',
                    collapsible=True,
                    collapsed=True,  # Start collapsed - managed by clientside callback
                    collapsedWidth=0,
                    width=300,
                    trigger=None,
                    style={'height': '100%'},
                    className="sidebar-mint"
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                fac.AntdSpin(
                                    [
                                        html.Div(
                                            [
                                                fac.AntdSpace(
                                                    [
                                                        fac.AntdCard(
                                                            id={'type': 'target-card-preview', 'index': i},
                                                            style={'cursor': 'pointer'},
                                                            styles={'header': {'display': 'none'},
                                                                    'body': {'padding': '5px'},
                                                                    'actions': {'height': 30}},
                                                            hoverable=True,
                                                            children=[
                                                                dcc.Graph(
                                                                    id={'type': 'graph', 'index': i},
                                                                    figure=go.Figure(
                                                                        layout=dict(
                                                                            xaxis_title="Retention Time [s]",
                                                                            yaxis_title="Intensity",
                                                                            showlegend=False,
                                                                            margin=dict(l=40, r=5, t=30, b=30),
                                                                            hovermode=False,
                                                                            dragmode=False,
                                                                        )
                                                                    ),
                                                                    style={
                                                                        'height': '220px', 'width': '350px',
                                                                        'margin': '0 0 14px 0',
                                                                    },
                                                                    config={
                                                                        'displayModeBar': False,
                                                                        'staticPlot': True,
                                                                        'doubleClick': False,
                                                                        'showTips': False,
                                                                        'responsive': False
                                                                    },
                                                                ),
                                                                fac.AntdRate(
                                                                    id={'type': 'bookmark-target-card', 'index': i},
                                                                    count=1,
                                                                    defaultValue=0,
                                                                    value=0,
                                                                    allowHalf=False,
                                                                    tooltips=['Bookmark this target'],
                                                                    style={'position': 'absolute', 'top': '8px',
                                                                           'right': '8px', 'zIndex': 20},
                                                                ),
                                                                fac.AntdTooltip(
                                                                    fac.AntdButton(
                                                                        icon=fac.AntdIcon(icon='antd-delete'),
                                                                        type='text',
                                                                        size='small',
                                                                        id={'type': 'delete-target-card', 'index': i},
                                                                        style={
                                                                            'padding': '4px',
                                                                            'minWidth': '24px',
                                                                            'height': '24px',
                                                                            'borderRadius': '50%',
                                                                            'background': 'rgba(0, 0, 0, 0.5)',
                                                                            'color': 'white',
                                                                            'position': 'absolute',
                                                                            'bottom': '8px',
                                                                            'right': '8px',
                                                                            'zIndex': 20,
                                                                            'opacity': '0.1',
                                                                            'transition': 'opacity 0.3s ease'
                                                                        },
                                                                        className='peak-action-button',
                                                                    ),
                                                                    title='Delete target',
                                                                    color='red',
                                                                    placement='bottom',
                                                                ),
                                                            ],
                                                            **{'data-target': None},
                                                            className='is-hidden'
                                                        ) for i in range(MAX_NUM_CARDS)
                                                    ],
                                                    id='chromatogram-preview',
                                                    wrap=True,
                                                    align='center',
                                                    style={'height': 'calc(100vh - 64px - 4rem)', 'overflowY': 'auto',
                                                           'width': '100%', 'padding': '0 10px 10px 10px',
                                                           'justifyContent': 'center'}
                                                ),
                                                fac.AntdPagination(
                                                    id='chromatogram-preview-pagination',
                                                    defaultPageSize=9,  # Reduced from 20 for faster load with large MS2 data
                                                    showSizeChanger=True,
                                                    pageSizeOptions=[4, 9, 20, 50],
                                                    locale='en-us',
                                                    align='center',
                                                    showTotalSuffix='targets',
                                                ),
                                                html.Div(
                                                    id='chromatograms-dummy-output',
                                                    style={'display': 'none'}
                                                )
                                            ])
                                    ],
                                    text='Loading plots...',
                                    id='chromatogram-preview-spin',
                                ),
                            ],
                            id='chromatogram-preview-container',
                            style={'display': 'none'}  # Hidden by default, shown when chromatograms exist
                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdEmpty(
                                    description=fac.AntdFlex(
                                        [
                                            fac.AntdText('No Chromatograms to preview', strong=True, style={'fontSize': '16px'}),
                                            fac.AntdText('Click "Compute Chromatograms" to generate the chromatograms', type='secondary'),
                                        ],
                                        vertical=True,
                                        align='center',
                                        gap='small',
                                    ),
                                    locale='en-us',
                                ),
                                fac.AntdTooltip(
                                    fac.AntdButton(
                                        'Compute Chromatograms',
                                        id='compute-chromatograms-empty-btn',
                                        size='large',
                                        style={'marginTop': '16px', 'textTransform': 'uppercase'},
                                    ),
                                    id='compute-chromatograms-empty-btn-tooltip',
                                    title="Calculate chromatograms from the MS files and Targets.",
                                    placement="bottom"
                                ),
                            ],
                            id='chromatogram-preview-empty',
                            vertical=True,
                            align='center',
                            style={"display": "flex", 'marginTop': '100px'}  # Shown by default when no chromatograms
                        ),
                    ],
                    className='ant-layout-content css-1v28nim',
                    style={'background': 'white',
                           # 'height': 'calc(100vh - 64px - 4rem)', 'overflowY': 'auto'
                           }
                ),
            ],
            style={'padding': '1rem 0', 'background': 'white'},
        ),
        html.Div(id="optimization-notifications-container"),
        fac.AntdModal(
            [
                fac.AntdFlex(
                    [
                        fac.AntdDivider('Recompute Chromatograms'),
                        fac.AntdFlex(
                            [
                                fac.AntdFormItem(
                                    fac.AntdCheckbox(
                                        id='chromatograms-recompute-ms1',
                                        label='Recompute MS1'
                                    ),

                                ),
                                fac.AntdFormItem(
                                    fac.AntdCheckbox(
                                        id='chromatograms-recompute-ms2',
                                        label='Recompute MS2'
                                    ),
                                ),
                            ],
                            gap='large'
                        ),
                        fac.AntdDivider('Configuration'),
                        fac.AntdFlex(
                            [
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='chromatogram-compute-cpu',
                                        value=cpu_count() // 2,
                                        min=1,
                                        max=cpu_count() - 2,
                                    ),
                                    label='CPU:',
                                    hasFeedback=True,
                                    help=f"Selected {cpu_count() // 2} / {cpu_count()} cpus",
                                    id='chromatogram-compute-cpu-item'
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='chromatogram-compute-ram',
                                        value=round(psutil.virtual_memory().available * 0.5 / (1024 ** 3), 1),
                                        min=1,
                                        precision=1,
                                        step=0.1,
                                        suffix='GB'
                                    ),
                                    label='RAM:',
                                    hasFeedback=True,
                                    id='chromatogram-compute-ram-item',
                                    help=f"Selected "
                                         f"{round(psutil.virtual_memory().available * 0.5 / (1024 ** 3), 1)}GB / "
                                         f"{round(psutil.virtual_memory().available / (1024 ** 3), 1)}GB available RAM"
                                ),
                                fac.AntdFormItem(
                                    fac.AntdInputNumber(
                                        id='chromatogram-compute-batch-size',
                                        placeholder='Calculating...',
                                        min=50,
                                        step=50,
                                    ),
                                    label='Batch Size:',
                                    tooltip='Optimal pairs per batch based on RAM/CPU. '
                                            'Higher values = faster but more memory.',
                                ),
                            ],
                            gap='large'
                        ),

                        fac.AntdDivider(),
                        fac.AntdAlert(
                            message='There are no targets selected. The chromatograms will be computed for all targets.',
                            type='info',
                            showIcon=True,
                            id='chromatogram-targets-info',
                        ),
                        fac.AntdAlert(
                            message='There are already computed chromatograms',
                            type='warning',
                            showIcon=True,
                            id='chromatogram-warning',
                            style={'display': 'none'},
                        ),
                    ],
                    id='chromatogram-compute-options-container',
                    vertical=True
                ),

                html.Div(
                    [
                        html.H4("Generating Chromatograms..."),
                        fac.AntdText(
                            id='chromatogram-processing-stage',
                            style={'marginBottom': '0.5rem'},
                        ),
                        fac.AntdProgress(
                            id='chromatogram-processing-progress',
                            percent=0,
                        ),
                        fac.AntdText(
                            id='chromatogram-processing-detail',
                            type='secondary',
                            style={
                                'marginTop': '0.5rem',
                                'marginBottom': '0.75rem',
                            },
                        ),
                        fac.AntdButton(
                            'Cancel',
                            id='cancel-chromatogram-processing',
                            style={
                                'alignText': 'center',
                                'marginTop': '0.25rem',
                            },
                        ),
                    ],
                    id='chromatogram-processing-progress-container',
                    style={'display': 'none'},
                ),
            ],
            id='compute-chromatogram-modal',
            title='Compute chromatograms',
            width=900,
            renderFooter=True,

            locale='en-us',
            confirmAutoSpin=True,
            loadingOkText='Generating Chromatograms...',
            okClickClose=False,
            closable=False,
            maskClosable=False,
            destroyOnClose=False,
            okText="Generate",
            centered=True,
            styles={'body': {'height': "50vh"}},
        ),
        fac.AntdModal(
            id="chromatogram-view-modal",
            width="100vw",
            centered=True,
            destroyOnClose=True,
            closable=True,
            maskClosable=False,
            children=[
                fac.AntdLayout(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id='chromatogram-view-plot',
                                    figure=go.Figure(
                                        layout=dict(
                                            xaxis_title="Retention Time [s]",
                                            yaxis_title="Intensity",
                                            showlegend=True,
                                            margin=dict(l=40, r=10, t=50, b=80),
                                        )
                                    ),
                                    config={**PLOTLY_HIGH_RES_CONFIG, 'edits': {'shapePosition': True}},
                                    style={'width': '100%', 'height': '600px'}
                                ),
                                # Invisible placeholder for spinner callbacks (keeps callbacks valid)
                                html.Div(id='chromatogram-view-spin', style={'display': 'none'}),
                            ],
                            id='chromatogram-view-container',
                            className='ant-layout-content css-1v28nim',
                            style={
                                # 'position': 'relative',
                                'overflowX': 'hidden',
                                'background': 'white',
                                'alignContent': 'center'
                            },
                        ),
                        fac.AntdSider(
                            [
                                fac.AntdTitle(
                                    'Options',
                                    level=4,
                                    style={'margin': '0px', 'marginBottom': '8px'}
                                ),
                                fac.AntdSpace(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span('Megatrace:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='Merge traces to improve performance. Color by sample type only.'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-megatrace',
                                                        checked=True,
                                                        checkedChildren='On',
                                                        unCheckedChildren='Off',
                                                        style={'width': '60px'}
                                                    ),
                                                    style={
                                                        'width': '110px',
                                                        'display': 'flex',
                                                        'justifyContent': 'flex-start'
                                                    }
                                                ),
                                            ],
                                            style={'display': 'flex', 'alignItems': 'center'}
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span('Intensity Scale:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='Linear vs Logarithmic scale'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-log-y',
                                                        checked=False,
                                                        checkedChildren='Log',
                                                        unCheckedChildren='Lin',
                                                        style={'width': '60px'}
                                                    ),
                                                    style={
                                                        'width': '110px',
                                                        'display': 'flex',
                                                        'justifyContent': 'flex-start'
                                                    }
                                                ),
                                            ],
                                            style={'display': 'flex', 'alignItems': 'center'}
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span('Legend Behavior:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='Single vs Group toggle'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-groupclick',
                                                        checked=False,
                                                        checkedChildren='Grp',
                                                        unCheckedChildren='Sng',
                                                        style={'width': '60px'}
                                                    ),
                                                    style={
                                                        'width': '110px',
                                                        'display': 'flex',
                                                        'justifyContent': 'flex-start'
                                                    }
                                                ),
                                            ],
                                            style={'display': 'flex', 'alignItems': 'center'}
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span('Full Range:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            id='chromatogram-view-full-range-tooltip',
                                                            title='Show entire chromatogram (slower) vs 30s window'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-full-range',
                                                        checked=False,
                                                        checkedChildren='All',
                                                        unCheckedChildren='30s',
                                                        style={'width': '60px'}
                                                    ),
                                                    style={
                                                        'width': '110px',
                                                        'display': 'flex',
                                                        'justifyContent': 'flex-start'
                                                    }
                                                ),
                                            ],
                                            style={'display': 'flex', 'alignItems': 'center'}
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span('Edit RT-span:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='Unlock to edit RT range'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-lock-range',
                                                        checked=False,
                                                        checkedChildren='Lock',
                                                        unCheckedChildren='Edit',
                                                        style={'width': '60px'}
                                                    ),
                                                    style={
                                                        'width': '110px',
                                                        'display': 'flex',
                                                        'justifyContent': 'flex-start'
                                                    }
                                                ),
                                            ],
                                            style={'display': 'flex', 'alignItems': 'center'}
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span('RT Alignment:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='Align chromatograms by peak apex within the RT span'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-rt-align',
                                                        checked=False,
                                                        checkedChildren='On',
                                                        unCheckedChildren='Off',
                                                        style={'width': '60px'}
                                                    ),
                                                    style={
                                                        'width': '110px',
                                                        'display': 'flex',
                                                        'justifyContent': 'flex-start'
                                                    }
                                                ),
                                            ],
                                            style={'display': 'flex', 'alignItems': 'center'}
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span('SAVGOL Smoothing:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='Apply Savitzky-Golay smoothing to intensities'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                html.Div(
                                                    fac.AntdSwitch(
                                                        id='chromatogram-view-savgol',
                                                        checked=True,
                                                        checkedChildren='On',
                                                        unCheckedChildren='Off',
                                                        style={'width': '60px'}
                                                    ),
                                                    style={
                                                        'width': '110px',
                                                        'display': 'flex',
                                                        'justifyContent': 'flex-start'
                                                    }
                                                ),
                                            ],
                                            style={'display': 'flex', 'alignItems': 'center'}
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span('Notes:'),
                                                        fac.AntdTooltip(
                                                            fac.AntdIcon(
                                                                icon='antd-question-circle',
                                                                style={'marginLeft': '5px', 'color': 'gray'}
                                                            ),
                                                            title='User notes for target'
                                                        )
                                                    ],
                                                    style={
                                                        'display': 'flex',
                                                        'alignItems': 'center',
                                                        'width': '170px',
                                                        'paddingRight': '8px'
                                                    }
                                                ),
                                                fac.AntdInput(
                                                    id='target-note',
                                                    allowClear=True,
                                                    mode='text-area',
                                                    autoSize={'minRows': 6, 'maxRows': 12},
                                                    style={'width': '230px'},
                                                    placeholder='Add notes for this target'
                                                ),
                                            ],
                                            style={
                                                'display': 'flex',
                                                'flexDirection': 'column',
                                                'alignItems': 'flex-start',
                                                'width': '100%',
                                                'marginTop': '6px'
                                            }
                                        ),
                                    ],
                                    direction='vertical',
                                    size='small',
                                    style={'alignItems': 'flex-start'}
                                )
                            ],
                            collapsible=False,
                            theme='light',
                            width=250,
                            style={'marginLeft': 20,
                                   'background': 'white'}
                        )
                    ],
                    style={'background': 'white'}
                ),
                fac.AntdDivider(size='small'),
                fac.AntdFlex(
                    [
                        fac.AntdSpace(
                            [
                                fac.AntdAlert(
                                    message="RT values changed. Will save automatically on navigation or close. Reset to restore original.",
                                    type="info",
                                    showIcon=True,
                                ),
                                fac.AntdButton(
                                    "Reset",
                                    id="reset-btn",
                                ),
                                # Hidden placeholder for save-btn to avoid breaking callbacks
                                html.Div(id="save-btn", style={"display": "none"}),
                            ],
                            align='center',
                            size=60,
                            id='action-buttons-container',
                            style={
                                "visibility": "hidden",
                                'opacity': '0',
                                'transition': 'opacity 0.3s ease-in-out',
                            }
                        ),
                        fac.AntdSpace(
                            [
                                # Navigation buttons group (compact spacing)
                                fac.AntdSpace(
                                    [
                                        fac.AntdButton(
                                            icon=fac.AntdIcon(icon='antd-left'),
                                            id="target-nav-prev",
                                            disabled=True,
                                        ),
                                        fac.AntdText(
                                            "1 / 1",
                                            id="target-nav-counter",
                                            style={'minWidth': '30px', 'textAlign': 'center'}
                                        ),
                                        fac.AntdButton(
                                            icon=fac.AntdIcon(icon='antd-right'),
                                            id="target-nav-next",
                                            disabled=True,
                                        ),
                                    ],
                                    size=20,
                                ),
                                fac.AntdButton(
                                    "Bookmark",
                                    id="bookmark-target-modal-btn",
                                    icon=fac.AntdIcon(icon="antd-star"),
                                    type="default",
                                ),
                                fac.AntdButton(
                                    "Delete",
                                    icon=fac.AntdIcon(icon='antd-delete'),
                                    id="delete-target-from-modal",
                                    danger=True,
                                    type="default",
                                ),
                            ],
                            size=20,
                            addSplitLine=False,
                            style={
                                'marginLeft': '50px',
                            },
                        ),

                    ],
                    justify='space-between',
                    align='center',
                ),
            ]
        ),
        fac.AntdModal(
            id="delete-targets-modal",
            title="Delete target",
            width=400,
            renderFooter=True,
            okText="Delete",
            okButtonProps={"danger": True},
            cancelText="Cancel",
            locale='en-us',
        ),
        fac.AntdModal(
            "Are you sure you want to close this window without saving your changes?",
            id="confirm-unsave-modal",
            title="Confirm close without saving",
            width=400,
            okButtonProps={'danger': True},
            renderFooter=True,
            locale='en-us'
        ),

        dcc.Store(id='slider-data'),
        dcc.Store(id='slider-reference-data'),
        dcc.Store(id='rt-alignment-data'),  # Stores RT alignment info for saving to notes
        dcc.Store(id='target-preview-clicked'),

        dcc.Store(id='chromatograms', data=True),
        dcc.Store(id='drop-chromatogram'),
        dcc.Store(id="delete-target-clicked"),
        dcc.Store(id='chromatogram-view-plot-max'),
        dcc.Store(id='chromatogram-view-plot-points'),
        dcc.Store(id='update-chromatograms', data=False),
        dcc.Store(id='target-nav-store', data={'targets': [], 'current_index': 0}),  # For Prev/Next navigation
        dcc.Store(id='pending-nav-direction', data=None),  # Stores pending navigation when unsaved changes exist
        fac.AntdModal(
            "You have unsaved changes. Are you sure you want to navigate to another target?",
            id="confirm-nav-modal",
            title="Unsaved changes",
            width=400,
            okButtonProps={'danger': True},
            okText="Discard & Navigate",
            renderFooter=True,
            locale='en-us'
        ),
        dcc.Store(id='keyboard-nav-trigger', data={'key': None, 'timestamp': 0}),
        dcc.Store(id='spinner-start-time', data=None),  # Track when spinner started
        dcc.Store(id='chromatogram-container-width', data=None),  # Container width for auto-sizing
        dcc.Interval(id='container-width-interval', interval=300, n_intervals=0, max_intervals=1),  # One-time trigger
        dcc.Interval(id='spinner-timeout-interval', interval=1000, disabled=True),  # Check every second
        # Clientside keyboard listener for arrow key navigation
        html.Div(
            id='keyboard-listener',
            style={'position': 'absolute', 'width': 0, 'height': 0, 'overflow': 'hidden'},
            tabIndex=-1
        ),
        # Tour for empty state (no chromatograms) - simplified
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'This tutorial shows how to compute and view chromatograms.',
                },
                {
                    'title': 'Compute chromatograms',
                    'description': 'Click to extract chromatograms for your files and targets.',
                    'targetSelector': '#compute-chromatograms-btn'
                },
            ],
            id='optimization-tour-empty',
            open=False,
            current=0,
        ),
        # Tour for populated state (chromatograms exist) - full tour
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Welcome',
                    'description': 'This tutorial shows how to optimize your targets.',
                },
                {
                    'title': 'Recompute',
                    'description': 'Click to re-compute chromatograms if needed.',
                    'targetSelector': '#compute-chromatograms-btn'
                },
                {
                    'title': 'Select samples',
                    'description': 'Choose which samples to display in the preview cards.',
                    'targetSelector': '#sample-selection'
                },
                {
                    'title': 'Select targets',
                    'description': 'Filter to show specific targets.',
                    'targetSelector': '#targets-select'
                },
                {
                    'title': 'Review cards',
                    'description': 'Inspect chromatograms, bookmark, or delete targets.',
                    'targetSelector': '#chromatogram-preview-container'
                },
                {
                    'title': 'Tune options',
                    'description': 'Adjust plot settings like MS type, sorting, and log scale.',
                    'targetSelector': '#sidebar-options'
                },
            ],
            id='optimization-tour-full',
            open=False,
            current=0,
        ),
        fac.AntdTour(
            locale='en-us',
            steps=[
                {
                    'title': 'Need help?',
                    'description': 'Click the info icon to open a quick tour of Optimization.',
                    'targetSelector': '#optimization-tour-icon',
                },
            ],
            mask=False,
            placement='rightTop',
            open=False,
            current=0,
            id='optimization-tour-hint',
            className='targets-tour-hint',
            style={
                'background': '#ffffff',
                'border': '0.5px solid #1677ff',
                'boxShadow': '0 6px 16px rgba(0,0,0,0.15), 0 0 0 1px rgba(22,119,255,0.2)',
                'opacity': 1,
            },
        ),
        dcc.Store(id='optimization-tour-hint-store', data={'open': False}, storage_type='local'),
    ],
    style={'height': '100%'},
)


def layout():
    return _layout


def _update_sample_type_tree(section_context, mark_action, expand_action, collapse_action, selection_ms_type, wdir, prop_id, workspace_status=None):
    if not section_context or section_context.get('page') != 'Optimization':
        raise PreventUpdate
    if not wdir:
        return [], [], [], {'display': 'none'}, {'display': 'block'}

    # INSTANT EARLY CHECK: Skip loading tree if no chromatograms exist (using cached status)
    if workspace_status and workspace_status.get('chromatograms_count', 0) == 0:
        return [], [], [], {'display': 'none'}, {'display': 'block'}



    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.error("update_sample_type_tree: Could not connect to database")
            return [], [], [], {'display': 'none'}, {'display': 'block'}
        
        df = conn.execute("""
                          SELECT sample_type,
                                 list({'title': label, 'key': label}) as children,
                                 list(label)                          as checked_keys
                          FROM samples
                          WHERE use_for_optimization = TRUE
                            AND CASE
                                    WHEN ? = 'ms1' THEN ms_type = 'ms1'
                                    WHEN ? = 'ms2' THEN ms_type = 'ms2'
                                    ELSE TRUE -- 'all' case
                              END
                          GROUP BY sample_type
                          ORDER BY sample_type
                          """, [selection_ms_type, selection_ms_type]).pl()

        if df.is_empty():
            # Chromatograms exist but no samples marked - still expand sidebar to show empty message
            return [], [], [], {'display': 'none'}, {'display': 'block'}

        if prop_id == 'mark-tree-action':
            # logger.debug(f"{df['checked_keys'].to_list() = }")
            checked_keys = [v for value in df['checked_keys'].to_list() for v in value]
        elif prop_id in ['section-context', 'workspace-status']:
            # Initialize with proportional selection (max 50 samples) when section loads or after chromatograms are computed
            quotas, checked_keys = proportional_min1_selection(df, 'sample_type', 'checked_keys', 50, 12345)
        else:
            checked_keys = dash.no_update

        # Rebuild tree data when section loads, MS type filter changes, or workspace-status updates (e.g. after computing chromatograms)
        if prop_id in ['section-context', 'chromatogram-preview-filter-ms-type', 'workspace-status']:
            tree_data = [
                {
                    'title': row['sample_type'],
                    'key': row['sample_type'],
                    'children': row['children']
                }
                for row in df.to_dicts()
            ]
        else:
            tree_data = dash.no_update

        if prop_id == 'expand-tree-action':
            expanded_keys = df['sample_type'].to_list()
        elif prop_id == 'collapse-tree-action':
            expanded_keys = []
        else:
            expanded_keys = dash.no_update
    return tree_data, checked_keys, expanded_keys, {'display': 'flex'}, {'display': 'none'}


def _delete_target_logic(target, wdir):
    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.error(f"delete_target_logic: Could not connect to database for target '{target}'")
            return (fac.AntdNotification(
                        message="Database connection failed",
                        description="Could not connect to the database.",
                        type="error",
                        duration=4,
                        placement='bottom',
                        showProgress=True,
                        stack=True
                    ),
                    dash.no_update,
                    False,
                    False)
        try:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM chromatograms WHERE peak_label = ?", [target])
            conn.execute("DELETE FROM targets WHERE peak_label = ?", [target])
            conn.execute("DELETE FROM results WHERE peak_label = ?", [target])
            conn.execute("COMMIT")
            logger.info(f"Deleted target '{target}' and associated chromatograms/results.")
        except Exception as e:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to delete target '{target}'", exc_info=True)
            return (fac.AntdNotification(
                        message="Failed to delete target",
                        description=f"Error: {e}",
                        type="error",
                        duration=4,
                        placement='bottom',
                        showProgress=True,
                        stack=True
                    ),
                    dash.no_update,
                    False,
                    False)

    return (fac.AntdNotification(message=f"Chromatograms deleted for '{target}'",
                                    type="success",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    ),
            True,
            False,
            False)


def _bookmark_target_logic(bookmarks, targets, trigger_id, wdir):
    with duckdb_connection(wdir) as conn:
        if conn is None:
            logger.error(f"Failed to connect to database to bookmark target '{targets[trigger_id]}'")
            return fac.AntdNotification(
                message="Database connection failed",
                description="Could not update bookmark status.",
                type="error",
                duration=5,
                placement='bottom',
                showProgress=True,
                stack=True
            )
        conn.execute("UPDATE targets SET bookmark = ? WHERE peak_label = ?", [bool(bookmarks[trigger_id]),
                                                                                targets[trigger_id]])
    
    status = "bookmarked" if bookmarks[trigger_id] else "unbookmarked"
    logger.info(f"Target '{targets[trigger_id]}' was {status}.")

    return fac.AntdNotification(message=f"Target '{targets[trigger_id]}' {'bookmarked' if bookmarks[trigger_id] else 'unbookmarked'}",
                                duration=3,
                                placement='bottom',
                                type="success",
                                showProgress=True,
                                stack=True)


def _toggle_bookmark_logic(target_label, wdir):
    with duckdb_connection(wdir) as conn:
        # Check current state
        res = conn.execute("SELECT bookmark FROM targets WHERE peak_label = ?", [target_label]).fetchone()
        current_state = res[0] if res else False
        
        # Toggle
        new_state = not current_state
        conn.execute("UPDATE targets SET bookmark = ? WHERE peak_label = ?", [new_state, target_label])
        
        logger.info(f"Toggled bookmark for {target_label} to {new_state}")
        
        icon_color = "gold" if new_state else "gray"
        return fac.AntdIcon(icon="antd-star", style={"color": icon_color})


def _compute_chromatograms_logic(set_progress, recompute_ms1, recompute_ms2, n_cpus, ram, batch_size, wdir):
    def progress_adapter(percent, stage="", detail=""):
        if set_progress:
            set_progress((percent, stage or "", detail or ""))

    activate_workspace_logging(wdir)

    with duckdb_connection(wdir, n_cpus=n_cpus, ram=ram) as con:
        if con is None:
            logger.error("Could not connect to database for chromatogram computation.")
            return "Could not connect to database."
        start = time.perf_counter()
        logger.info("Starting chromatogram computation.")
        progress_adapter(0, "Chromatograms", "Preparing batches...")
        compute_chromatograms_in_batches(wdir, use_for_optimization=True, batch_size=batch_size,
                                            set_progress=progress_adapter, recompute_ms1=recompute_ms1,
                                            recompute_ms2=recompute_ms2, n_cpus=n_cpus, ram=ram)
        logger.info(f"Chromatograms computed in {time.perf_counter() - start:.2f} seconds")
        
        # Optimize RT spans for targets that had RT auto-adjusted
        # This uses adaptive peak detection to find optimal rt_min, rt_max based on actual data
        progress_adapter(95, "Chromatograms", "Optimizing RT spans...")
        try:
            updated_count = optimize_rt_spans_batch(con)
            logger.info(f"Optimized RT spans for {updated_count} auto-adjusted targets")
        except Exception as e:
            logger.warning(f"Could not optimize RT spans: {e}")
        
    return True, False


def _calc_y_range_numpy(data, x_left, x_right, is_log=False):
    """
    Calculate the Y-axis range for the given x-range using NumPy for performance.
    
    Args:
        data: List of trace dictionaries (from Plotly figure['data'])
        x_left: Left bound of X-axis
        x_right: Right bound of X-axis
        is_log: Whether the Y-axis is in log scale
        
    Returns:
        list: [y_min, y_max] or None if no valid data
    """
    ys_all = []
    if not data:
        return None
        
    for trace in data:
        # Data from Dash callbacks comes as lists (if from JSON)
        xs = trace.get('x')
        ys = trace.get('y')
        
        if xs is None or ys is None or len(xs) == 0:
            continue
            
        # Convert to numpy for speed
        try:
            xs = np.array(xs, dtype=np.float64)
            ys = np.array(ys, dtype=np.float64)
        except Exception:
             continue
        
        mask = (xs >= x_left) & (xs <= x_right)
        ys_filtered = ys[mask]
        
        # Filter out None/NaN/Inf
        valid_mask = np.isfinite(ys_filtered)
        ys_filtered = ys_filtered[valid_mask]
        
        if len(ys_filtered) > 0:
            ys_all.append(ys_filtered)
            
    if not ys_all:
        return None
        
    ys_concat = np.concatenate(ys_all)
    
    if len(ys_concat) == 0:
        return None

    if is_log:
        min_floor = 1.0001
        ys_pos = ys_concat[ys_concat > min_floor]
        if len(ys_pos) == 0:
            return [math.log10(min_floor), math.log10(min_floor * 1.05)]

        # Use median of the 10 smallest values > 1 to avoid a "wall" from a single low outlier.
        k = min(10, len(ys_pos))
        smallest = np.partition(ys_pos, k - 1)[:k]
        y_min = np.median(smallest)
        if y_min <= min_floor:
            y_min = min_floor

        y_max = np.max(ys_pos)
        return [math.log10(y_min), math.log10(y_max * 1.05)]

    y_min = np.min(ys_concat)
    y_max = np.max(ys_concat)
    y_min = 0 if y_min > 0 else y_min
    return [y_min, y_max * 1.05]


def callbacks(app, fsc, cache, cpu=None):
    app.clientside_callback(
        """(nClicks, status, collapsed) => {
            const ctx = window.dash_clientside.callback_context;
            const trigger = ctx.triggered && ctx.triggered.length ? ctx.triggered[0].prop_id : '';
            
            // Auto-state logic based on workspace status
            if (trigger.includes('workspace-status') || (!trigger && status)) {
                if (!status) return window.dash_clientside.no_update;
                const count = status.chromatograms_count || 0;
                
                // State 1: No Chroms -> Collapsed (True). State 2: Chroms -> Expanded (False).
                const targetCollapsed = (count === 0);
                
                // Only update if state actually changes? No, enforce it to ensure consistency.
                return [targetCollapsed, targetCollapsed ? 'antd-right' : 'antd-left'];
            }
            
            // Manual toggle logic
            if (trigger.includes('optimization-sidebar-collapse')) {
                 const newCollapsed = !collapsed;
                 return [newCollapsed, newCollapsed ? 'antd-right' : 'antd-left'];
            }
            
            return window.dash_clientside.no_update;
        }""",
        Output('optimization-sidebar', 'collapsed'),
        Output('optimization-sidebar-collapse-icon', 'icon'),
        
        Input('optimization-sidebar-collapse', 'nClicks'),
        Input('workspace-status', 'data'),
        State('optimization-sidebar', 'collapsed'),
        prevent_initial_call=False,
    )

    # Disable compute buttons when no targets or ms-files exist
    app.clientside_callback(
        """(status) => {
            if (!status) return [true, true, "Load MS-Files and Targets first", "Load MS-Files and Targets first"];
            const hasFiles = (status.ms_files_count || 0) > 0;
            const hasTargets = (status.targets_count || 0) > 0;
            const disabled = !(hasFiles && hasTargets);
            const tooltip = disabled ? "Load MS-Files and Targets first" : "Calculate chromatograms from the MS files and Targets.";
            return [disabled, disabled, tooltip, tooltip];
        }""",
        Output('compute-chromatograms-btn', 'disabled'),
        Output('compute-chromatograms-empty-btn', 'disabled'),
        Output('compute-chromatograms-btn-tooltip', 'title'),
        Output('compute-chromatograms-empty-btn-tooltip', 'title'),
        Input('workspace-status', 'data'),
        prevent_initial_call=False,
    )

    # Single server-side callback: Opens modal AND populates all values atomically
    # This matches the robust Run Mint pattern - no race conditions possible
    @app.callback(
        Output('compute-chromatogram-modal', 'visible', allow_duplicate=True),
        Output('chromatogram-warning', 'style'),
        Output('chromatogram-warning', 'message'),
        Output('chromatogram-targets-info', 'message'),
        Output('chromatogram-compute-ram', 'max'),
        Output('chromatogram-compute-cpu', 'value'),
        Output('chromatogram-compute-ram', 'value'),
        Output('chromatogram-compute-batch-size', 'value', allow_duplicate=True),
        Output('chromatogram-compute-ram-item', 'help', allow_duplicate=True),
        Output('chromatogram-compute-cpu-item', 'help', allow_duplicate=True),
        Output("chromatograms-recompute-ms1", "checked"),
        Output("chromatograms-recompute-ms2", "checked"),
        Output("chromatogram-processing-progress", "percent"),
        Output("chromatogram-processing-stage", "children"),
        Output("chromatogram-processing-detail", "children"),
        
        Input('compute-chromatograms-btn', 'nClicks'),
        Input('compute-chromatograms-empty-btn', 'nClicks'),
        State('wdir', 'data'),
        prevent_initial_call=True,
    )
    def open_compute_chromatograms_modal(nClicks, nClicks_empty, wdir):
        """Open modal and populate all values in one atomic operation."""
        if not nClicks and not nClicks_empty:
            raise PreventUpdate
        
        # Calculate system defaults
        n_cpus = cpu_count()
        default_cpus = max(1, n_cpus // 2)
        ram_avail = psutil.virtual_memory().available / (1024 ** 3)
        default_ram = round(min(float(default_cpus), ram_avail), 1)
        
        # Default values if DB unavailable
        chromatograms_count = 0
        chroms_ms1_count = 0
        chroms_ms2_count = 0
        selected_targets_count = 0
        optimization_samples_count = 0
        batch_size = 1000
        
        # Query DB for current counts
        if wdir:
            try:
                with duckdb_connection(wdir) as conn:
                    if conn is not None:
                        counts = conn.execute("""
                            SELECT 
                                (SELECT COUNT(*) FROM chromatograms) as chroms,
                                (SELECT COUNT(*) FROM chromatograms WHERE ms_type = 'ms1') as chroms_ms1,
                                (SELECT COUNT(*) FROM chromatograms WHERE ms_type = 'ms2') as chroms_ms2,
                                (SELECT COUNT(*) FROM targets WHERE peak_selection = TRUE) as selected_targets,
                                (SELECT COUNT(*) FROM samples WHERE use_for_optimization = TRUE) as opt_samples
                        """).fetchone()
                        if counts:
                            chromatograms_count = counts[0] or 0
                            chroms_ms1_count = counts[1] or 0
                            chroms_ms2_count = counts[2] or 0
                            selected_targets_count = counts[3] or 0
                            optimization_samples_count = counts[4] or 0
                            
                            # Calculate optimal batch size
                            batch_size = calculate_optimal_batch_size(
                                default_ram,
                                max(selected_targets_count * optimization_samples_count, 100000),
                                default_cpus
                            )
            except Exception as e:
                logger.warning(f"Could not query DB for modal defaults: {e}")
        
        # Calculate display values
        warning_style = {'display': 'flex'} if chromatograms_count > 0 else {'display': 'none'}
        warning_message = f"There are already computed {chromatograms_count} chromatograms" if chromatograms_count > 0 else ""
        info_message = f"Ready to compute chromatograms for {selected_targets_count} targets and {optimization_samples_count} samples."
        
        help_cpu = f"Selected {default_cpus} / {n_cpus} cpus"
        help_ram = f"Selected {default_ram}GB / {round(ram_avail, 1)}GB available RAM"
        
        recompute_ms1 = chroms_ms1_count > 0
        recompute_ms2 = chroms_ms2_count > 0
        
        return (
            True,  # visible
            warning_style,
            warning_message,
            info_message,
            round(ram_avail, 1),  # max RAM
            default_cpus,
            default_ram,
            batch_size,
            help_ram,
            help_cpu,
            recompute_ms1,
            recompute_ms2,
            0,  # progress percent
            "",  # progress stage
            ""   # progress detail
        )
    # Clientside callback to detect container width for smart auto-sizing
    app.clientside_callback(
        """(n_intervals, sidebar_collapsed) => {
            const container = document.getElementById('chromatogram-preview');
            if (container) {
                return Math.floor(container.getBoundingClientRect().width);
            }
            // Fallback estimate
            const sidebarWidth = sidebar_collapsed ? 0 : 300;
            return Math.floor(window.innerWidth - sidebarWidth - 350);
        }""",
        Output('chromatogram-container-width', 'data'),
        Input('container-width-interval', 'n_intervals'),
        Input('optimization-sidebar', 'collapsed'),
    )

    # Update workspace-status store when chromatograms are computed/deleted
    @app.callback(
        Output('workspace-status', 'data', allow_duplicate=True),
        Input('chromatograms', 'data'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def update_workspace_status_on_chromatograms(chromatograms_trigger, wdir):
        """Keep workspace-status in sync when chromatograms change."""
        if not wdir:
            raise PreventUpdate
        
        # Calculate system defaults (safe to do without DB)
        n_cpus = cpu_count()
        default_cpus = max(1, n_cpus // 2)
        ram_avail = psutil.virtual_memory().available / (1024 ** 3)
        default_ram = round(min(float(default_cpus), ram_avail), 1)
        
        workspace_status = {
            'ms_files_count': 0,
            'targets_count': 0,
            'chromatograms_count': 0,
            'selected_targets_count': 0,
            'optimization_samples_count': 0,
            'results_count': 0,
            'n_cpus': n_cpus,
            'default_cpus': default_cpus,
            'ram_avail': round(ram_avail, 1),
            'default_ram': default_ram,
            'batch_size': 1000  # Safe default
        }
        
        try:
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
                        # Update with DB-derived counts, preserving the defaults set above
                        n_cpus = cpu_count()
                        default_cpus = max(1, n_cpus // 2)
                        ram_avail = psutil.virtual_memory().available / (1024 ** 3)
                        default_ram = round(min(float(default_cpus), ram_avail), 1)

                        batch_size = calculate_optimal_batch_size(
                            default_ram,
                            max((counts[1] or 0) * (counts[6] or 0), 100000),
                            default_cpus
                        )

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
                            'default_ram': default_ram,
                            'batch_size': batch_size
                        }
        except Exception as e:
            logger.warning(f"Could not update workspace status from DB: {e}")
            # workspace_status already contains safe defaults from lines 1561-1573
        
        logger.info(f"workspace-status updated (chromatograms changed): {workspace_status}")
        return workspace_status

    @app.callback(
        Output('optimization-tour-empty', 'current'),
        Output('optimization-tour-empty', 'open'),
        Output('optimization-tour-full', 'current'),
        Output('optimization-tour-full', 'open'),
        Input('optimization-tour-icon', 'nClicks'),
        State('workspace-status', 'data'),
        prevent_initial_call=True,
    )
    def optimization_tour_open(n_clicks, workspace_status):
        has_chromatograms = workspace_status and (workspace_status.get('chromatograms_count', 0) or 0) > 0
        if has_chromatograms:
            # Open full tour, keep empty tour closed
            return 0, False, 0, True
        else:
            # Open empty tour, keep full tour closed
            return 0, True, 0, False

    @app.callback(
        Output('optimization-tour-hint', 'open'),
        Output('optimization-tour-hint', 'current'),
        Input('optimization-tour-hint-store', 'data'),
    )
    def optimization_hint_sync(store_data):
        if not store_data:
            logger.debug("optimization_hint_sync: No store data, preventing update")
            raise PreventUpdate
        return store_data.get('open', True), 0

    @app.callback(
        Output('optimization-tour-hint-store', 'data'),
        Input('optimization-tour-hint', 'closeCounts'),
        Input('optimization-tour-icon', 'nClicks'),
        State('optimization-tour-hint-store', 'data'),
        prevent_initial_call=True,
    )
    def optimization_hide_hint(close_counts, n_clicks, store_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            logger.debug("optimization_hide_hint: No callback trigger, preventing update")
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'optimization-tour-icon':
            return {'open': False}

        if close_counts:
            return {'open': False}

        return store_data or {'open': True}

    @app.callback(
        Output("optimization-notifications-container", "children"),
        Input('section-context', 'data'),
        Input("wdir", "data"),
    )
    def warn_missing_workspace(section_context, wdir):
        if not section_context or section_context.get('page') != 'Optimization':
            return dash.no_update
        if not wdir:
            logger.debug("warn_missing_workspace: No workspace directory set, preventing update")
            raise PreventUpdate
        if wdir:
            return []
        return fac.AntdNotification(
            message="Activate a workspace",
            description="Please select or create a workspace before using Optimization.",
            type="warning",
            duration=4,
            placement='bottom',
            showProgress=True,
            stack=True,
        )

    ############# TREE BEGIN #####################################
    @app.callback(
        Output('sample-type-tree', 'treeData'),
        Output('sample-type-tree', 'checkedKeys'),
        Output('sample-type-tree', 'expandedKeys'),
        Output('sample-type-tree', 'style'),
        Output('sample-type-tree-empty', 'style'),

        Input('section-context', 'data'),
        Input('mark-tree-action', 'nClicks'),
        Input('expand-tree-action', 'nClicks'),
        Input('collapse-tree-action', 'nClicks'),
        Input('chromatogram-preview-filter-ms-type', 'value'),
        Input('workspace-status', 'data'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def update_sample_type_tree(section_context, mark_action, expand_action, collapse_action, selection_ms_type, workspace_status, wdir):
        ctx = dash.callback_context
        # Handle cases where ctx might be empty during tests if not mocked
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        return _update_sample_type_tree(section_context, mark_action, expand_action, collapse_action, selection_ms_type, wdir, prop_id, workspace_status)

    ############# TREE END #######################################

    ############# GRAPH OPTIONS BEGIN #####################################
    @app.callback(
        Output({'type': 'graph', 'index': ALL}, 'style'),
        Output('chromatogram-graph-width', 'value'),
        Output('chromatogram-graph-height', 'value'),
        Input('chromatogram-graph-button', 'nClicks'),
        Input('chromatogram-preview-pagination', 'pageSize'),
        Input('chromatogram-container-width', 'data'),
        State('chromatogram-graph-width', 'value'),
        State('chromatogram-graph-height', 'value'),
        prevent_initial_call=True
    )
    def set_chromatogram_graph_size(nClicks, page_size, container_width, width, height):
        """
        Auto-tune preview plot size based on container width and cards per page.
        Calculates optimal dimensions to achieve exact grid layout.
        Falls back to hard-coded defaults if container width unavailable.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            logger.debug("set_chromatogram_graph_size: No callback trigger, preventing update")
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        def calculate_optimal_size(ps, cont_width):
            """
            Calculate exact plot width for desired grid layout.
            Width is auto-calculated based on container; height uses fixed values per page size.
            """
            # Determine columns and fixed height based on page size
            if ps <= 4:
                cols = 2
                fixed_height = 350
            elif ps <= 9:
                cols = 3
                fixed_height = 220
            elif ps <= 20:
                cols = 5
                fixed_height = 180
            else:
                cols = 7
                fixed_height = 180

            # Default fallback widths (for when container width is not yet available)
            default_widths = {4: 500, 9: 350, 20: 250, 50: 250}
            
            if not cont_width or cont_width < 400:
                # Use fallback defaults
                return default_widths.get(ps, DEFAULT_GRAPH_WIDTH), fixed_height

            # Calculate exact width to fit exactly 'cols' columns
            gap = 12  # Gap between cards from AntdSpace
            card_extras = 22  # Card padding, border, margins
            
            # Solve for plot_width
            available = cont_width - (cols - 1) * gap - 40  # 40 for container padding
            plot_width = (available // cols) - card_extras
            
            # Safety bounds
            plot_width = max(200, min(600, plot_width))
            
            return int(plot_width), fixed_height

        width = width or DEFAULT_GRAPH_WIDTH
        height = height or DEFAULT_GRAPH_HEIGHT

        if trigger in ('chromatogram-preview-pagination', 'chromatogram-container-width'):
            width, height = calculate_optimal_size(page_size or 9, container_width)
        elif trigger != 'chromatogram-graph-button':
            logger.debug("set_chromatogram_graph_size: Update not triggered by valid input, preventing update")
            raise PreventUpdate

        return ([{
            'width': width,
            'height': height,
            'margin': '0px',
        } for _ in range(MAX_NUM_CARDS)],
                width,
                height)

    ############# GRAPH OPTIONS END #######################################

    @app.callback(
        Output({'type': 'graph', 'index': ALL}, 'figure', allow_duplicate=True),
        Input('chromatogram-preview-log-y', 'checked'),
        State({'type': 'graph', 'index': ALL}, 'figure'),
        prevent_initial_call=True
    )
    def update_preview_log_scale(log_scale, figures):
        if not figures:
            raise PreventUpdate
        
        t1 = time.perf_counter()
        updated_figures = []
        for fig in figures:
            if not fig:
                updated_figures.append(fig)
                continue
            
            new_fig = Patch()
            
            # Update Y-axis type
            new_fig['layout']['yaxis']['type'] = 'log' if log_scale else 'linear'
            new_fig['layout']['yaxis']['nticks'] = 3
            new_fig['layout']['yaxis']['dtick'] = None  # Ensure auto-ticking takes over
            
            # Recalculate range using existing data in the figure
            # Figure data structure: fig['data'] is a list of traces
            # Each trace has 'x' and 'y' arrays
            # We already have the optimized function available in scope? No, it's defined at module level now.
            
            # Try to get x-range from layout to be consistent, or just use full data if zoomed out
            x_range = fig.get('layout', {}).get('xaxis', {}).get('range')
            if x_range:
                x_min, x_max = x_range
            else:
                 # If no range, find global min/max from data? 
                 # Usually previews are full range. Let's assume full range of data if not set.
                 # But _calc_y_range_numpy needs x bounds.
                 # Let's use a very wide range if not specified, or just parse trace data.
                 # Actually, for preview, the x-axis is fixed to [rt_min, rt_max] usually.
                 # But simpler: scan all data in traces.
                 x_min, x_max = -float('inf'), float('inf')

            y_range = _calc_y_range_numpy(fig.get('data', []), x_min, x_max, is_log=log_scale)
            
            if y_range:
                new_fig['layout']['yaxis']['range'] = y_range
                new_fig['layout']['yaxis']['autorange'] = False
            else:
                new_fig['layout']['yaxis']['autorange'] = True

            updated_figures.append(new_fig)
            
        logger.debug(f"Log scale updated in {time.perf_counter() - t1:.4f}s")
        return updated_figures

    ############# PREVIEW BEGIN #####################################
    @app.callback(
        Output({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        Output({'type': 'graph', 'index': ALL}, 'figure'),
        Output({'type': 'bookmark-target-card', 'index': ALL}, 'value'),
        Output('chromatogram-preview-pagination', 'total'),
        Output('chromatogram-preview-pagination', 'current', allow_duplicate=True),
        Output('chromatograms-dummy-output', 'children'),
        Output('targets-select', 'options'),

        Input('chromatograms', 'data'),
        Input('chromatogram-preview-pagination', 'current'),
        Input('chromatogram-preview-pagination', 'pageSize'),
        Input('sample-type-tree', 'checkedKeys'),
        Input('chromatogram-preview-filter-bookmark', 'value'),
        Input('chromatogram-preview-filter-ms-type', 'value'),
        Input('chromatogram-preview-order', 'value'),
        Input('drop-chromatogram', 'data'),
        Input('targets-select', 'value'),
        State('chromatogram-preview-log-y', 'checked'),
        State('chromatograms-dummy-output', 'children'),
        State('wdir', 'data'),
        State('workspace-status', 'data'),
        prevent_initial_call=True
    )
    def chromatograms_preview(chromatograms, current_page, page_size, checkedkeys, selection_bookmark,
                              selection_ms_type, targets_order, dropped_target, selected_targets,
                              log_scale, preview_y_range, wdir, workspace_status):

        ctx = dash.callback_context
        if 'targets-select' in ctx.triggered[0]['prop_id'] and selected_targets:
            current_page = 1
        if not wdir:
            logger.debug("chromatograms_preview: No workspace directory, preventing update")
            raise PreventUpdate

        # INSTANT EARLY CHECK: Use cached workspace-status to skip DB entirely when no chromatograms
        # BUT: Skip this check if chromatograms just updated (workspace_status might be stale due to race condition)
        triggered_prop = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
        chroms_just_updated = 'chromatograms.data' in triggered_prop
        
        if not chroms_just_updated and workspace_status and workspace_status.get('chromatograms_count', 0) == 0:
            logger.debug("chromatograms_preview: No chromatograms (from workspace-status cache), returning empty")
            return (
                [None] * MAX_NUM_CARDS,  # data-target
                [dash.no_update] * MAX_NUM_CARDS,  # figures
                [0] * MAX_NUM_CARDS,  # bookmark values
                0,  # total
                1,  # current page
                dash.no_update,  # dummy
                []  # targets options
            )

        page_size = page_size or 1
        start_idx = (current_page - 1) * page_size
        t1 = time.perf_counter()

        with duckdb_connection(wdir) as conn:
            if conn is None:
                # If the DB is locked/unavailable, keep current preview as-is
                raise PreventUpdate
            all_targets = conn.execute("""
                                       SELECT peak_label
                                       from targets t
                                       WHERE (
                                           CASE
                                               WHEN ? = 'ms1' THEN t.ms_type = 'ms1'
                                               WHEN ? = 'ms2' THEN t.ms_type = 'ms2'
                                               ELSE TRUE
                                               END
                                           )
                                         AND (
                                           CASE
                                               WHEN ? = 'Bookmarked' THEN t.bookmark = TRUE
                                               WHEN ? = 'Unmarked' THEN t.bookmark = FALSE
                                               ELSE TRUE -- 'all' case 
                                               END
                                           )
                                         AND (
                                           t.peak_selection IS TRUE
                                               OR NOT EXISTS (SELECT 1
                                                              FROM targets t1
                                                              WHERE t1.peak_selection IS TRUE)
                                           )
                                         AND (
                                           (SELECT COUNT(*) FROM unnest(?::VARCHAR[])) = 0
                                               OR peak_label IN (SELECT unnest(?::VARCHAR[]))
                                           )
                                       """, [selection_ms_type, selection_ms_type,
                                             selection_bookmark, selection_bookmark,
                                             selected_targets, selected_targets]).fetchall()

            all_targets = [row[0] for row in all_targets]
            
            # Query ALL targets for dropdown options (without selected_targets filter)
            # This prevents the dropdown from getting "stuck" showing only selected targets
            dropdown_targets = conn.execute("""
                SELECT peak_label
                FROM targets t
                WHERE (
                    CASE
                        WHEN ? = 'ms1' THEN t.ms_type = 'ms1'
                        WHEN ? = 'ms2' THEN t.ms_type = 'ms2'
                        ELSE TRUE
                    END
                )
                AND (
                    CASE
                        WHEN ? = 'Bookmarked' THEN t.bookmark = TRUE
                        WHEN ? = 'Unmarked' THEN t.bookmark = FALSE
                        ELSE TRUE
                    END
                )
                AND (
                    t.peak_selection IS TRUE
                    OR NOT EXISTS (SELECT 1 FROM targets t1 WHERE t1.peak_selection IS TRUE)
                )
                ORDER BY peak_label
            """, [selection_ms_type, selection_ms_type,
                  selection_bookmark, selection_bookmark]).fetchall()
            dropdown_targets = [row[0] for row in dropdown_targets]
            
            # Adjust current_page if it's beyond the available pages (e.g., after deleting all targets on current page)
            total_targets = len(all_targets)
            max_page = max(1, math.ceil(total_targets / page_size)) if total_targets else 1
            if current_page > max_page:
                current_page = max_page
                start_idx = (current_page - 1) * page_size

            # Autosave the current targets table to the workspace data folder, but throttle I/O.
            try:
                data_dir = Path(wdir) / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                backup_path = data_dir / "targets_backup.csv"
                should_write = True
                if backup_path.exists():
                    last_write = backup_path.stat().st_mtime
                    # Avoid hammering disk on every preview refresh.
                    should_write = (time.time() - last_write) > 30
                if should_write:
                    # Use DuckDB COPY for faster backup (3.41x speedup vs pandas)
                    conn.execute(
                        "COPY (SELECT * FROM targets) TO ? (HEADER, DELIMITER ',')",
                        (str(backup_path),)
                    )
            except Exception:
                pass

            query = f"""
                                WITH picked_samples AS (
                                    SELECT ms_file_label, color, label, sample_type
                                    FROM samples
                                    WHERE use_for_optimization = TRUE
                                      AND (
                                        (SELECT COUNT(*) FROM unnest(?::VARCHAR[])) = 0
                                        OR ms_file_label IN (SELECT unnest(?::VARCHAR[]))
                                      )
                                ),
                                picked_targets AS (
                                    SELECT 
                                           t.peak_label,
                                           t.ms_type,
                                           t.bookmark,
                                           t.rt_min,
                                           t.rt_max,
                                           t.rt,
                                           t.intensity_threshold,
                                           t.mz_mean,
                                           t.rt_align_enabled,
                                           t.rt_align_shifts,
                                           t.filterLine
                                    FROM targets t
                                    WHERE (
                                        CASE
                                            WHEN ? = 'ms1' THEN t.ms_type = 'ms1'
                                            WHEN ? = 'ms2' THEN t.ms_type = 'ms2'
                                            ELSE TRUE
                                        END
                                    )
                                    AND (
                                        CASE
                                            WHEN ? = 'Bookmarked' THEN t.bookmark = TRUE
                                            WHEN ? = 'Unmarked' THEN t.bookmark = FALSE
                                            ELSE TRUE -- 'all' case 
                                        END
                                    )
                                    AND (
                                        t.peak_selection IS TRUE
                                        OR NOT EXISTS (
                                                SELECT 1 
                                                FROM targets t1
                                                WHERE t1.peak_selection IS TRUE
                                            )
                                    )
                                    AND (
                                        (SELECT COUNT(*) FROM unnest(?::VARCHAR[])) = 0
                                        OR peak_label IN (SELECT unnest(?::VARCHAR[]))
                                    )
                                    ORDER BY 
                                        CASE WHEN ? = 'mz_mean' THEN mz_mean END,
                                        peak_label
                                    -- 3) order by
                                    LIMIT ? -- 1) limit
                                        OFFSET ? -- 2) offset
                                ),
                                base AS (
                                    SELECT 
                                       c.*,
                                       s.color,
                                       s.label,
                                       t.rt_min,
                                       t.rt_max,
                                       t.rt,
                                       t.intensity_threshold,
                                       t.mz_mean,
                                       t.bookmark,  -- Add additional fields as needed
                                       t.ms_type,
                                        t.rt_align_enabled,
                                        t.rt_align_shifts,
                                        t.filterLine,
                                           s.sample_type
                                    FROM chromatograms c
                                          JOIN picked_samples s USING (ms_file_label)
                                          JOIN picked_targets t USING (peak_label)
                                ),
                                     -- Pair up (scan_time[i], intensity[i]) into a list of structs
                                filtered AS (
                                    SELECT peak_label,
                                           ms_file_label,
                                           color,
                                           label,
                                           rt_min,
                                           rt_max,
                                           rt,
                                           mz_mean,
                                           bookmark,
                                           ms_type,
                                           rt_align_enabled,
                                           rt_align_shifts,
                                           filterLine,
                                           sample_type,
                                           list_transform(
                                                   range(1, len(scan_time) + 1),
                                                   i -> struct_pack(
                                                       t := list_extract(scan_time, i),
                                                       i := list_extract(intensity, i)
                                                   )
                                               )
                                           AS pairs_raw,
                                           -- Filter to RT window for preview (no margin needed - display is fixed to this range)
                                           list_filter(pairs_raw, p -> p.t >= rt_min AND p.t <= rt_max) AS pairs_in
                                    FROM base
                                ),
                                final AS (
                                    SELECT peak_label,
                                           ms_file_label,
                                           color,
                                           label,
                                           mz_mean,
                                           rt_min,
                                           rt_max,
                                           rt,
                                           bookmark,
                                           ms_type,
                                           rt_align_enabled,
                                           rt_align_shifts,
                                           filterLine,
                                           sample_type,
                                           list_transform(pairs_in, p -> p.t) AS scan_time_sliced,
                                           list_transform(pairs_in, p -> p.i) AS intensity_sliced
                                    FROM filtered
                                )
                                SELECT *
                                FROM final
                                ORDER BY CASE WHEN ? = 'mz_mean' THEN mz_mean END,
                                        peak_label;
                                """
            df = conn.execute(query, [checkedkeys, checkedkeys,  # picked_samples: COUNT and IN
                                      selection_ms_type, selection_ms_type,  # picked_targets: ms_type filter
                                      selection_bookmark, selection_bookmark,  # picked_targets: bookmark filter
                                      selected_targets, selected_targets,  # picked_targets: specific targets
                                      targets_order, page_size, start_idx, targets_order]  # ordering and pagination
                              ).pl()

        titles = []
        figures = []
        bookmarks = []
        if not isinstance(preview_y_range, dict):
            preview_y_range = {}
        updated_preview_y_range = dict(preview_y_range)

        for peak_label_data, peak_data in df.group_by(
                ['peak_label', 'ms_type', 'bookmark', 'rt_min', 'rt_max', 'rt', 'mz_mean', 'filterLine', 'rt_align_enabled', 'rt_align_shifts'],
                maintain_order=True):
            peak_label, ms_type, bookmark, rt_min, rt_max, rt, mz_mean, filterLine, rt_align_enabled, rt_align_shifts = peak_label_data
            
            # Parse alignment shifts if enabled
            shifts_map = {}
            if rt_align_enabled and rt_align_shifts:
                try:
                    shifts_map = json.loads(rt_align_shifts)
                except Exception as e:
                    logger.error(f"Error parsing alignment shifts for {peak_label}: {e}")

            titles.append(peak_label)
            bookmarks.append(int(bookmark))  # convert bool to int

            fig = Patch()
            traces = []
            y_max = 0.0
            y_min_pos = None
            # Count samples per sample_type to sort by group size
            rows_list = list(peak_data.iter_rows(named=True))
            sample_type_counts = {}
            for row in rows_list:
                stype = row.get('sample_type')
                sample_type_counts[stype] = sample_type_counts.get(stype, 0) + 1
            
            # Sort rows: larger sample_type groups first, so smaller groups are drawn last (on top)
            rows_sorted = sorted(rows_list, key=lambda r: sample_type_counts.get(r.get('sample_type'), 0), reverse=True)
            
            for i, row in enumerate(rows_sorted):
                
                scan_time = np.array(row['scan_time_sliced'])
                intensity = np.array(row['intensity_sliced'])

                # Apply alignment shift if available (shifts are stored per-file in DB)
                if rt_align_enabled and shifts_map:
                    ms_file_label = row.get('ms_file_label')
                    shift = shifts_map.get(ms_file_label, 0.0)
                    if shift != 0:
                        scan_time = scan_time + shift
                
                # Filter by rt_min/rt_max (since we fetched full traces)
                mask = (scan_time >= rt_min) & (scan_time <= rt_max)
                if not np.any(mask):
                    continue
                
                scan_time_sliced = scan_time[mask]
                intensity_sliced = intensity[mask]

                intensity_sliced = apply_savgol_smoothing(
                    intensity_sliced, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_ORDER
                )

                if ms_type == 'ms1':
                    scan_time_sliced, intensity_sliced = apply_lttb_downsampling(
                        scan_time_sliced, intensity_sliced, n_out=LTTB_TARGET_POINTS
                    )

                # MS2/SRM data has sparse peaks - use min_peak_width=1 and higher baseline
                # MS1 uses default parameters (min_peak_width=3, baseline=1.0)
                if ms_type == 'ms2':
                    scan_time_sparse, intensity_sparse = sparsify_chrom(
                        scan_time_sliced, intensity_sliced, min_peak_width=1, baseline=10.0
                    )
                else:
                    scan_time_sparse, intensity_sparse = sparsify_chrom(
                        scan_time_sliced, intensity_sliced
                    )
                
                if len(intensity_sparse) > 0:
                    local_max = intensity_sparse.max()
                    if local_max > y_max:
                        y_max = float(local_max)
                    
                    # Vectorized min > 0
                    pos_vals = intensity_sparse[intensity_sparse > 0]
                    if len(pos_vals) > 0:
                        local_min_pos = pos_vals.min()
                        if y_min_pos is None or local_min_pos < y_min_pos:
                            y_min_pos = float(local_min_pos)

                traces.append({
                    'type': 'scatter',
                    'mode': 'lines',
                    'x': scan_time_sparse,
                    'y': intensity_sparse,
                    'name': row['label'] or row['ms_file_label'],
                    'line': {'color': row['color'], 'width': 1.5},
                })

            fig['data'] = traces

            fig['layout']['shapes'] = [
                {
                    'line': {'color': 'black', 'width': 1.5, 'dash': 'dashdot'},
                    'type': 'line',
                    'x0': rt,
                    'x1': rt,
                    'xref': 'x',
                    'y0': 0,
                    'y1': 1,
                    'yref': 'y domain'
                }
            ]
            fig['layout']['template'] = 'plotly_white'

            filter_type = (f"mz_mean = {mz_mean}"
                           if ms_type == 'ms1'
                           else f"{filterLine}")
            fig['layout']['title'] = dict(
                text=f"{peak_label}<br><sup>{filter_type}</sup>",
                font={'size': 14},
                y=0.90,
                yanchor='top'
            )

            fig['layout']['xaxis']['title'] = dict(text="Retention Time [s]", font={'size': 10})
            fig['layout']['xaxis']['autorange'] = False
            fig['layout']['xaxis']['fixedrange'] = True
            fig['layout']['xaxis']['range'] = [rt_min, rt_max]

            fig['layout']['yaxis']['title'] = dict(text="Intensity", font={'size': 10})
            fig['layout']['yaxis']['autorange'] = True
            fig['layout']['yaxis']['automargin'] = True
            fig['layout']['yaxis']['tickformat'] = "~s"

            y_key = f"{peak_label}|{ms_type}"
            prev_range = preview_y_range.get(y_key, {})
            prev_y_max = prev_range.get("y_max")
            prev_y_min = prev_range.get("y_min_pos")
            use_prev = False
            if prev_y_max and y_max:
                diff_ratio = abs(y_max - prev_y_max) / max(prev_y_max, 1.0)
                use_prev = diff_ratio < 0.05
            if use_prev:
                y_max_use = prev_y_max
                y_min_use = prev_y_min if prev_y_min else y_min_pos
            else:
                y_max_use = y_max
                y_min_use = y_min_pos

            if log_scale:
                fig['layout']['yaxis']['type'] = 'log'
                fig['layout']['yaxis']['nticks'] = 3
                fig['layout']['yaxis']['tickfont'] = {'size': 9}
                if y_max_use and y_max_use > 0:
                    y_min_use = y_min_use if y_min_use and y_min_use > 0 else max(y_max_use * 1e-6, 1e-6)
                    fig['layout']['yaxis']['range'] = [math.log10(y_min_use), math.log10(y_max_use)]
                    fig['layout']['yaxis']['autorange'] = False
            else:
                fig['layout']['yaxis']['type'] = 'linear'
                fig['layout']['yaxis']['nticks'] = 3
                fig['layout']['yaxis']['tickfont'] = {'size': 9}
                if y_max_use and y_max_use > 0:
                    fig['layout']['yaxis']['range'] = [0, y_max_use * 1.05]
                    fig['layout']['yaxis']['autorange'] = False

            if y_max_use:
                updated_preview_y_range[y_key] = {
                    "y_max": y_max_use,
                    "y_min_pos": y_min_use,
                }

            fig["layout"]["showlegend"] = False
            fig['layout']['margin'] = dict(l=45, r=5, t=55, b=30)
            # fig['layout']['uirevision'] = f"xr_{peak_label}"
            figures.append(fig)

        titles.extend([None for _ in range(MAX_NUM_CARDS - len(figures))])
        figures.extend([EMPTY_PLOTLY_FIGURE for _ in range(MAX_NUM_CARDS - len(figures))])
        bookmarks.extend([0 for _ in range(MAX_NUM_CARDS - len(bookmarks))])

        if 'targets-select' in ctx.triggered[0]['prop_id']:
            targets_select_options = dash.no_update
        else:
            # Use dropdown_targets (unfiltered list) for dropdown options
            targets_select_options = [
                {"label": target, "value": target} for target in dropdown_targets
            ]
        
        logger.debug(f"Preview refreshed in {time.perf_counter() - t1:.4f}s")
        return titles, figures, bookmarks, len(all_targets), current_page, "", targets_select_options

    app.clientside_callback(
        """(status) => {
            if (!status) {
                return [{'display': 'none'}, {'display': 'block', 'marginTop': '100px'}];
            }
            const hasChromatograms = (status.chromatograms_count || 0) > 0;
            const containerStyle = hasChromatograms ? {'display': 'block'} : {'display': 'none'};
            const emptyStyle = hasChromatograms ? {'display': 'none'} : {'display': 'flex', 'marginTop': '100px'};
            return [containerStyle, emptyStyle];
        }""",
        [
            Output('chromatogram-preview-container', 'style', allow_duplicate=True),
            Output('chromatogram-preview-empty', 'style', allow_duplicate=True),
        ],
        Input('workspace-status', 'data'),
        prevent_initial_call='initial_duplicate'
    )

    @app.callback(
        Output({'type': 'target-card-preview', 'index': ALL}, 'className'),
        Output('chromatogram-preview-container', 'style', allow_duplicate=True),
        Output('chromatogram-preview-empty', 'style', allow_duplicate=True),
        Output('optimization-sidebar', 'collapsed', allow_duplicate=True),
        Output('optimization-sidebar-collapse-icon', 'icon', allow_duplicate=True),

        Input({'type': 'graph', 'index': ALL}, 'figure'),
        State({'type': 'target-card-preview', 'index': ALL}, 'className'),
        prevent_initial_call=True
    )
    def toggle_card_visibility(figures, current_class):
        visible_fig = 0
        cards_classes = []

        for i, figure in enumerate(figures):
            has_traces = bool(figure) and bool(figure.get('data'))
            if has_traces:
                cc = current_class[i].split() if current_class[i] else []
                cc.remove('is-hidden') if 'is-hidden' in cc else None
                cards_classes.append(' '.join(cc))
                visible_fig += 1
            else:
                cc = current_class[i].split() if current_class[i] else []
                cc.append('is-hidden') if 'is-hidden' not in cc else None
                cards_classes.append(' '.join(cc))

        show_empty = {'display': 'flex', 'marginTop': '100px'} if visible_fig == 0 else {'display': 'none'}
        show_space = {'display': 'none'} if visible_fig == 0 else {'display': 'block'}
        
        # Collapse sidebar when no chromatograms, expand when there are chromatograms
        sidebar_collapsed = visible_fig == 0
        sidebar_icon = 'antd-right' if sidebar_collapsed else 'antd-left'
        return cards_classes, show_space, show_empty, sidebar_collapsed, sidebar_icon

    ############# PREVIEW END #######################################

    ############# VIEW MODAL BEGIN #####################################
    @app.callback(
        Output('target-preview-clicked', 'data'),

        Input({'type': 'target-card-preview', 'index': ALL}, 'nClicks'),
        Input({'type': 'bookmark-target-card', 'index': ALL}, 'value'),
        Input({'type': 'delete-target-card', 'index': ALL}, 'nClicks'),
        State({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        prevent_initial_call=True
    )
    def open_chromatogram_view_modal(card_preview_clicks, bookmark_target_clicks, delete_target_clicks, data_target):
        if not any([clicks for clicks in card_preview_clicks if clicks]):
            logger.debug("open_chromatogram_view_modal: No card clicks detected, preventing update")
            raise PreventUpdate

        ctx = dash.callback_context
        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_type = ctx_trigger['type']

        if len(ctx.triggered) > 1 or trigger_type != 'target-card-preview':
            raise PreventUpdate

        prop_id = ctx_trigger['index']
        return data_target[prop_id]

    @app.callback(
        Output('chromatogram-view-modal', 'visible'),
        Output('slider-reference-data', 'data', allow_duplicate=True),
        Output('chromatograms', 'data', allow_duplicate=True),

        Input('target-preview-clicked', 'data'),
        Input('confirm-unsave-modal', 'okCounts'),
        State('update-chromatograms', 'data'),
        State('target-note', 'value'),
        State('rt-alignment-data', 'data'),  # Get RT alignment calculation data
        State('chromatogram-view-rt-align', 'checked'),  # Get current toggle state

        State('slider-reference-data', 'data'),
        State('slider-data', 'data'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def handle_modal_open_close(target_clicked, close_without_save_clicks, update_chromatograms,
                                target_note, rt_alignment_data, rt_align_toggle, slider_ref, slider_data, wdir):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'target-preview-clicked':
            return True, dash.no_update, dash.no_update
            # if not has_changes, close it
        elif False: # trigger_id == 'chromatogram-view-close':
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    return dash.no_update, dash.no_update, dash.no_update
                
                # Always save the current RT alignment toggle state
                if rt_align_toggle:
                    # Toggle is ON - save alignment data
                    if rt_alignment_data and rt_alignment_data.get('enabled'):
                        # We have valid alignment data to save
                        import json
                        # Save per-file shifts for accurate processing (not sample-type averages)
                        shifts_json = json.dumps(rt_alignment_data.get('shifts_per_file', {}))
                       
                        conn.execute("""
                            UPDATE targets 
                            SET rt_align_enabled = TRUE,
                                rt_align_reference_rt = ?,
                                rt_align_shifts = ?,
                                rt_align_rt_min = ?,
                                rt_align_rt_max = ?
                            WHERE peak_label = ?
                        """, [
                            rt_alignment_data['reference_rt'],
                            shifts_json,
                            rt_alignment_data['rt_min'],
                            rt_alignment_data['rt_max'],

                            target_clicked
                        ])
                        logger.debug(f"Saved RT alignment: enabled=TRUE, ref={rt_alignment_data['reference_rt']:.2f}s")
                    else:
                        logger.warning("RT align toggle is ON but no alignment data available - not saving")
                else:
                    # Toggle is OFF - clear alignment data
                    conn.execute("""
                        UPDATE targets 
                        SET rt_align_enabled = FALSE,
                            rt_align_reference_rt = NULL,
                            rt_align_shifts = NULL,
                            rt_align_rt_min = NULL,
                            rt_align_rt_max = NULL
                        WHERE peak_label = ?
                    """, [target_clicked])
                    logger.debug("Saved RT alignment: enabled=FALSE (cleared all data)")
                
                # Prepare final notes:
                # 1. Remove any existing auto-generated RT Alignment note to prevent duplication
                #    or persistence when disabled.
                raw_note = target_note or ''
                # Split by double newline to find blocks
                note_parts = raw_note.split('\n\n')
                # Filter out lines starting with our specific prefix
                clean_parts = [p for p in note_parts if not p.startswith("RT Alignment: ✓ Applied")]
                final_note = '\n\n'.join(clean_parts)
                
                if rt_align_toggle and rt_alignment_data and rt_alignment_data.get('enabled'):
                    # Generate human-readable alignment note
                    ref_rt = rt_alignment_data['reference_rt']
                    shifts = rt_alignment_data.get('shifts_by_sample_type', {})
                    shift_str = ', '.join([f"{st}: {shift:+.1f}s" for st, shift in sorted(shifts.items())])
                    alignment_note = f"RT Alignment: ✓ Applied, ref={ref_rt:.2f}s | {shift_str}"
                    
                    # Prepend alignment note (so it's always at top)
                    if final_note:
                        final_note = f"{alignment_note}\n\n{final_note}"
                    else:
                        final_note = alignment_note
                
                # Save notes
                conn.execute("UPDATE targets SET notes = ? WHERE peak_label = ?",
                             (final_note, target_clicked))

                # Auto-save RT-span if changed (no more confirmation modal)
                # This must be inside the 'with' block to have valid connection
                if slider_data and slider_ref:
                    slider_value = slider_data.get('value') if isinstance(slider_data, dict) else None
                    reference_value = slider_ref.get('value') if isinstance(slider_ref, dict) else None
                    
                    if slider_value and isinstance(slider_value, dict):
                        # Check if values changed
                        if reference_value is None or slider_value != reference_value:
                            rt_min = slider_value.get('rt_min')
                            rt_max = slider_value.get('rt_max')
                            rt = slider_value.get('rt', (rt_min + rt_max) / 2 if rt_min and rt_max else None)
                            
                            if rt_min is not None and rt_max is not None:
                                conn.execute("""
                                    UPDATE targets 
                                    SET rt_min = ?, rt_max = ?, rt = ?
                                    WHERE peak_label = ?
                                """, [rt_min, rt_max, rt, target_clicked])
                                logger.info(f"Auto-saved RT-span for '{target_clicked}' on close: [{rt_min:.2f}, {rt_max:.2f}]")

            # Always refresh preview when RT alignment was changed
            # Use a timestamp to trigger the chromatograms_preview callback
            import time
            refresh_signal = update_chromatograms or {'refresh': time.time()}

            
            # Always close the modal
            return False, None, refresh_signal
        elif trigger_id == 'confirm-unsave-modal':
            # Close modal without saving changes - but still refresh preview
            if close_without_save_clicks:
                import time as time_module
                refresh_signal = update_chromatograms or {'refresh': time_module.time()}
                return False, None, refresh_signal

        return dash.no_update, dash.no_update, dash.no_update


    ############# VIEW MODAL END #######################################

    ############# VIEW BEGIN #######################################
    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),

        Input('chromatogram-view-log-y', 'checked'),
        State('chromatogram-view-plot', 'figure'),
        State('chromatogram-view-plot-max', 'data'),
        State('chromatogram-view-plot-points', 'data'),
        State('rt-alignment-data', 'data'),
        prevent_initial_call=True
    )
    def chromatogram_view_y_scale(log_scale, figure, max_y, total_points, rt_alignment_data):
        # max_y is stored as {"min_y": ..., "max_y": ...}
        if not max_y or not isinstance(max_y, dict):
            raise PreventUpdate
        y_min = max_y.get("min_y", 0)
        y_max = max_y.get("max_y", 1)
        fig = Patch()

        # Use RT span only for y-range calculations.
        shape = (figure.get('layout', {}).get('shapes') or [{}])[0]
        x_left, x_right = shape.get('x0'), shape.get('x1')
        if x_left is None or x_right is None:
            x_left = x_right = None

        if log_scale:
            fig['layout']['yaxis']['type'] = 'log'
            y_range_calc = None
            if x_left is not None and x_right is not None:
                y_range_calc = _calc_y_range_numpy(figure.get('data', []), min(x_left, x_right), max(x_left, x_right), True)
            if y_range_calc:
                fig['layout']['yaxis']['range'] = y_range_calc
                fig['layout']['yaxis']['autorange'] = False
            else:
                log_y_min = math.log10(y_min) if y_min > 0 else y_min
                log_y_max = math.log10(y_max) if y_max > 0 else y_max
                fig['layout']['yaxis']['range'] = [log_y_min, log_y_max]
                fig['layout']['yaxis']['autorange'] = False
        else:
            fig['layout']['yaxis']['type'] = 'linear'
            y_range_calc = None
            if x_left is not None and x_right is not None:
                y_range_calc = _calc_y_range_numpy(figure.get('data', []), min(x_left, x_right), max(x_left, x_right), False)
            if y_range_calc:
                fig['layout']['yaxis']['range'] = y_range_calc
                fig['layout']['yaxis']['autorange'] = False
            else:
                fig['layout']['yaxis']['range'] = [0, y_max * 1.05]
                fig['layout']['yaxis']['autorange'] = False
        return fig

    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Input('chromatogram-view-groupclick', 'checked'),
        State('rt-alignment-data', 'data'),
        prevent_initial_call=True
    )
    def chromatogram_view_legend_group(groupclick, rt_alignment_data):
        fig = Patch()
        if groupclick:
            fig['layout']['legend']['groupclick'] = 'togglegroup'
        else:
            fig['layout']['legend']['groupclick'] = 'toggleitem'
        return fig

    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Input('chromatogram-view-megatrace', 'checked'),
        Input('chromatogram-view-full-range', 'checked'),
        Input('chromatogram-view-savgol', 'checked'),
        State('chromatogram-view-plot', 'figure'),
        State('target-preview-clicked', 'data'),
        State('wdir', 'data'),
        State('rt-alignment-data', 'data'),  # Check if RT alignment is active
        prevent_initial_call=True
    )
    def update_megatrace_mode(use_megatrace, full_range, use_savgol, figure, target_clicked, wdir, rt_alignment_data):
        if not wdir or not target_clicked:
            logger.debug("update_megatrace_mode: No workspace directory or target clicked, preventing update")
            raise PreventUpdate
        
        with duckdb_connection(wdir) as conn:
            chrom_df = get_chromatogram_dataframe(
                conn, target_clicked, full_range=full_range, wdir=wdir
            )
            
            # Fetch ms_type for this target
            target_ms_type = conn.execute(
                "SELECT ms_type FROM targets WHERE peak_label = ?", [target_clicked]
            ).fetchone()
            target_ms_type = target_ms_type[0] if target_ms_type else None
        
        # Apply RT alignment if active
        rt_alignment_shifts = None
        if rt_alignment_data and rt_alignment_data.get('enabled'):
            rt_alignment_shifts = calculate_rt_alignment(
                chrom_df, 
                rt_alignment_data['rt_min'], 
                rt_alignment_data['rt_max']
            )

            # logger.debug(f"Megatrace callback: Applying RT alignment with {len(rt_alignment_shifts)} shifts")
        
        smoothing_params = None
        if use_savgol and (not full_range or target_ms_type != 'ms1'):
            smoothing_params = {
                'enabled': True,
                'window_length': SAVGOL_WINDOW,
                'polyorder': SAVGOL_ORDER
            }

        downsample_params = None
        if target_ms_type == 'ms1' and not full_range:
            downsample_params = {
                'enabled': True,
                'n_out': LTTB_TARGET_POINTS
            }

        traces, x_min, x_max, y_min, y_max = generate_chromatogram_traces(
            chrom_df, 
            use_megatrace=use_megatrace,
            rt_alignment_shifts=rt_alignment_shifts,
            ms_type=target_ms_type,
            smoothing_params=smoothing_params,
            downsample_params=downsample_params
        )
        
        fig = Patch()
        fig['data'] = traces
        # Recompute y-range using RT span only
        is_log = figure and figure.get('layout', {}).get('yaxis', {}).get('type') == 'log'
        shape = (figure.get('layout', {}).get('shapes') or [{}])[0] if figure else {}
        x_left, x_right = shape.get('x0'), shape.get('x1')
        if x_left is None or x_right is None:
            x_left, x_right = x_min, x_max

        if x_left is not None and x_right is not None:
            y_range_calc = _calc_y_range_numpy(traces, min(x_left, x_right), max(x_left, x_right), is_log=is_log)
            if y_range_calc:
                fig['layout']['yaxis']['range'] = y_range_calc
                fig['layout']['yaxis']['autorange'] = False
            else:
                if is_log:
                    log_y_min = math.log10(y_min) if y_min and y_min > 0 else y_min
                    log_y_max = math.log10(y_max) if y_max and y_max > 0 else y_max
                    fig['layout']['yaxis']['range'] = [log_y_min, log_y_max]
                else:
                    fig['layout']['yaxis']['range'] = [y_min, y_max * 1.05]
                fig['layout']['yaxis']['autorange'] = False
        # We don't necessarily update ranges here to preserve user zoom/pan if desired, 
        # but to be consistent with main load, we might want to. 
        # For now let's update data only, or update everything if user expects a "reset" view.
        # Given this is a toggle, replacing data is key.
        if use_megatrace:
             fig['layout']['hovermode'] = False
        else:
             fig['layout']['hovermode'] = 'closest'
        return fig

    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Output('rt-alignment-data', 'data'),  # Store alignment info for saving to notes
        Input('chromatogram-view-rt-align', 'checked'),
        State('chromatogram-view-plot', 'figure'),
        State('chromatogram-view-megatrace', 'checked'),
        State('chromatogram-view-full-range', 'checked'),
        State('chromatogram-view-savgol', 'checked'),
        State('slider-data', 'data'),  # Use current slider values, not reference
        State('target-preview-clicked', 'data'),
        State('wdir', 'data'),
        State('rt-alignment-data', 'data'),  # Check if this is a restoration
        prevent_initial_call=True
    )
    def apply_rt_alignment(use_alignment, figure, use_megatrace, full_range, use_savgol, slider_current, target_clicked, wdir, existing_rt_data):
        """Apply or remove RT alignment when toggle changes"""
        # logger.debug(f"RT Alignment callback triggered: use_alignment={use_alignment}")
        
        # If turning ON and we already have matching alignment data in the store,
        # this is likely a state restoration - skip to avoid overwriting pre-aligned figure
        if use_alignment and existing_rt_data and existing_rt_data.get('enabled'):
            logger.debug("apply_rt_alignment: State restoration detected (data already in store), preventing update")
            raise PreventUpdate
        
        if not wdir or not target_clicked or not slider_current:
            logger.warning("RT Alignment: Missing required data, raising PreventUpdate")
            raise PreventUpdate
        
        # rt_min and rt_max are nested inside the 'value' key
        value_dict = slider_current.get('value', {})
        rt_min = value_dict.get('rt_min')
        rt_max = value_dict.get('rt_max')
        
        # logger.debug(f"RT range: rt_min={rt_min}, rt_max={rt_max}")
        
        if rt_min is None or rt_max is None:
            logger.warning("RT Alignment: rt_min or rt_max is None, raising PreventUpdate")
            raise PreventUpdate
        
        with duckdb_connection(wdir) as conn:
            query = """
                    WITH picked_samples AS (SELECT ms_file_label, color, label, sample_type
                                            FROM samples
                                            WHERE use_for_optimization = TRUE
                    ),
                         picked_target AS (SELECT peak_label,
                                                  intensity_threshold
                                           FROM targets
                                           WHERE peak_label = ?),
                         base AS (SELECT c.*,
                                         s.color,
                                         s.label,
                                         s.sample_type,
                                         t.intensity_threshold
                                  FROM chromatograms c
                                           JOIN picked_samples s USING (ms_file_label)
                                           JOIN picked_target t USING (peak_label)),
                         zipped AS (SELECT ms_file_label,
                                           color,
                                           label,
                                           sample_type,
                                           intensity_threshold,
                                           list_transform(
                                                   range(1, len(scan_time) + 1),
                                                   i -> struct_pack(
                                                           t := list_extract(scan_time, i),
                                                           i := list_extract(intensity,  i)
                                                        )
                                           ) AS pairs
                                    FROM base),

                         sliced AS (SELECT ms_file_label,
                                           color,
                                           label,
                                           sample_type,
                                           pairs,
                                           list_filter(pairs, p -> p.i >= COALESCE(intensity_threshold, 0)) AS pairs_in
                                    FROM zipped),
                         final AS (SELECT ms_file_label,
                                          color,
                                          label,
                                          sample_type,
                                          list_transform(pairs_in, p -> p.t)                            AS scan_time_sliced,
                                          list_transform(pairs_in, p -> p.i)                            AS intensity_sliced,
                                          CASE
                                              WHEN len(pairs) = 0 THEN NULL
                                              ELSE list_max(list_transform(pairs, p -> p.i)) * 1.10 END AS
                                                                                                           intensity_max_in_range,
                                          CASE
                                              WHEN len(pairs) = 0 THEN NULL
                                              ELSE list_min(list_transform(pairs, p -> p.i)) END        AS intensity_min_in_range,
                                          CASE
                                              WHEN len(pairs) = 0 THEN NULL
                                              ELSE list_max(list_transform(pairs, p -> p.t)) END        AS scan_time_max_in_range,
                                          CASE
                                              WHEN len(pairs) = 0 THEN NULL
                                              ELSE list_min(list_transform(pairs, p -> p.t)) END        AS scan_time_min_in_range

                                   FROM sliced)
                    SELECT *
                    FROM final
                    ORDER BY ms_file_label;
                    """

            chrom_df = conn.execute(query, [target_clicked]).pl()
            
            # Fetch ms_type for this target
            target_ms_type = conn.execute(
                "SELECT ms_type FROM targets WHERE peak_label = ?", [target_clicked]
            ).fetchone()
            target_ms_type = target_ms_type[0] if target_ms_type else None
        
        # Calculate RT alignment shifts if alignment is enabled
        rt_alignment_shifts = None
        alignment_data = None
        
        if use_alignment:
            rt_alignment_shifts = calculate_rt_alignment(chrom_df, rt_min, rt_max)
            
            # Calculate shifts per sample type for notes
            shifts_per_sample_type = calculate_shifts_per_sample_type(chrom_df, rt_alignment_shifts)
            
            # Find reference RT (median of apex RTs)
            apex_rts = []
            for row in chrom_df.iter_rows(named=True):
                scan_time = np.array(row['scan_time_sliced'])
                intensity = np.array(row['intensity_sliced'])
                mask = (scan_time >= rt_min) & (scan_time <= rt_max)
                if mask.any():
                    rt_in_range = scan_time[mask]
                    int_in_range = intensity[mask]
                    apex_idx = int_in_range.argmax()
                    apex_rts.append(rt_in_range[apex_idx])
            
            reference_rt = float(np.median(apex_rts)) if apex_rts else None
            
            # Store alignment data for saving to notes
            alignment_data = {
                'enabled': True,
                'reference_rt': reference_rt,
                'shifts_by_sample_type': shifts_per_sample_type,  # For notes (human-readable)
                'shifts_per_file': rt_alignment_shifts,  # For processing (per-file accuracy)
                'rt_min': rt_min,
                'rt_max': rt_max
            }
            # logger.debug(f"RT Alignment data prepared: {alignment_data}")
        
        # Regenerate traces with or without alignment
        smoothing_params = None
        if use_savgol and (not full_range or target_ms_type != 'ms1'):
            smoothing_params = {
                'enabled': True,
                'window_length': SAVGOL_WINDOW,
                'polyorder': SAVGOL_ORDER
            }

        downsample_params = None
        if target_ms_type == 'ms1' and not full_range:
            downsample_params = {
                'enabled': True,
                'n_out': LTTB_TARGET_POINTS
            }

        traces, x_min, x_max, y_min, y_max = generate_chromatogram_traces(
            chrom_df, 
            use_megatrace=use_megatrace,
            rt_alignment_shifts=rt_alignment_shifts,
            ms_type=target_ms_type,
            smoothing_params=smoothing_params,
            downsample_params=downsample_params
        )
        
        fig = Patch()
        fig['data'] = traces
        
        # Update x-axis range if alignment is applied
        if use_alignment and rt_alignment_shifts:
            # Recalculate x_min and x_max based on aligned data
            all_x_values = []
            for trace in traces:
                if trace.get('x'):
                    all_x_values.extend([x for x in trace['x'] if x is not None])
            if all_x_values:
                fig['layout']['xaxis']['range'] = [min(all_x_values), max(all_x_values)]
        
        return fig, alignment_data


    @app.callback(
        Output('chromatogram-view-lock-range', 'checked', allow_duplicate=True),
        Input('chromatogram-view-rt-align', 'checked'),
        prevent_initial_call=True
    )
    def lock_rt_span_when_aligning(rt_align_on):
        """Force RT span to Lock mode when RT alignment is ON"""
        # TEMPORARILY DISABLED FOR TESTING
        raise PreventUpdate
        # if not rt_align_on:
        #     logger.debug("lock_rt_span_when_aligning: RT alignment is off, preventing update")
        #     raise PreventUpdate
        # # logger.debug(f"Lock RT span callback: rt_align_on={rt_align_on}, setting Lock mode (checked={rt_align_on})")
        # return rt_align_on  # True = Lock mode, False = Edit mode


    @app.callback(
        Output('chromatogram-view-rt-align', 'checked', allow_duplicate=True),
        Input('chromatogram-view-lock-range', 'checked'),
        State('chromatogram-view-rt-align', 'checked'),
        prevent_initial_call=True
    )
    def turn_off_alignment_when_editing(is_locked, rt_align_on):
        """Turn OFF RT alignment when user switches from Lock to Edit mode"""
        # TEMPORARILY DISABLED FOR TESTING
        raise PreventUpdate
        # # When switching to Edit mode (is_locked=False), turn off alignment
        # if not is_locked and rt_align_on:
        #     logger.debug("RT span switched to Edit mode - turning OFF RT alignment")
        #     return False
        # raise PreventUpdate


    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Output('chromatogram-view-modal', 'title'),
        Output('chromatogram-view-modal', 'loading'),
        Output('slider-reference-data', 'data'),
        Output('slider-data', 'data', allow_duplicate=True),  # make sure this is reset
        Output('chromatogram-view-plot-max', 'data'),
        Output('chromatogram-view-plot-points', 'data'),
        Output('chromatogram-view-megatrace', 'checked', allow_duplicate=True),
        Output('chromatogram-view-log-y', 'checked', allow_duplicate=True),
        Output('chromatogram-view-groupclick', 'checked', allow_duplicate=True),
        Output('chromatogram-view-full-range', 'checked', allow_duplicate=True),
        Output('chromatogram-view-full-range', 'disabled'),
        Output('chromatogram-view-full-range-tooltip', 'title'),
        Output('chromatogram-view-rt-align', 'checked', allow_duplicate=True),  # Set RT alignment state
        Output('rt-alignment-data', 'data', allow_duplicate=True),  # Load alignment data
        Output('target-note', 'value', allow_duplicate=True),
        Output('chromatogram-view-lock-range', 'checked', allow_duplicate=True), # Set initial lock state
        Output('bookmark-target-modal-btn', 'icon'),

        Input('target-preview-clicked', 'data'),
        State('chromatogram-preview-log-y', 'checked'),
        State('sample-type-tree', 'checkedKeys'),
        State('wdir', 'data'),
        State('chromatogram-view-modal', 'visible'),  # Check if modal is already open (navigation)
        State('chromatogram-view-megatrace', 'checked'),  # Current megatrace state
        State('chromatogram-view-log-y', 'checked'),  # Current log-y state  
        State('chromatogram-view-groupclick', 'checked'),  # Current legend behavior state
        State('chromatogram-view-full-range', 'checked'),  # Current full range state
        State('chromatogram-view-savgol', 'checked'),
        prevent_initial_call=True
    )
    def chromatogram_view_modal(target_clicked, log_scale, checkedKeys, wdir, 
                                 modal_already_open, current_megatrace, current_log_y, current_groupclick, current_full_range,
                                 current_savgol):

        if not wdir:
            raise PreventUpdate
        with duckdb_connection(wdir) as conn:
            # Load target data including RT alignment columns
            d = conn.execute("""
            SELECT rt, rt_min, rt_max, COALESCE(notes, ''), ms_type,
                   rt_align_enabled, rt_align_reference_rt, rt_align_shifts,
                   rt_align_rt_min, rt_align_rt_max, bookmark
            FROM targets 
            WHERE peak_label = ?
        """, [target_clicked]).fetchall()
        
            if d:
                rt, rt_min, rt_max, note, target_ms_type, align_enabled, align_ref_rt, align_shifts_json, align_rt_min, align_rt_max, bookmark_state = d[0]
            else:
                rt, rt_min, rt_max, note = None, None, None, ''
                target_ms_type = None
                align_enabled = False
                align_ref_rt = None
                align_shifts_json = None
                align_rt_min = None
                align_rt_max = None
                bookmark_state = False

            # Use helper function to fetch data
            chrom_df = get_chromatogram_dataframe(
                conn,
                target_clicked,
                full_range=current_full_range if modal_already_open else False,
                wdir=wdir,
            )
            
            # Count samples for optimization
            n_samples = conn.execute("SELECT COUNT(*) FROM samples WHERE use_for_optimization = TRUE").fetchone()[0]
        
        # Limit full range to 90 samples to prevent OOM
        full_range_disabled = n_samples > 100
        if full_range_disabled:
            full_range = False
            full_range_tooltip = f"Full Range disabled (>90 samples, total optimization samples: {n_samples}) to prevent OOM."
        else:
            full_range_tooltip = "Show entire chromatogram (slower) vs 30s window"
            
        try:
            n_sample_types = chrom_df['sample_type'].n_unique()
            group_legend = True if n_sample_types > 1 else False
        except Exception as e:
            logger.warning(f"Error determining sample types: {e}")
            group_legend = False

        t1 = time.perf_counter()
        fig = Patch()
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

        legend_groups = set()
        traces = []
        total_points = 0
        # TODO: check if chrom_df is empty and Implement an empty widget to show when no data

        MAX_TRACES = 200

        if len(chrom_df) <= MAX_TRACES:
            use_megatrace = False
        else:
            use_megatrace = True
        
        # If modal is already open (navigation), preserve current toggle states
        if modal_already_open:
            # Use current states instead of recalculating defaults
            if current_megatrace is not None:
                use_megatrace = current_megatrace
            if current_log_y is not None:
                log_scale = current_log_y
            if current_groupclick is not None:
                group_legend = current_groupclick
            if current_full_range is not None:
                full_range = current_full_range
            else:
                full_range = False # Default off
        else:
             full_range = False # Reset if new open

        # Calculate RT alignment shifts if enabled in database
        rt_alignment_shifts_to_apply = None
        if align_enabled and align_ref_rt is not None:
            # Calculate alignment shifts from stored data
            rt_alignment_shifts_to_apply = calculate_rt_alignment(chrom_df, align_rt_min, align_rt_max)
            logger.info(f"Applying saved RT alignment on modal open: ref={align_ref_rt:.2f}s")
        
        smoothing_params = None
        if current_savgol and (not full_range or target_ms_type != 'ms1'):
            smoothing_params = {
                'enabled': True,
                'window_length': SAVGOL_WINDOW,
                'polyorder': SAVGOL_ORDER
            }

        downsample_params = None
        if target_ms_type == 'ms1' and not full_range:
            downsample_params = {
                'enabled': True,
                'n_out': LTTB_TARGET_POINTS
            }

        traces, x_min, x_max, y_min, y_max = generate_chromatogram_traces(
            chrom_df, 
            use_megatrace=use_megatrace,
            rt_alignment_shifts=rt_alignment_shifts_to_apply,
            ms_type=target_ms_type,
            smoothing_params=smoothing_params,
            downsample_params=downsample_params
        )

        if traces:
            total_points = sum(len(t['x']) for t in traces)

        # ------------------------------------
        # Assemble Figure
        # ------------------------------------
        fig['layout']['xaxis']['range'] = [x_min, x_max]
        fig['layout']['yaxis']['range'] = [y_min, y_max * 1.05]
        fig['layout']['xaxis']['autorange'] = False
        fig['layout']['yaxis']['autorange'] = False
        fig['layout']['_initial_alignment_applied'] = (rt_alignment_shifts_to_apply is not None)  # Marker for debugging
        fig['data'] = traces
        # fig['layout']['title'] = {'text': f"{target_clicked} (rt={rt})"}
        fig['layout']['shapes'] = []
        if use_megatrace:
            fig['layout']['hovermode'] = False
        else:
            fig['layout']['hovermode'] = 'closest'

        fig['layout']['annotations'] = [
            {
                'bgcolor': 'white',
                'font': {'color': 'black', 'size': 12, 'weight': 'bold'},
                'showarrow': False,
                'ax': -20,
                'ay': -15,
                'axref': 'pixel',
                'ayref': 'pixel',
                'text': f"RT-min: {rt_min:.1f}s" if rt_min is not None else 'RT-min',
                'x': rt_min,
                'xanchor': 'right',
                'xref': 'x',
                'y': 1,
                'yanchor': 'top',
                # 'yref': 'y domain',
                'yref': 'paper',
                'yshift': 15

            },
            {
                'bgcolor': 'white',
                'font': {'color': 'black', 'size': 12, 'weight': 'bold'},
                'showarrow': False,
                'ax': 20,
                'ay': -15,
                'axref': 'pixel',
                'ayref': 'pixel',
                'text': f"RT-max: {rt_max:.1f}s" if rt_max is not None else 'RT-max',
                'x': rt_max,
                'xanchor': 'left',
                'xref': 'x',
                'y': 1,
                'yanchor': 'top',
                # 'yref': 'y domain',
                'yref': 'paper',
                'yshift': 15
            },
        ]

        fig['layout']['shapes'] = [
            {
                'fillcolor': 'green',
                'line': {'width': 0},
                'opacity': 0.1,
                'type': 'rect',
                'x0': rt_min,
                'x1': rt_max,
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y domain'
            },
            # RT vertical line (dashdot) - same style as cards, not editable
            {
                'line': {'color': 'black', 'width': 1.5, 'dash': 'dashdot'},
                'type': 'line',
                'x0': rt,
                'x1': rt,
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y domain',
                'editable': False  # Prevent dragging - use click to set position
            }
        ]
        fig['layout']['template'] = 'plotly_white'

        t_xmin = (rt_min - (rt_max - rt_min)) if rt_min else 0
        nx_min = max(t_xmin, x_min)

        t_xmax = (rt_max + (rt_max - rt_min)) if rt_max else 0
        nx_max = min(t_xmax, x_max)

        fig['layout']['xaxis']['range'] = [nx_min, nx_max]
        fig['layout']['xaxis']['autorange'] = False
        fig['layout']['yaxis']['autorange'] = False

        fig['layout']['yaxis']['type'] = 'log' if log_scale else 'linear'
        # Use RT span only for initial y-range
        y_left = rt_min if rt_min is not None else nx_min
        y_right = rt_max if rt_max is not None else nx_max
        y_range_zoom = _calc_y_range_numpy(traces, y_left, y_right, is_log=log_scale)
        if y_range_zoom:
            fig['layout']['yaxis']['range'] = y_range_zoom
            fig['layout']['yaxis']['autorange'] = False
        else:
            if log_scale:
                log_y_min = math.log10(y_min) if y_min > 0 else y_min
                log_y_max = math.log10(y_max) if y_max > 0 else y_max
                fig['layout']['yaxis']['range'] = [log_y_min, log_y_max]
            else:
                fig['layout']['yaxis']['range'] = [y_min, y_max]

        fig['layout']['margin'] = dict(l=60, r=10, t=40, b=40)

        s_data = {
            'min': nx_min,
            'max': nx_max,
            'pushable': 1,
            'step': 1,
            'tooltip': None,
            'marks': None,
            'value': {'rt_min': rt_min, 'rt': rt, 'rt_max': rt_max},
            'v_comp': {'rt_min': True, 'rt': True, 'rt_max': True}
        }
        slider_reference = s_data
        slider_dict = slider_reference.copy()
        
        # Parse RT alignment data from database
        # Simple approach: restore the exact state that was saved
        rt_align_toggle_state = False  # Default if no alignment saved
        rt_alignment_data_to_load = None
        
        if align_enabled and align_ref_rt is not None:
            try:
                import json
                shifts_per_file = json.loads(align_shifts_json) if align_shifts_json else {}
                
                # Calculate shifts_by_sample_type from per-file shifts (for notes display)
                # We need to group by sample_type and calculate median
                sample_type_shifts = {}
                for row in chrom_df.iter_rows(named=True):
                    sample_type = row['sample_type']
                    ms_file_label = row['ms_file_label']
                    shift = shifts_per_file.get(ms_file_label, 0.0)
                    if sample_type not in sample_type_shifts:
                        sample_type_shifts[sample_type] = []
                    sample_type_shifts[sample_type].append(shift)
                
                shifts_by_sample_type = {st: float(np.median(shifts)) for st, shifts in sample_type_shifts.items()}
                
                rt_alignment_data_to_load = {
                    'enabled': True,
                    'reference_rt': align_ref_rt,
                    'shifts_by_sample_type': shifts_by_sample_type,  # For notes (human-readable)
                    'shifts_per_file': shifts_per_file,  # For processing (per-file accuracy)
                    'rt_min': align_rt_min,
                    'rt_max': align_rt_max
                }
                rt_align_toggle_state = True  # Set toggle to ON to match saved state
                logger.debug(f"Restoring RT alignment state: toggle=ON, ref={align_ref_rt:.2f}s")
            except Exception as e:
                logger.error(f"Error parsing RT alignment data: {e}")

        logger.debug(f"Modal view prepared in {time.perf_counter() - t1:.4f}s")
        
        bookmark_icon = "antd-star" if bookmark_state else "antd-star" # Warning: AntdIcon names check needed. 
        # Actually standard AntD icons: 'star' (outline), 'star-filled', 'star-two-tone'.
        # feffery_antd_components uses 'antd-...' prefix.
        # Let's try 'antd-star' (outline) and 'antd-star' (filled - wait, how to distinguish?)
        # A common pattern is 'antd-star' vs 'antd-star-filled' if available, or 'antd-star' with theme.
        # But looking at existing icons: 'antd-delete', 'antd-right', 'antd-question-circle'.
        # I'll use 'antd-star' for empty, 'antd-star' (filled is usually not a separate icon name in fac unless specifically supported).
        # However, looking at other usages, 'antd-home', 'antd-filter'.
        # Let's try 'antd-star' and 'antd-star' with a different color/type, BUT the Output is 'icon'.
        # I can return a fac.AntdIcon component? No, usually just string properties if updating property.
        # Wait, the Output is `Output('bookmark-target-modal-btn', 'icon')`. The `icon` prop of AntdButton expects a Component (fac.AntdIcon) usually?
        # NO, looking at other callbacks, `Output('some-btn', 'icon')` usually expects the component structure if using Dash.
        # OR if I update the property of a component, I might need to return the component itself.
        # Let's check `Output('bookmark-target-modal-btn', 'icon')`. `icon` prop of AntdButton accepts a node.
        # So I should return `fac.AntdIcon(icon='antd-star')` or `fac.AntdIcon(icon='antd-star', style={'color': 'gold'})`?
        # Yes.
        
        icon_color = "gold" if bookmark_state else "gray"
        bookmark_icon_node = fac.AntdIcon(icon="antd-star", style={"color": icon_color})
        if bookmark_state:
             # Try to find a filled star if possible, or just use color.
             # 'antd-star' is usually outline. 'antd-star' + theme='filled' -> fac.AntdIcon(icon='antd-star', mode='filled')?
             # Let's check if I can assume 'antd-star' and just change color for now.
             pass

        return (fig, f"{target_clicked}", False, slider_reference,
                slider_dict, {"min_y": y_min, "max_y": y_max}, total_points, use_megatrace, log_scale, group_legend, 
                full_range, full_range_disabled, full_range_tooltip, rt_align_toggle_state, rt_alignment_data_to_load, note, False, bookmark_icon_node)

    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Output('slider-data', 'data', allow_duplicate=True),
        Output('action-buttons-container', 'style', allow_duplicate=True),

        Input('chromatogram-view-plot', 'relayoutData'),
        Input('slider-reference-data', 'data'),
        State('slider-data', 'data'),
        State('chromatogram-view-plot', 'figure'),
        State('chromatogram-view-plot-points', 'data'),
        State('chromatogram-view-lock-range', 'checked'),
        State('rt-alignment-data', 'data'),
        prevent_initial_call=True
    )
    def update_rt_range_from_shape(relayout, slider_reference_data, slider_data, figure_state, total_points, lock_range, rt_alignment_data):
        if not slider_reference_data:
            raise PreventUpdate

        def _maybe_pad_x_range(current_range, rt_min_val, rt_max_val, pad_seconds):
            if rt_min_val is None or rt_max_val is None:
                return None

            desired_min = rt_min_val - pad_seconds
            desired_max = rt_max_val + pad_seconds

            if not current_range or len(current_range) != 2:
                return [desired_min, desired_max]

            cur_min, cur_max = current_range
            if cur_min is None or cur_max is None:
                return [desired_min, desired_max]
            if cur_min > cur_max:
                cur_min, cur_max = cur_max, cur_min

            new_min = cur_min
            new_max = cur_max

            # Only expand if the current padding is smaller than the minimum requested.
            if rt_min_val < cur_min or (rt_min_val - cur_min) < pad_seconds:
                new_min = desired_min
            if rt_max_val > cur_max or (cur_max - rt_max_val) < pad_seconds:
                new_max = desired_max

            return [new_min, new_max]

        ctx = dash.callback_context
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if prop_id == 'slider-reference-data':
            if not slider_data:
                slider_data = slider_reference_data.copy()
            else:
                slider_data['value'] = slider_reference_data['value']
            rt_min_ref = slider_data['value']['rt_min']
            rt_max_ref = slider_data['value']['rt_max']
            is_log = figure_state and figure_state.get('layout', {}).get('yaxis', {}).get('type') == 'log'
            pad = 2.5  # seconds of padding on each side of the RT span
            current_x_range = figure_state.get('layout', {}).get('xaxis', {}).get('range') if figure_state else None

            fig = Patch()
            fig['layout']['shapes'][0]['x0'] = rt_min_ref
            fig['layout']['shapes'][0]['x1'] = rt_max_ref
            fig['layout']['shapes'][0]['y0'] = 0
            fig['layout']['shapes'][0]['y1'] = 1
            fig['layout']['shapes'][0]['yref'] = 'y domain'
            fig['layout']['shapes'][0]['fillcolor'] = 'red' if lock_range else 'green'
            fig['layout']['shapes'][0]['opacity'] = 0.1
            x_range = _maybe_pad_x_range(current_x_range, rt_min_ref, rt_max_ref, pad)
            if x_range:
                fig['layout']['xaxis']['range'] = x_range
                fig['layout']['xaxis']['autorange'] = False

            if figure_state:
                y_calc = _calc_y_range_numpy(figure_state.get('data', []), rt_min_ref, rt_max_ref, is_log)
                if y_calc:
                    fig['layout']['yaxis']['range'] = y_calc
                    fig['layout']['yaxis']['autorange'] = False

            fig['layout']['annotations'][0]['x'] = rt_min_ref
            fig['layout']['annotations'][0]['text'] = f"RT-min: {rt_min_ref:.1f}s"
            fig['layout']['annotations'][1]['x'] = rt_max_ref
            fig['layout']['annotations'][1]['text'] = f"RT-max: {rt_max_ref:.1f}s"

            # Fix: Also reset the RT line (shapes[1]) to the reference RT
            rt_ref = slider_data['value']['rt']
            fig['layout']['shapes'][1]['x0'] = rt_ref
            fig['layout']['shapes'][1]['x1'] = rt_ref

            buttons_style = {
                'visibility': 'hidden',
                'opacity': '0',
                'transition': 'opacity 0.3s ease-in-out'
            }
            return fig, slider_data, buttons_style

        if not relayout:
            logger.debug("update_rt_range_from_shape: No relayout event, preventing update")
            raise PreventUpdate

        x_range = (relayout.get('xaxis.range[0]'), relayout.get('xaxis.range[1]'))
        y_range = (relayout.get('yaxis.range[0]'), relayout.get('yaxis.range[1]'))
        has_shape_update = relayout.get('shapes[0].x0') is not None or relayout.get('shapes[0].x1') is not None

        # Allow plotly zooming (x and y) to drive axis ranges even when the RT span is locked.
        if (x_range[0] is not None and x_range[1] is not None) or (y_range[0] is not None and y_range[1] is not None):
            fig_zoom = Patch()
            is_log = figure_state and figure_state.get('layout', {}).get('yaxis', {}).get('type') == 'log'

            if x_range[0] is not None and x_range[1] is not None:
                fig_zoom['layout']['xaxis']['range'] = [x_range[0], x_range[1]]
                fig_zoom['layout']['xaxis']['autorange'] = False

            if is_log and figure_state:
                shape = (figure_state.get('layout', {}).get('shapes') or [{}])[0]
                rt_left, rt_right = shape.get('x0'), shape.get('x1')
                if rt_left is not None and rt_right is not None:
                    y_calc = _calc_y_range_numpy(figure_state.get('data', []), min(rt_left, rt_right), max(rt_left, rt_right), True)
                elif x_range[0] is not None and x_range[1] is not None:
                    y_calc = _calc_y_range_numpy(figure_state.get('data', []), x_range[0], x_range[1], True)
                else:
                    y_calc = None
                if y_calc:
                    fig_zoom['layout']['yaxis']['range'] = y_calc
                    fig_zoom['layout']['yaxis']['autorange'] = False
            else:
                if y_range[0] is not None and y_range[1] is not None:
                    fig_zoom['layout']['yaxis']['range'] = [y_range[0], y_range[1]]
                    fig_zoom['layout']['yaxis']['autorange'] = False
                elif x_range[0] is not None and x_range[1] is not None and figure_state:
                    y_calc = _calc_y_range_numpy(figure_state.get('data', []), x_range[0], x_range[1], is_log)
                    if y_calc:
                        fig_zoom['layout']['yaxis']['range'] = y_calc
                        fig_zoom['layout']['yaxis']['autorange'] = False

            return fig_zoom, dash.no_update, dash.no_update

        if lock_range and has_shape_update:
            raise PreventUpdate

        # Handle RT line drag (shapes[1]) - constrain to within span
        rt_line_x0 = relayout.get('shapes[1].x0')
        if rt_line_x0 is not None:
            rt_min_current = slider_data['value'].get('rt_min') if slider_data else None
            rt_max_current = slider_data['value'].get('rt_max') if slider_data else None
            
            if rt_min_current is not None and rt_max_current is not None:
                # Check if dragged RT is within span
                if rt_min_current <= rt_line_x0 <= rt_max_current:
                    # Valid position - update RT
                    slider_data['value']['rt'] = rt_line_x0
                    has_changes = slider_data['value'] != slider_reference_data['value']
                    buttons_style = {
                        'visibility': 'visible' if has_changes else 'hidden',
                        'opacity': '1' if has_changes else '0',
                        'transition': 'opacity 0.3s ease-in-out'
                    }
                    fig = Patch()
                    fig['layout']['shapes'][1]['x0'] = rt_line_x0
                    fig['layout']['shapes'][1]['x1'] = rt_line_x0
                    # Reset y coordinates to ensure full height
                    fig['layout']['shapes'][1]['y0'] = 0
                    fig['layout']['shapes'][1]['y1'] = 1
                    fig['layout']['shapes'][1]['yref'] = 'y domain'
                    return fig, slider_data, buttons_style
                else:
                    # Outside span - snap back to max intensity
                    rt_at_max = (rt_min_current + rt_max_current) / 2  # fallback to midpoint
                    max_intensity = -1
                    for trace in (figure_state.get('data', []) if figure_state else []):
                        xs = trace.get('x', [])
                        ys = trace.get('y', [])
                        for xv, yv in zip(xs, ys):
                            if xv is None or yv is None:
                                continue
                            if rt_min_current <= xv <= rt_max_current and yv > max_intensity:
                                max_intensity = yv
                                rt_at_max = xv
                    
                    slider_data['value']['rt'] = rt_at_max
                    has_changes = slider_data['value'] != slider_reference_data['value']
                    buttons_style = {
                        'visibility': 'visible' if has_changes else 'hidden',
                        'opacity': '1' if has_changes else '0',
                        'transition': 'opacity 0.3s ease-in-out'
                    }
                    fig = Patch()
                    fig['layout']['shapes'][1]['x0'] = rt_at_max
                    fig['layout']['shapes'][1]['x1'] = rt_at_max
                    # Reset y coordinates to ensure full height
                    fig['layout']['shapes'][1]['y0'] = 0
                    fig['layout']['shapes'][1]['y1'] = 1
                    fig['layout']['shapes'][1]['yref'] = 'y domain'
                    return fig, slider_data, buttons_style

        x0 = relayout.get('shapes[0].x0')
        x1 = relayout.get('shapes[0].x1')
        if x0 is None or x1 is None:
            raise PreventUpdate

        if not slider_data:
            slider_data = slider_reference_data.copy()

        rt_min_new = min(x0, x1)
        rt_max_new = max(x0, x1)

        # Get current RT value
        current_rt = slider_data['value'].get('rt')
        
        # Only recalculate RT to max intensity if current RT is outside the new span
        if current_rt is not None and rt_min_new <= current_rt <= rt_max_new:
            # Current RT is still within span, keep it
            rt_new = current_rt
        else:
            # Current RT is outside span, find max intensity position
            rt_new = (rt_min_new + rt_max_new) / 2  # fallback to midpoint
            max_intensity = -1
            for trace in (figure_state.get('data', []) if figure_state else []):
                xs = trace.get('x', [])
                ys = trace.get('y', [])
                for xv, yv in zip(xs, ys):
                    if xv is None or yv is None:
                        continue
                    if rt_min_new <= xv <= rt_max_new and yv > max_intensity:
                        max_intensity = yv
                        rt_new = xv

        slider_data['value'] = {
            'rt_min': rt_min_new,
            'rt': rt_new,
            'rt_max': rt_max_new,
        }

        has_changes = slider_data['value'] != slider_reference_data['value']
        buttons_style = {
            'visibility': 'visible' if has_changes else 'hidden',
            'opacity': '1' if has_changes else '0',
            'transition': 'opacity 0.3s ease-in-out'
        }

        fig = Patch()
        fig['layout']['shapes'][0]['x0'] = rt_min_new
        fig['layout']['shapes'][0]['x1'] = rt_max_new
        fig['layout']['shapes'][0]['y0'] = 0
        fig['layout']['shapes'][0]['y1'] = 1
        fig['layout']['shapes'][0]['yref'] = 'y domain'
        fig['layout']['shapes'][0]['fillcolor'] = 'green'
        fig['layout']['shapes'][0]['opacity'] = 0.1
        
        # Update RT line position (shapes[1]) to new RT value
        fig['layout']['shapes'][1]['x0'] = rt_new
        fig['layout']['shapes'][1]['x1'] = rt_new

        # adjust axes to the current RT span box for better scaling
        is_log = figure_state and figure_state.get('layout', {}).get('yaxis', {}).get('type') == 'log'
        if figure_state:
            y_range_zoom = _calc_y_range_numpy(figure_state.get('data', []), rt_min_new, rt_max_new, is_log)
            if y_range_zoom:
                fig['layout']['yaxis']['range'] = y_range_zoom
                fig['layout']['yaxis']['autorange'] = False
            pad = 2.5  # seconds of padding on each side
            current_x_range = figure_state.get('layout', {}).get('xaxis', {}).get('range')
            x_range = _maybe_pad_x_range(current_x_range, rt_min_new, rt_max_new, pad)
            if x_range:
                fig['layout']['xaxis']['range'] = x_range
                fig['layout']['xaxis']['autorange'] = False

        fig['layout']['annotations'][0]['x'] = rt_min_new
        fig['layout']['annotations'][0]['text'] = f"RT-min: {rt_min_new:.1f}s"
        fig['layout']['annotations'][1]['x'] = rt_max_new
        fig['layout']['annotations'][1]['text'] = f"RT-max: {rt_max_new:.1f}s"

        return fig, slider_data, buttons_style

    @app.callback(
        Output('chromatogram-view-plot', 'config', allow_duplicate=True),
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Input('chromatogram-view-lock-range', 'checked'),
        State('rt-alignment-data', 'data'),
        prevent_initial_call=True
    )
    def chromatogram_view_lock_range(lock_range, rt_alignment_data):
        config_patch = Patch()
        config_patch['edits']['shapePosition'] = not lock_range

        fig = Patch()
        fig['layout']['shapes'][0]['fillcolor'] = 'red' if lock_range else 'green'
        fig['layout']['shapes'][0]['opacity'] = 0.1
        fig['layout']['shapes'][0]['y0'] = 0
        fig['layout']['shapes'][0]['y1'] = 1
        fig['layout']['shapes'][0]['yref'] = 'y domain'

        return config_patch, fig

    @app.callback(
        Output('chromatogram-view-plot', 'figure', allow_duplicate=True),
        Output('slider-data', 'data', allow_duplicate=True),
        Output('action-buttons-container', 'style', allow_duplicate=True),
        Input('chromatogram-view-plot', 'clickData'),
        State('slider-data', 'data'),
        State('slider-reference-data', 'data'),
        State('rt-alignment-data', 'data'),
        prevent_initial_call=True
    )
    def set_rt_on_click(click_data, slider_data, slider_reference, rt_alignment_data):
        """Set RT position when user clicks on the chromatogram."""
        if not click_data:
            logger.debug("set_rt_on_click: No click data, preventing update")
            raise PreventUpdate
        if not slider_data or not slider_reference:
            logger.debug("set_rt_on_click: No slider data or reference, preventing update")
            raise PreventUpdate
        
        # Get clicked x position (retention time)
        point = click_data.get('points', [{}])[0]
        clicked_rt = point.get('x')
        if clicked_rt is None:
            raise PreventUpdate
        
        # If RT alignment is enabled, prevent manual RT adjustment
        if rt_alignment_data and rt_alignment_data.get('enabled'):
            logger.debug("set_rt_on_click: RT alignment is enabled, preventing manual RT adjustment")
            raise PreventUpdate
        
        # Ensure clicked RT is within the RT span
        rt_min = slider_data['value'].get('rt_min')
        rt_max = slider_data['value'].get('rt_max')
        if rt_min is not None and rt_max is not None:
            if clicked_rt < rt_min or clicked_rt > rt_max:
                raise PreventUpdate  # Don't allow setting RT outside the span
        
        # Update slider_data with new RT
        slider_data['value']['rt'] = clicked_rt
        
        # Check if there are changes compared to reference
        has_changes = slider_data['value'] != slider_reference['value']
        buttons_style = {
            'visibility': 'visible' if has_changes else 'hidden',
            'opacity': '1' if has_changes else '0',
            'transition': 'opacity 0.3s ease-in-out'
        }
        
        # Update RT line position in the figure
        fig = Patch()
        fig['layout']['shapes'][1]['x0'] = clicked_rt
        fig['layout']['shapes'][1]['x1'] = clicked_rt
        
        return fig, slider_data, buttons_style

    ############# VIEW END #######################################

    ############# COMPUTE CHROMATOGRAM BEGIN #####################################


    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("compute-chromatograms-btn", "nClicks"),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def check_requirements_server(nClicks, wdir):
        if not nClicks:
            raise PreventUpdate
        
        if not wdir:
             return fac.AntdNotification(
                message="Workspace required",
                description="Please select or create a workspace.",
                type="error",
                duration=4,
                placement="bottom",
                showProgress=True,
            )

        with duckdb_connection(wdir) as conn:
            if conn is None:
                 return fac.AntdNotification(
                    message="Database Error",
                    description="Could not connect to workspace database.",
                    type="error",
                    duration=4,
                )
            
            counts = conn.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM samples WHERE use_for_optimization = TRUE) as opt_samples,
                    (SELECT COUNT(*) FROM targets) as targets
            """).fetchone()
            
            opt_samples_count = counts[0] or 0
            targets_count = counts[1] or 0

            if opt_samples_count == 0 or targets_count == 0:
                return fac.AntdNotification(
                    message="Requirements not met",
                    description="At least one MS-file and one target are required.",
                    type="warning",
                    duration=4,
                    placement="bottom",
                    showProgress=True,
                )

        return dash.no_update


    @app.callback(
        Output('chromatograms', 'data'),
        Output('compute-chromatogram-modal', 'visible', allow_duplicate=True),

        Input('compute-chromatogram-modal', 'okCounts'),
        State("chromatograms-recompute-ms1", "checked"),
        State("chromatograms-recompute-ms2", "checked"),
        State("chromatogram-compute-cpu", "value"),
        State("chromatogram-compute-ram", "value"),
        State("chromatogram-compute-batch-size", "value"),
        State("wdir", "data"),
        background=True,
        running=[
            (Output('chromatogram-processing-progress-container', 'style'), {
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "flexDirection": "column",
                "minWidth": "200px",
                "maxWidth": "400px",
                "margin": "auto",
                'height': "60vh"
            }, {'display': 'none'}),
            (Output("chromatogram-compute-options-container", "style"), {'display': 'none'}, {'display': 'flex'}),

            (Output('compute-chromatogram-modal', 'confirmAutoSpin'), True, False),
            (Output('compute-chromatogram-modal', 'cancelButtonProps'), {'disabled': True},
             {'disabled': False}),
            (Output('compute-chromatogram-modal', 'confirmLoading'), True, False),
        ],
        progress=[
            Output("chromatogram-processing-progress", "percent", allow_duplicate=True),
            Output("chromatogram-processing-stage", "children", allow_duplicate=True),
            Output("chromatogram-processing-detail", "children", allow_duplicate=True),
        ],
        cancel=[
            Input('cancel-chromatogram-processing', 'nClicks')
        ],
        prevent_initial_call=True
    )
    def compute_chromatograms(set_progress, okCounts, recompute_ms1, recompute_ms2, n_cpus, ram, batch_size, wdir):

        if not okCounts:
            logger.debug("compute_chromatograms: Modal not confirmed, preventing update")
            raise PreventUpdate

        return _compute_chromatograms_logic(set_progress, recompute_ms1, recompute_ms2, n_cpus, ram, batch_size, wdir)

    ############# COMPUTE CHROMATOGRAM END #######################################

    @app.callback(
        Output('chromatograms', 'data', allow_duplicate=True),
        Output('chromatogram-processing-progress-container', 'style', allow_duplicate=True),
        Output('chromatogram-compute-options-container', 'style', allow_duplicate=True),
        Output('compute-chromatogram-modal', 'visible', allow_duplicate=True),
        Output('chromatogram-processing-progress', 'percent', allow_duplicate=True),
        Output('chromatogram-processing-stage', 'children', allow_duplicate=True),
        Output('chromatogram-processing-detail', 'children', allow_duplicate=True),
        Input('cancel-chromatogram-processing', 'nClicks'),
        prevent_initial_call=True
    )
    def cancel_compute_chromatograms(cancel_clicks):
        if not cancel_clicks:
            logger.debug("cancel_compute_chromatograms: No cancel clicks, preventing update")
            raise PreventUpdate
        logger.info("Chromatogram computation cancelled by user.")
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
        Output("chromatogram-compute-cpu-item", "help"),
        Output("chromatogram-compute-ram-item", "help"),
        Output("chromatogram-compute-batch-size", "value", allow_duplicate=True),
        Input("chromatogram-compute-cpu", "value"),
        Input("chromatogram-compute-ram", "value"),
        prevent_initial_call=True
    )
    def update_resource_usage_help(cpu, ram):
        help_cpu = _get_cpu_help_text(cpu)
        help_ram = _get_ram_help_text(ram)
        # Auto-calculate optimal batch size based on current CPU and RAM
        optimal_batch = calculate_optimal_batch_size(
            float(ram) if ram else 8.0,
            100000,  # Estimate for total pairs
            int(cpu) if cpu else 4
        )
        return help_cpu, help_ram, optimal_batch

    @app.callback(
        # only save the current values stored in slider-reference-data since this will shut all the actions
        Output('slider-reference-data', 'data', allow_duplicate=True),

        Input('reset-btn', 'nClicks'),
        State('slider-reference-data', 'data'),
        prevent_initial_call=True
    )
    def reset_changes(reset_clicks, slider_reference):

        if not reset_clicks:
            logger.debug("reset_changes: No reset clicks, preventing update")
            raise PreventUpdate
        return slider_reference

    @app.callback(
        Output('delete-targets-modal', 'visible'),
        Output('delete-targets-modal', 'children'),
        Output('delete-target-clicked', 'children'),

        Input({'type': 'delete-target-card', 'index': ALL}, 'nClicks'),
        Input('delete-target-from-modal', 'nClicks'),
        State({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        State('target-preview-clicked', 'data'),
        prevent_initial_call=True
    )
    def show_delete_modal(delete_clicks, delete_modal_click, data_target, target_clicked):

        ctx = dash.callback_context
        if not ctx.triggered:
            logger.debug("show_delete_modal: No callback trigger, preventing update")
            raise PreventUpdate
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        # Delete from card icon
        if trigger.startswith('{'):
            if not any(delete_clicks):
                logger.debug("show_delete_modal: No delete clicks from card, preventing update")
                raise PreventUpdate
            ctx_trigger = json.loads(trigger)
            if len(dash.callback_context.triggered) > 1:
                raise PreventUpdate
            prop_id = ctx_trigger['index']
            target = data_target[prop_id]
        # Delete from modal button
        elif trigger == 'delete-target-from-modal':
            if not delete_modal_click:
                logger.debug("show_delete_modal: No delete clicks from modal button, preventing update")
                raise PreventUpdate
            if not target_clicked:
                logger.debug("show_delete_modal: No target clicked for modal delete, preventing update")
                raise PreventUpdate
            target = target_clicked
        else:
            raise PreventUpdate

        return True, fac.AntdParagraph(f"Are you sure you want to delete `{target}` target?"), target

    #
    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        Output('drop-chromatogram', 'data'),
        Output('delete-targets-modal', 'visible', allow_duplicate=True),
        Output('target-preview-clicked', 'data', allow_duplicate=True),

        Input('delete-targets-modal', 'okCounts'),
        State('delete-target-clicked', 'children'),
        State('target-nav-store', 'data'),
        State("wdir", "data"),
        prevent_initial_call=True
    )
    def delete_targets_chromatograms(okCounts, target, nav_store, wdir):
        if not okCounts:
            logger.debug("delete_targets_chromatograms: Delete not confirmed, preventing update")
            raise PreventUpdate
        
        # Delete the target
        with duckdb_connection(wdir) as conn:
            if conn is None:
                logger.error(f"delete_target_logic: Could not connect to database for target '{target}'")
                return (fac.AntdNotification(
                            message="Database connection failed",
                            description="Could not connect to the database.",
                            type="error",
                            duration=4,
                            placement='bottom'
                        ),
                        dash.no_update,
                        False,
                        dash.no_update)
            try:
                conn.execute("BEGIN")
                conn.execute("DELETE FROM chromatograms WHERE peak_label = ?", [target])
                conn.execute("DELETE FROM targets WHERE peak_label = ?", [target])
                conn.execute("DELETE FROM results WHERE peak_label = ?", [target])
                conn.execute("COMMIT")
                logger.info(f"Deleted target '{target}' and associated chromatograms/results.")
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"Failed to delete target '{target}'", exc_info=True)
                return (fac.AntdNotification(
                            message="Failed to delete target",
                            description=f"Error: {e}",
                            type="error",
                            duration=4,
                            placement='bottom'
                        ),
                        dash.no_update,
                        False,
                        dash.no_update)
        
        # Navigate to next target instead of closing modal
        next_target = None
        if nav_store and nav_store.get('targets'):
            targets = nav_store['targets']
            current_index = nav_store.get('current_index', 0)
            
            # Remove deleted target from list
            if target in targets:
                targets.remove(target)
            
            # Navigate to next target (or previous if deleted was last)
            if targets:
                new_index = min(current_index, len(targets) - 1)
                next_target = targets[new_index]
                logger.debug(f"After deletion, navigating to target '{next_target}' (index {new_index})")
        
        notification = fac.AntdNotification(
            message=f"Target '{target}' deleted",
            type="success",
            duration=3,
            placement='bottom'
        )
        
        return (notification, True, False, next_target)

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),

        Input({'type': 'bookmark-target-card', 'index': ALL}, 'value'),
        State({'type': 'target-card-preview', 'index': ALL}, 'data-target'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def bookmark_target(bookmarks, targets, wdir):
        # TODO: change bookmark to bool since the AntdRate component returns an int and the db require a bool
        ctx = dash.callback_context
        if not ctx.triggered or len(dash.callback_context.triggered) > 1:
            logger.debug("bookmark_target: No callback trigger or multiple triggers, preventing update")
            raise PreventUpdate
        if not wdir:
            logger.debug("bookmark_target: No workspace directory set, preventing update")
            raise PreventUpdate

        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']

        return _bookmark_target_logic(bookmarks, targets, trigger_id, wdir)

    @app.callback(
        Output('bookmark-target-modal-btn', 'icon', allow_duplicate=True),
        Input('bookmark-target-modal-btn', 'nClicks'),
        State('chromatogram-view-modal', 'title'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def toggle_bookmark_from_modal(n_clicks, target_label, wdir):
        if not n_clicks or not target_label or not wdir:
            raise PreventUpdate
            
        return _toggle_bookmark_logic(target_label, wdir)

    @app.callback(
        Output('slider-reference-data', 'data', allow_duplicate=True),
        Output('notifications-container', 'children', allow_duplicate=True),
        
        Input('save-btn', 'nClicks'),
        State('slider-data', 'data'),
        State('target-preview-clicked', 'data'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def save_changes(save_clicks, slider_data, target_label, wdir):
        if not save_clicks:
             logger.debug("save_changes: No save clicks, preventing update")
             raise PreventUpdate
        
        rt_min = slider_data['value']['rt_min']
        rt_max = slider_data['value']['rt_max']
        rt = slider_data['value']['rt']
        
        with duckdb_connection(wdir) as conn:
             conn.execute("UPDATE targets SET rt_min = ?, rt_max = ?, rt = ? WHERE peak_label = ?", 
                          [rt_min, rt_max, rt, target_label])
        
        notification = fac.AntdNotification(
            message="Changes saved",
            description=f"Retention time for {target_label} updated.",
            type="success",
            duration=3,
            placement="bottom"
        )
        
        return slider_data, notification

    # =====================================================
    # TARGET NAVIGATION (Previous / Next)
    # =====================================================
    
    @app.callback(
        Output('target-nav-store', 'data'),
        Output('target-nav-counter', 'children'),
        Output('target-nav-prev', 'disabled'),
        Output('target-nav-next', 'disabled'),
        
        Input('target-preview-clicked', 'data'),
        State('chromatogram-preview-filter-bookmark', 'value'),
        State('chromatogram-preview-filter-ms-type', 'value'),
        State('chromatogram-preview-order', 'value'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def update_target_nav_on_modal_open(target_clicked, bookmark_filter, ms_type_filter, order_by, wdir):
        """Populate navigation store when modal opens."""
        if not target_clicked or not wdir:
            raise PreventUpdate
        
        try:
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                    
                # Get ordered list of targets matching current filters
                filters = []
                params = []
                
                if bookmark_filter == 'Bookmarked':
                    filters.append("bookmark = true")
                elif bookmark_filter == 'Unmarked':
                    filters.append("(bookmark = false OR bookmark IS NULL)")
                
                if ms_type_filter and ms_type_filter != 'All':
                    filters.append("ms_type = ?")
                    params.append(ms_type_filter.lower())
                
                where_clause = "WHERE " + " AND ".join(filters) if filters else ""
                order_clause = f"ORDER BY {order_by} ASC" if order_by else "ORDER BY mz_mean ASC"
                
                query = f"SELECT peak_label FROM targets {where_clause} {order_clause}"
                targets = [row[0] for row in conn.execute(query, params).fetchall()]
        except Exception as e:
            logger.warning(f"Navigation store update failed for target '{target_clicked}' (possibly due to rapid navigation): {e}")
            raise PreventUpdate
        
        if not targets:
            return {'targets': [], 'current_index': 0}, "0 / 0", True, True
        
        try:
            current_index = targets.index(target_clicked)
        except ValueError:
            current_index = 0
        
        total = len(targets)
        counter_text = f"{current_index + 1} / {total}"
        prev_disabled = current_index == 0
        next_disabled = current_index >= total - 1
        
        return {'targets': targets, 'current_index': current_index}, counter_text, prev_disabled, next_disabled

    @app.callback(
        Output('pending-nav-direction', 'data'),
        Output('confirm-nav-modal', 'visible'),
        Output('target-preview-clicked', 'data', allow_duplicate=True),
        
        Input('target-nav-prev', 'nClicks'),
        Input('target-nav-next', 'nClicks'),
        State('target-nav-store', 'data'),
        State('slider-reference-data', 'data'),
        State('slider-data', 'data'),
        State('target-note', 'value'),  # Current note text
        State('chromatogram-view-modal', 'title'),  # Current target name
        State('wdir', 'data'),
        State('chromatogram-view-rt-align', 'checked'),  # RT alignment toggle state
        State('rt-alignment-data', 'data'),  # RT alignment calculation data
        prevent_initial_call=True
    )
    def navigate_targets(prev_clicks, next_clicks, nav_store, reference_data, slider_data, 
                         current_note, current_target, wdir, rt_align_toggle, rt_alignment_data):
        """Handle Previous/Next button clicks with unsaved changes check."""
        if not nav_store or not nav_store.get('targets'):
            raise PreventUpdate
        
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        targets = nav_store['targets']
        current_index = nav_store['current_index']
        
        if trigger == 'target-nav-prev' and prev_clicks:
            direction = 'prev'
            new_index = max(0, current_index - 1)
        elif trigger == 'target-nav-next' and next_clicks:
            direction = 'next'
            new_index = min(len(targets) - 1, current_index + 1)
        else:
            raise PreventUpdate
        
        # Auto-save notes and RT alignment before navigating
        if wdir and current_target:
            try:
                with duckdb_connection(wdir) as conn:
                    if conn is not None:
                        # Save notes
                        conn.execute("UPDATE targets SET notes = ? WHERE peak_label = ?",
                                    (current_note or '', current_target))
                        logger.debug(f"Auto-saved notes for '{current_target}' before navigation")
                        
                        # Auto-save RT-span if changed
                        if slider_data:
                            slider_value = slider_data.get('value') if isinstance(slider_data, dict) else None
                            if slider_value and isinstance(slider_value, dict):
                                rt_min = slider_value.get('rt_min')
                                rt_max = slider_value.get('rt_max')
                                rt = slider_value.get('rt', (rt_min + rt_max) / 2 if rt_min is not None and rt_max is not None else None)
                                
                                if rt_min is not None and rt_max is not None:
                                    conn.execute("""
                                        UPDATE targets 
                                        SET rt_min = ?, rt_max = ?, rt = ?
                                        WHERE peak_label = ?
                                    """, [rt_min, rt_max, rt, current_target])
                                    logger.info(f"Auto-saved RT-span for '{current_target}': [{rt_min:.2f}, {rt_max:.2f}]")
                        
                        # Save RT alignment state
                        if rt_align_toggle:
                            # Toggle is ON - save alignment data
                            if rt_alignment_data and rt_alignment_data.get('enabled'):
                                import json
                                shifts_json = json.dumps(rt_alignment_data.get('shifts_per_file', {}))
                                conn.execute("""
                                    UPDATE targets 
                                    SET rt_align_enabled = TRUE,
                                        rt_align_reference_rt = ?,
                                        rt_align_shifts = ?,
                                        rt_align_rt_min = ?,
                                        rt_align_rt_max = ?
                                    WHERE peak_label = ?
                                """, [
                                    rt_alignment_data['reference_rt'],
                                    shifts_json,
                                    rt_alignment_data.get('rt_min'),
                                    rt_alignment_data.get('rt_max'),
                                    current_target
                                ])
                                logger.debug(f"Auto-saved RT alignment for '{current_target}' before navigation")
                        else:
                            # Toggle is OFF - clear alignment data
                            conn.execute("""
                                UPDATE targets 
                                SET rt_align_enabled = FALSE,
                                    rt_align_reference_rt = NULL,
                                    rt_align_shifts = NULL,
                                    rt_align_rt_min = NULL,
                                    rt_align_rt_max = NULL
                                WHERE peak_label = ?
                            """, [current_target])
                            logger.debug(f"Cleared RT alignment for '{current_target}' before navigation")
            except Exception as e:
                logger.warning(f"Failed to auto-save data for '{current_target}': {e}")
        
        # Navigate directly (no more confirmation modal)
        new_target = targets[new_index]
        return None, False, new_target

    @app.callback(
        Output('target-preview-clicked', 'data', allow_duplicate=True),
        Output('confirm-nav-modal', 'visible', allow_duplicate=True),
        
        Input('confirm-nav-modal', 'okCounts'),
        State('pending-nav-direction', 'data'),
        State('target-nav-store', 'data'),
        State('target-note', 'value'),  # Current note text
        State('chromatogram-view-modal', 'title'),  # Current target name
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def confirm_navigation(ok_counts, direction, nav_store, current_note, current_target, wdir):
        """Navigate after user confirms discarding unsaved changes."""
        if not ok_counts or not direction or not nav_store:
            raise PreventUpdate
        
        # Auto-save notes before navigating (even when discarding RT-span changes)
        if wdir and current_target:
            try:
                with duckdb_connection(wdir) as conn:
                    if conn is not None:
                        conn.execute("UPDATE targets SET notes = ? WHERE peak_label = ?",
                                    (current_note or '', current_target))
                        logger.debug(f"Auto-saved notes for '{current_target}' on confirm navigation")
            except Exception as e:
                logger.warning(f"Failed to auto-save notes for '{current_target}': {e}")
        
        targets = nav_store['targets']
        current_index = nav_store['current_index']
        
        if direction == 'prev':
            new_index = max(0, current_index - 1)
        elif direction == 'next':
            new_index = min(len(targets) - 1, current_index + 1)
        else:
            raise PreventUpdate
        
        new_target = targets[new_index]
        return new_target, False

    @app.callback(
        Output('chromatogram-preview-pagination', 'current', allow_duplicate=True),
        
        Input('chromatogram-view-modal', 'visible'),
        State('target-nav-store', 'data'),
        State('chromatogram-preview-pagination', 'pageSize'),
        prevent_initial_call=True
    )
    def sync_pagination_on_modal_close(modal_visible, nav_store, page_size):
        """When modal closes, navigate to the page containing the current target."""
        # Only trigger when modal is being closed (visible becomes False)
        if modal_visible:
            raise PreventUpdate
        
        if not nav_store or not nav_store.get('targets'):
            raise PreventUpdate
        
        current_index = nav_store.get('current_index', 0)
        
        # Calculate which page the current target is on (1-indexed)
        page_size = page_size or 9  # Default pageSize
        new_page = (current_index // page_size) + 1
        
        return new_page

    @app.callback(
        Output('notifications-container', 'children', allow_duplicate=True),
        
        Input('chromatogram-view-modal', 'visible'),
        State('slider-data', 'data'),
        State('slider-reference-data', 'data'),
        State('target-note', 'value'),
        State('chromatogram-view-modal', 'title'),  # Current target name
        State('chromatogram-view-rt-align', 'checked'),
        State('rt-alignment-data', 'data'),
        State('wdir', 'data'),
        prevent_initial_call=True
    )
    def auto_save_on_modal_close(modal_visible, slider_data, reference_data, 
                                   current_note, current_target, rt_align_toggle, 
                                   rt_alignment_data, wdir):
        """Auto-save RT-span, notes, and RT alignment when modal closes."""
        # Only trigger when modal is being closed (visible becomes False)
        if modal_visible:
            raise PreventUpdate
        
        if not wdir or not current_target:
            raise PreventUpdate
        
        saved_items = []
        
        try:
            with duckdb_connection(wdir) as conn:
                if conn is None:
                    raise PreventUpdate
                
                # Auto-save RT-span if changed
                if slider_data and reference_data:
                    slider_value = slider_data.get('value') if isinstance(slider_data, dict) else None
                    reference_value = reference_data.get('value') if isinstance(reference_data, dict) else None
                    
                    if slider_value and isinstance(slider_value, dict):
                        # Check if values changed
                        if reference_value is None or slider_value != reference_value:
                            rt_min = slider_value.get('rt_min')
                            rt_max = slider_value.get('rt_max')
                            rt = slider_value.get('rt', (rt_min + rt_max) / 2 if rt_min and rt_max else None)
                            
                            if rt_min is not None and rt_max is not None:
                                conn.execute("""
                                    UPDATE targets 
                                    SET rt_min = ?, rt_max = ?, rt = ?
                                    WHERE peak_label = ?
                                """, [rt_min, rt_max, rt, current_target])
                                logger.info(f"Auto-saved RT-span for '{current_target}' on modal close")
                                saved_items.append("RT-span")

                
                # Save notes
                conn.execute("UPDATE targets SET notes = ? WHERE peak_label = ?",
                            (current_note or '', current_target))
                
                # Save RT alignment state
                if rt_align_toggle:
                    if rt_alignment_data and rt_alignment_data.get('enabled'):
                        import json
                        shifts_json = json.dumps(rt_alignment_data.get('shifts_per_file', {}))
                        conn.execute("""
                            UPDATE targets 
                            SET rt_align_enabled = TRUE,
                                rt_align_reference_rt = ?,
                                rt_align_shifts = ?,
                                rt_align_rt_min = ?,
                                rt_align_rt_max = ?
                            WHERE peak_label = ?
                        """, [
                            rt_alignment_data['reference_rt'],
                            shifts_json,
                            rt_alignment_data.get('rt_min'),
                            rt_alignment_data.get('rt_max'),
                            current_target
                        ])
                        saved_items.append("RT alignment")
                else:
                    conn.execute("""
                        UPDATE targets 
                        SET rt_align_enabled = FALSE,
                            rt_align_reference_rt = NULL,
                            rt_align_shifts = NULL,
                            rt_align_rt_min = NULL,
                            rt_align_rt_max = NULL
                        WHERE peak_label = ?
                    """, [current_target])
                
                logger.debug(f"Auto-saved data for '{current_target}' on modal close")
        
        except Exception as e:
            logger.warning(f"Failed to auto-save data for '{current_target}' on modal close: {e}")
            return dash.no_update
        
        # Only show notification if RT-span was changed
        if saved_items:
            return fac.AntdNotification(
                message="Changes auto-saved",
                description=f"Saved: {', '.join(saved_items)} for {current_target}",
                type="success",
                duration=2,
                placement="bottom"
            )
        
        return dash.no_update

    # Clientside callback to handle arrow key navigation
    # This simulates clicking the prev/next buttons when arrow keys are pressed
    app.clientside_callback(
        """
        function(modalVisible) {
            if (!modalVisible) {
                // Remove listener when modal is closed
                if (window._mintKeyHandler) {
                    document.removeEventListener('keydown', window._mintKeyHandler);
                    window._mintKeyHandler = null;
                }
                return window.dash_clientside.no_update;
            }
            
            // Add keyboard listener when modal opens
            if (!window._mintKeyHandler) {
                window._mintKeyHandler = function(e) {
                    // Only handle arrow keys when not in an input/textarea
                    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                        return;
                    }
                    
                    if (e.key === 'ArrowLeft') {
                        const prevBtn = document.getElementById('target-nav-prev');
                        if (prevBtn && !prevBtn.disabled) {
                            prevBtn.click();
                        }
                    } else if (e.key === 'ArrowRight') {
                        const nextBtn = document.getElementById('target-nav-next');
                        if (nextBtn && !nextBtn.disabled) {
                            nextBtn.click();
                        }
                    }
                };
                document.addEventListener('keydown', window._mintKeyHandler);
            }
            
            return window.dash_clientside.no_update;
        }
        """,
        Output('keyboard-nav-trigger', 'data'),
        Input('chromatogram-view-modal', 'visible')
    )

    # Spinner timeout: Enable interval and record start time when spinner starts
    @app.callback(
        Output('spinner-start-time', 'data'),
        Output('spinner-timeout-interval', 'disabled'),
        Input('chromatogram-view-spin', 'spinning'),
        prevent_initial_call=True
    )
    def manage_spinner_timeout(spinning):
        import time as time_module
        if spinning:
            # Spinner started - record time and enable interval
            return time_module.time(), False
        else:
            # Spinner stopped - disable interval
            return None, True

    # Check if spinner has been running too long and reset it
    @app.callback(
        Output('chromatogram-view-spin', 'spinning', allow_duplicate=True),
        Output('spinner-timeout-interval', 'disabled', allow_duplicate=True),
        Input('spinner-timeout-interval', 'n_intervals'),
        State('spinner-start-time', 'data'),
        State('chromatogram-view-spin', 'spinning'),
        prevent_initial_call=True
    )
    def reset_stuck_spinner(n_intervals, start_time, is_spinning):
        import time as time_module
        if not start_time or not is_spinning:
            raise PreventUpdate
        
        elapsed = time_module.time() - start_time
        if elapsed > 8:  # 8 second timeout
            logger.warning(f"Spinner was stuck for {elapsed:.1f}s - forcing reset")
            return False, True  # Stop spinning and disable interval
        
        raise PreventUpdate
