import logging
import dash
import base64
from io import BytesIO, StringIO
from pathlib import Path
import time
import feffery_antd_components as fac
import feffery_utils_components as fuc
from itertools import cycle
import numpy as np
import pandas as pd
from dash import html, dcc, ALL, MATCH
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from ..pca import SciPyPCA as PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly import colors as plotly_colors
from ..duckdb_manager import duckdb_connection, create_pivot, get_physical_cores
import os
from ..plugin_interface import PluginInterface
import plotly.express as px
from scipy.stats import ttest_ind, f_oneway
from sklearn.manifold import TSNE
from ..sample_metadata import GROUP_COLUMNS, GROUP_LABELS
from .scalir import (
    intersect_peaks,
    fit_estimator,
    build_concentration_table,
    training_plot_frame,
    plot_standard_curve,
    slugify_label,
)

_label = "Analysis"

logger = logging.getLogger(__name__)


def load_persisted_scalir_results(wdir: str) -> dict | None:
    """
    Load persisted SCALiR results from workspace if they exist.
    
    Returns:
        Dictionary with results data if files exist, None otherwise
    """
    output_dir = Path(wdir) / "results" / "scalir"
    train_frame_path = output_dir / "train_frame.csv"
    params_path = output_dir / "standard_curve_parameters.csv"
    concentrations_path = output_dir / "concentrations.csv"
    units_path = output_dir / "units.csv"
    plots_dir = output_dir / "plots"
    
    # Check if essential files exist
    if not train_frame_path.exists() or not params_path.exists():
        return None
    
    try:
        train_frame = pd.read_csv(train_frame_path)
        params = pd.read_csv(params_path)
        units = pd.read_csv(units_path) if units_path.exists() else None
        
        # Get list of metabolites from params
        common = params['peak_label'].tolist() if 'peak_label' in params.columns else []
        
        return {
            "train_frame": train_frame.to_json(orient="split"),
            "units": units.to_json(orient="split") if units is not None else None,
            "params": params.to_json(orient="split"),
            "plot_dir": str(plots_dir),
            "common": common,
            "generated_all_plots": plots_dir.exists() and any(plots_dir.iterdir()) if plots_dir.exists() else False,
            "concentrations_path": str(concentrations_path),
            "params_path": str(params_path),
        }
    except Exception as e:
        logger.warning(f"Failed to load persisted SCALiR results: {e}")
        return None

PCA_COMPONENT_OPTIONS = [
    {'label': f'PC{i}', 'value': f'PC{i}'}
    for i in range(1, 6)
]
TSNE_COMPONENT_OPTIONS = [
    {'label': f't-SNE-{i}', 'value': f't-SNE-{i}'}
    for i in range(1, 4)
]
NORM_OPTIONS = [
    {'label': 'None (raw)', 'value': 'none'},
    {'label': 'Z-score', 'value': 'zscore'},
    {'label': 'Rocke-Durbin', 'value': 'durbin'},
    {'label': 'Z-score + Rocke-Durbin', 'value': 'zscore_durbin'},
]
TAB_DEFAULT_NORM = {
    'clustermap': 'zscore',
    'pca': 'durbin',
    'tsne': 'none',
    'raincloud': 'durbin',
    'bar': 'durbin',
}
GROUPING_FIELDS = ['sample_type'] + GROUP_COLUMNS
GROUP_SELECT_OPTIONS = [
    {'label': GROUP_LABELS.get(field, field.replace('_', ' ').title()), 'value': field}
    for field in GROUPING_FIELDS
]

METRIC_OPTIONS = [
    {'label': 'Peak Area', 'value': 'peak_area'},
    {'label': 'Peak Area (Top 3)', 'value': 'peak_area_top3'},
    {'label': 'Peak Max', 'value': 'peak_max'},
    {'label': 'Peak Mean', 'value': 'peak_mean'},
    {'label': 'Peak Median', 'value': 'peak_median'},
    {'label': 'Peak Area (EMG Fitted)', 'value': 'peak_area_fitted'},
    {'label': 'Concentration', 'value': 'scalir_conc'},
]

# High-resolution export configuration for Plotly graphs
# Scale of 4 provides ~300 DPI (default is 72 DPI)
PLOTLY_HIGH_RES_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'scale': 4,  # 4x scale ≈ 300 DPI
        'height': None,  # maintains aspect ratio
        'width': None,   # maintains aspect ratio
    },
    'displayModeBar': True,
    'displaylogo': False,
    'responsive': True,
}


class AnalysisPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 9
        logger.info(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)

    def outputs(self):
        return _outputs


def rocke_durbin(df: pd.DataFrame, c: float) -> pd.DataFrame:
    # df: samples x features (metabolites)
    z = df.to_numpy(dtype=float)
    ef = np.log((z + np.sqrt(z ** 2 + c ** 2)) / 2.0)
    return pd.DataFrame(ef, index=df.index, columns=df.columns)


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
    import math
    ys_all = []
    if not data:
        return None
        
    for trace in data:
        xs = trace.get('x')
        ys = trace.get('y')
        
        if xs is None or ys is None or len(xs) == 0:
            continue
            
        try:
            xs = np.array(xs, dtype=np.float64)
            ys = np.array(ys, dtype=np.float64)
        except Exception:
            continue
        
        mask = (xs >= x_left) & (xs <= x_right)
        ys_filtered = ys[mask]
        
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
        pos_mask = ys_concat > 1
        ys_pos = ys_concat[pos_mask]
        if len(ys_pos) == 0:
            return None
        return [math.log10(np.min(ys_pos)), math.log10(np.max(ys_pos) * 1.05)]

    y_min = np.min(ys_concat)
    y_max = np.max(ys_concat)
    y_min = 0 if y_min > 0 else y_min
    return [y_min, y_max * 1.05]

def run_pca_samples_in_cols(df: pd.DataFrame, n_components=None, random_state=0):
    """Run PCA with samples in rows and targets in columns."""

    X = df.to_numpy(dtype=float)  # samples x targets

    pca = PCA(n_components=n_components, random_state=random_state)
    scores = pca.fit_transform(X)  # samples x components

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    pc_labels = [f"PC{i + 1}" for i in range(pca.n_components_)]

    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_), # https://stats.stackexchange.com/questions/143905/loadings-vs-eigenvectors-in-pca-when-to-use-one-or-another
        columns=pc_labels,
        index=df.columns,
    )
    scores_df = pd.DataFrame(
        scores,
        index=df.index,
        columns=pc_labels,
    )

    return {
        "pca": pca,
        "scores": scores_df,
        "loadings": loadings,
        "explained_variance_ratio": pd.Series(explained, index=pc_labels),
        "cumulative_variance_ratio": pd.Series(cumulative, index=pc_labels),
    }


clustermap_tab = html.Div([
    fac.AntdFlex(
        [
            # Left side: Options panel
            html.Div(
                [
                    fac.AntdText("Clustering", strong=True, style={'fontSize': '14px', 'marginBottom': '16px', 'display': 'block'}),
                    fac.AntdFlex(
                        [
                            fac.AntdSwitch(
                                id='clustermap-cluster-rows',
                                checked=True,
                                checkedChildren='On',
                                unCheckedChildren='Off',
                            ),
                            fac.AntdText("Cluster Rows", style={'marginLeft': '8px'}),
                        ],
                        align='center',
                        style={'marginBottom': '24px'},
                    ),
                    fac.AntdFlex(
                        [
                            fac.AntdSwitch(
                                id='clustermap-cluster-cols',
                                checked=False,
                                checkedChildren='On',
                                unCheckedChildren='Off',
                            ),
                            fac.AntdText("Cluster Columns", style={'marginLeft': '8px'}),
                        ],
                        align='center',
                        style={'marginBottom': '8px'},
                    ),
                    fac.AntdDivider(style={'margin': '16px 0'}),
                    fac.AntdText("Fontsize", strong=True, style={'fontSize': '14px', 'marginBottom': '16px', 'display': 'block'}),
                    fac.AntdText("X-axis:", style={'fontWeight': 500, 'fontSize': '12px', 'display': 'block', 'marginBottom': '8px'}),
                    fac.AntdSlider(
                        id='clustermap-fontsize-x-slider',
                        min=0,
                        max=20,
                        step=1,
                        value=5,
                        marks={0: '0', 10: '10', 20: '20'},
                        style={'width': '100%', 'marginBottom': '24px'},
                    ),
                    fac.AntdText("Y-axis:", style={'fontWeight': 500, 'fontSize': '12px', 'display': 'block', 'marginBottom': '8px'}),
                    fac.AntdSlider(
                        id='clustermap-fontsize-y-slider',
                        min=0,
                        max=20,
                        step=1,
                        value=5,
                        marks={0: '0', 10: '10', 20: '20'},
                        style={'width': '100%', 'marginBottom': '24px'},
                    ),
                    fac.AntdFlex(
                        [
                            fac.AntdButton(
                                "Regenerate",
                                id='clustermap-regenerate-btn',
                                type='default',
                                style={'flex': '1'},
                            ),
                            fac.AntdTooltip(
                                fac.AntdIcon(
                                    icon='antd-question-circle',
                                    style={'marginLeft': '8px', 'color': 'gray', 'fontSize': '14px'}
                                ),
                                title='Regenerate the clustermap with current settings',
                                placement='right'
                            ),
                        ],
                        align='center',
                        style={'marginBottom': '16px'},
                    ),
                    fac.AntdDivider(style={'margin': '16px 0'}),
                    fac.AntdFlex(
                        [
                            fac.AntdButton(
                                "Save PNG",
                                id='clustermap-save-png-btn',
                                type='default',
                                style={'flex': '1'},
                            ),
                            fac.AntdTooltip(
                                fac.AntdIcon(
                                    icon='antd-question-circle',
                                    style={'marginLeft': '8px', 'color': 'gray', 'fontSize': '14px'}
                                ),
                                title='Download the clustermap as a PNG image',
                                placement='right'
                            ),
                        ],
                        align='center',
                    ),
                    dcc.Download(id='clustermap-download'),
                ],
                style={
                    'width': '250px',
                    'minWidth': '250px',
                    'padding': '16px',
                    'flexShrink': 0,
                },
            ),
            # Right side: Clustermap image
            html.Div(
                fac.AntdSpin(
                    html.Div(
                        fac.AntdImage(
                            id='bar-graph-matplotlib',
                            preview={'mask': 'Click to Zoom'},
                            locale='en-us',
                            fallback='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
                            style={
                                'maxWidth': '100%',
                                'maxHeight': 'calc(100vh - 200px)',
                                'objectFit': 'contain',
                                'cursor': 'zoom-in',
                            },
                        ),
                        style={
                            'display': 'flex',
                            'justifyContent': 'center',
                            'alignItems': 'center',
                            'width': '100%',
                        },
                    ),
                    id='clustermap-spinner',
                    spinning=True,
                    text='Loading clustermap...',
                    style={
                        'minHeight': 'calc(100vh - 250px)',
                        'width': '100%',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                    },
                ),
                style={
                    'flex': '1',
                    'width': '100%',
                    'minHeight': 'calc(100vh - 200px)',
                },
            ),
        ],
        style={'height': '100%'},
    ),
], style={'height': '100%'})
pca_tab = html.Div(
    [
        # PCA-specific controls: X axis and Y axis
        fac.AntdFlex(
            [
                fac.AntdSpace(
                    [
                        html.Span("X axis:", style={'fontWeight': 500}),
                        fac.AntdSelect(
                            id='pca-x-comp',
                            options=PCA_COMPONENT_OPTIONS,
                            value='PC1',
                            allowClear=False,
                            style={'width': 100},
                        ),
                    ],
                    align='center',
                    size='small',
                ),
                fac.AntdSpace(
                    [
                        html.Span("Y axis:", style={'fontWeight': 500}),
                        fac.AntdSelect(
                            id='pca-y-comp',
                            options=PCA_COMPONENT_OPTIONS,
                            value='PC2',
                            allowClear=False,
                            style={'width': 100},
                        ),
                    ],
                    align='center',
                    size='small',
                ),
            ],
            gap='middle',
            align='center',
            style={'marginBottom': '12px'},
        ),
        fac.AntdSpin(
            dcc.Graph(
                id='pca-graph',
                config=PLOTLY_HIGH_RES_CONFIG,
                style={'height': 'calc(100vh - 220px)', 'width': '100%', 'minHeight': '400px'},
                # Start with invisible figure to show only spinner during loading
                figure={
                    'data': [],
                    'layout': {
                        'xaxis': {'visible': False},
                        'yaxis': {'visible': False},
                        'paper_bgcolor': 'white',
                        'plot_bgcolor': 'white',
                        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                        'autosize': True,
                    }
                },
            ),
            text='Loading PCA...',
            style={'minHeight': '20vh', 'width': '100%'},
        ),
    ],
    style={'height': '100%'},
)

tsne_tab = html.Div(
    [
        # t-SNE-specific controls
        fac.AntdFlex(
            [
                fac.AntdSpace(
                    [
                        html.Span("X axis:", style={'fontWeight': 500}),
                        fac.AntdSelect(
                            id='tsne-x-comp',
                            options=TSNE_COMPONENT_OPTIONS,
                            value='t-SNE-1',
                            allowClear=False,
                            style={'width': 100},
                        ),
                    ],
                    align='center',
                    size='small',
                ),
                fac.AntdSpace(
                    [
                        html.Span("Y axis:", style={'fontWeight': 500}),
                        fac.AntdSelect(
                            id='tsne-y-comp',
                            options=TSNE_COMPONENT_OPTIONS,
                            value='t-SNE-2',
                            allowClear=False,
                            style={'width': 100},
                        ),
                    ],
                    align='center',
                    size='small',
                ),
                fac.AntdDivider(direction='vertical', style={'height': '24px', 'margin': '0 12px'}),
                fac.AntdSpace(
                    [
                        fac.AntdText("Perplexity:", style={'fontWeight': 500}),
                        fac.AntdSlider(
                            id='tsne-perplexity-slider',
                            min=1,
                            max=100,
                            step=1,
                            value=30,
                            style={'width': '200px'},
                            tooltipPrefix='Perplexity: '
                        ),
                    ],
                    align='center',
                    size='small',
                ),
                fac.AntdButton(
                    "Generate",
                    id="tsne-regenerate-btn",
                    icon=fac.AntdIcon(icon="antd-sync"),
                    style={'textTransform': 'uppercase'},
                ),
            ],
            gap='middle',
            align='center',
            style={'marginBottom': '12px'},
        ),
        fac.AntdSpin(
            dcc.Graph(
                id='tsne-graph',
                config=PLOTLY_HIGH_RES_CONFIG,
                style={'height': 'calc(100vh - 220px)', 'width': '80%', 'minHeight': '400px', 'margin': '0 auto'},
                figure={
                    'data': [],
                    'layout': {
                        'xaxis': {'visible': False},
                        'yaxis': {'visible': False},
                        'paper_bgcolor': 'white',
                        'plot_bgcolor': 'white',
                        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                        'autosize': True,
                    }
                },
            ),
            text='Loading t-SNE...',
            style={'minHeight': '20vh', 'width': '100%'},
        ),
    ],
    style={'height': '100%'},
)

# SCALiR tab has been moved to Processing plugin as a modal workflow

# Violin tab content (extracted for reuse in sidebar layout)
violin_content = html.Div(
    [
        fac.AntdFlex(
            [
                fac.AntdText('Target to display', strong=True),
                fac.AntdSelect(
                    id='violin-comp-checks',
                    options=[],
                    value=None,
                    allowClear=False,
                    optionFilterProp='label',
                    optionFilterMode='case-insensitive',
                    style={'width': '320px'},
                ),
            ],
            align='center',
            gap='small',
            wrap=True,
            style={'paddingBottom': '0.75rem'},
        ),

        # Main content: violin plot on left, chromatogram on right
        fac.AntdFlex(
            [
                # Violin plot container (left side)
                html.Div(
                    fac.AntdSpin(
                        html.Div(
                            id='violin-graphs',
                            style={
                                'display': 'flex',
                                'flexDirection': 'column',
                                'gap': '24px',
                            },
                        ),
                        id='violin-spinner',
                        spinning=True,
                        text='Loading Violin...',
                        style={'minHeight': '300px', 'width': '100%'},
                    ),
                    style={'width': 'calc(55% - 6px)', 'height': '450px', 'overflowY': 'auto'},
                ),
                # Chromatogram container (right side)
                html.Div(
                    [
                        fac.AntdSpin(
                            dcc.Graph(
                                id='violin-chromatogram',
                                config={'displayModeBar': True, 'responsive': True},
                                style={'height': '450px', 'width': '100%'},
                            ),
                            text='Loading Chromatogram...',
                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdText("Log2 Scale", style={'marginRight': '8px', 'fontSize': '12px'}),
                                fac.AntdSwitch(
                                    id='violin-log-scale-switch',
                                    checked=False,
                                    checkedChildren='On',
                                    unCheckedChildren='Off',
                                    size='small',
                                ),
                            ],
                            justify='end',
                            align='center',
                            style={'marginTop': '4px', 'width': '100%'}
                        )
                    ],
                    id='violin-chromatogram-container',
                    style={'display': 'block', 'width': 'calc(43% - 6px)', 'height': 'auto'} # Allow height to grow
                ),
            ],
            gap='middle',
            wrap=False,
            justify='center',
            align='center',
            style={'width': '100%', 'height': 'calc(100vh - 200px)'},
        ),
    ],
    id='analysis-violin-content',
)

# Store to track the currently selected sample for violin highlighting (defined outside for callback access)
violin_selected_sample_store = dcc.Store(id='violin-selected-sample', data=None)

# Bar tab content (similar to violin)
bar_content = html.Div(
    [
        fac.AntdFlex(
            [
                fac.AntdText('Target to display', strong=True),
                fac.AntdSelect(
                    id='bar-comp-checks',
                    options=[],
                    value=None,
                    allowClear=False,
                    optionFilterProp='label',
                    optionFilterMode='case-insensitive',
                    style={'width': '320px'},
                ),
            ],
            align='center',
            gap='small',
            wrap=True,
            style={'paddingBottom': '0.75rem'},
        ),

        # Main content: bar plot on left, chromatogram on right
        fac.AntdFlex(
            [
                # Bar plot container (left side)
                html.Div(
                    fac.AntdSpin(
                        html.Div(
                            id='bar-graphs',
                            style={
                                'display': 'flex',
                                'flexDirection': 'column',
                                'gap': '24px',
                            },
                        ),
                        id='bar-spinner',
                        spinning=True,
                        text='Loading Bar Plot...',
                        style={'minHeight': '300px', 'width': '100%'},
                    ),
                    style={'width': 'calc(55% - 6px)', 'height': '450px', 'overflowY': 'auto'},
                ),
                # Chromatogram container (right side)
                html.Div(
                    [
                        fac.AntdSpin(
                            dcc.Graph(
                                id='bar-chromatogram',
                                config={'displayModeBar': True, 'responsive': True},
                                style={'height': '450px', 'width': '100%'},
                            ),
                            text='Loading Chromatogram...',
                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdText("Log2 Scale", style={'marginRight': '8px', 'fontSize': '12px'}),
                                fac.AntdSwitch(
                                    id='bar-log-scale-switch',
                                    checked=False,
                                    checkedChildren='On',
                                    unCheckedChildren='Off',
                                    size='small',
                                ),
                            ],
                            justify='end',
                            align='center',
                            style={'marginTop': '4px', 'width': '100%'}
                        )
                    ],
                    id='bar-chromatogram-container',
                    style={'display': 'block', 'width': 'calc(43% - 6px)', 'height': 'auto'} # Allow height to grow
                ),
            ],
            gap='middle',
            wrap=False,
            justify='center',
            align='center',
            style={'width': '100%', 'height': 'calc(100vh - 200px)'},
        ),
    ],
    id='analysis-bar-content',
)

# Store to track the currently selected sample for bar highlighting (defined outside for callback access)
bar_selected_sample_store = dcc.Store(id='bar-selected-sample', data=None)

# QC tab content (Quality Control scatter plots over acquisition time)
qc_content = html.Div(
    [
        fac.AntdFlex(
            [
                # Left side: Stacked QC Scatter Plots
                html.Div(
                    fac.AntdSpin(
                        fac.AntdFlex(
                            [
                                fac.AntdDivider(
                                    children="RT",
                                    lineColor="#ccc",
                                    fontColor="#444",
                                    fontSize="14px",
                                    style={'margin': '12px 0'}
                                ),

                                html.Div([
                                    dcc.Graph(
                                        id='qc-rt-graph',
                                        config=PLOTLY_HIGH_RES_CONFIG,
                                        style={'height': '280px', 'width': '100%'},
                                        figure={
                                            'data': [],
                                            'layout': {
                                                'xaxis': {'visible': False},
                                                'yaxis': {'visible': False},
                                                'paper_bgcolor': 'white',
                                                'plot_bgcolor': 'white',
                                                'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                                                'autosize': True,
                                            }
                                        },
                                    )
                                ], style={'height': '280px', 'width': '100%', 'display': 'block'}),
                                fac.AntdDivider(
                                    children="Peak Area",
                                    lineColor="#ccc",
                                    fontColor="#444",
                                    fontSize="14px",
                                    style={'margin': '12px 0'}
                                ),

                                html.Div([
                                    dcc.Graph(
                                        id='qc-mz-graph',
                                        config=PLOTLY_HIGH_RES_CONFIG,
                                        style={'height': '280px', 'width': '100%'},
                                        figure={
                                            'data': [],
                                            'layout': {
                                                'xaxis': {'visible': False},
                                                'yaxis': {'visible': False},
                                                'paper_bgcolor': 'white',
                                                'plot_bgcolor': 'white',
                                                'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                                                'autosize': True,
                                            }
                                        },
                                    )
                                ], style={'height': '280px', 'width': '100%', 'display': 'block'}),
                            ],
                            vertical=True,
                            style={'width': '100%'}
                        ),
                        id='qc-spinner',
                        spinning=True,
                        text='Loading QC plots...',
                        style={'minHeight': '300px', 'width': '100%'},
                    ),
                    style={'width': 'calc(55% - 6px)', 'height': '100%', 'overflowY': 'auto'},
                ),
                
                # Right side: Chromatogram
                html.Div(
                    [
                        fac.AntdSpin(
                            dcc.Graph(
                                id='qc-chromatogram',
                                config={'displayModeBar': True, 'responsive': True},
                                style={'height': '100%', 'width': '100%'},
                            ),
                            text='Loading Chromatogram...',
                        ),
                        fac.AntdFlex(
                            [
                                fac.AntdText("Log2 Scale", style={'marginRight': '8px', 'fontSize': '12px'}),
                                fac.AntdSwitch(
                                    id='qc-log-scale-switch',
                                    checked=False,
                                    checkedChildren='On',
                                    unCheckedChildren='Off',
                                    size='small',
                                ),
                            ],
                            justify='end',
                            align='center',
                            style={'marginTop': '4px', 'width': '100%'}
                        ),
                        # Store for QC selection
                        dcc.Store(id='qc-selected-sample', data=None)
                    ],
                    id='qc-chromatogram-container',
                    style={'display': 'block', 'width': 'calc(43% - 6px)', 'marginTop': '65px'}
                ),
            ],
            gap='middle',
            wrap=False,
            justify='center',
            align='center',
            style={'width': '100%', 'height': 'calc(100vh - 200px)'},
        ),
    ],
    id='analysis-qc-content',
)

# Analysis menu items for sidebar
ANALYSIS_MENU_ITEMS = [
    {'component': 'Item', 'props': {'key': 'qc', 'title': 'QC', 'icon': 'antd-check-circle'}},
    {'component': 'Item', 'props': {'key': 'pca', 'title': 'PCA', 'icon': 'antd-dot-chart'}},
    {'component': 'Item', 'props': {'key': 'tsne', 'title': 't-SNE', 'icon': 'antd-deployment-unit'}},
    {'component': 'Item', 'props': {'key': 'raincloud', 'title': 'Violin', 'icon': 'antd-control'}},
    {'component': 'Item', 'props': {'key': 'bar', 'title': 'Bar', 'icon': 'antd-bar-chart'}},
    {'component': 'Item', 'props': {'key': 'clustermap', 'title': 'Clustermap', 'icon': 'antd-build'}},
]

_layout = fac.AntdLayout(
    [
        fac.AntdHeader(
            [
                fac.AntdFlex(
                    [
                        fac.AntdTitle(
                            'Analysis', level=4, style={'margin': '0', 'whiteSpace': 'nowrap'}
                        ),
                        fac.AntdTooltip(
                            fac.AntdIcon(
                                id='analysis-tour-icon',
                                icon='pi-info',
                                style={"cursor": "pointer", 'paddingLeft': '10px'},
                                **{'aria-label': 'Show tutorial'},
                            ),
                            title='Show tutorial'
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
                        fac.AntdFlex(
                            [
                                fac.AntdMenu(
                                    id='analysis-sidebar-menu',
                                    menuItems=ANALYSIS_MENU_ITEMS,
                                    mode='inline',
                                    defaultSelectedKey='qc',
                                    style={'border': 'none'},
                                ),
                            ],
                            vertical=True,
                            style={'height': '100%', 'paddingTop': '8px'}
                        ),
                        html.Div(
                            fac.AntdTooltip(
                                fac.AntdButton(
                                    id='analysis-sidebar-collapse',
                                    type='text',
                                    icon=fac.AntdIcon(
                                        id='analysis-sidebar-collapse-icon',
                                        icon='antd-left',
                                        style={'fontSize': '14px'},
                                    ),
                                    shape='default',
                                    **{'aria-label': 'Collapse/Expand Analysis Sidebar'},
                                ),
                                title='Collapse/Expand Analysis Sidebar'
                            ),
                            style={
                                'position': 'absolute',
                                'zIndex': 1,
                                'right': -8,
                                'bottom': 16,
                                'boxShadow': '2px 2px 5px 1px rgba(0,0,0,0.5)',
                                'background': 'white',
                                'borderRadius': '4px',
                            },
                        ),
                    ],
                    id='analysis-sidebar',
                    collapsible=True,
                    collapsed=False,
                    collapsedWidth=60,
                    width=180,
                    trigger=None,
                    style={'height': '100%', 'background': 'white'},
                    className="sidebar-mint"
                ),
                html.Div(
                    [
                        # Shared controls header - visible for all views
                        fac.AntdFlex(
                            [
                                html.Div(
                                    fac.AntdSpace(
                                        [
                                            fac.AntdText("Metric:", style={'fontWeight': 500, 'marginLeft': '15px'}),
                                            fac.AntdSelect(
                                                id='analysis-metric-select',
                                                options=METRIC_OPTIONS,
                                                value='peak_area',
                                                optionFilterProp='label',
                                                optionFilterMode='case-insensitive',
                                                allowClear=False,
                                                style={'width': 160},
                                            ),
                                        ],
                                        align='center',
                                        size='small',
                                    ),
                                    id='analysis-metric-wrapper',
                                    style={'display': 'flex'},
                                ),
                                html.Div(
                                    fac.AntdSpace(
                                        [
                                            fac.AntdText("Transformations:", style={'fontWeight': 500}),
                                            fac.AntdSelect(
                                                id='analysis-normalization-select',
                                                options=NORM_OPTIONS,
                                                value='zscore',
                                                optionFilterProp='label',
                                                optionFilterMode='case-insensitive',
                                                allowClear=False,
                                                style={'width': 160},
                                            ),
                                        ],
                                        align='center',
                                        size='small',
                                    ),
                                    id='analysis-normalization-wrapper',
                                    style={'display': 'flex'},
                                ),
                                fac.AntdSpace(
                                    [
                                        fac.AntdText("Group by:", style={'fontWeight': 500}),
                                        fac.AntdSelect(
                                            id='analysis-grouping-select',
                                            options=GROUP_SELECT_OPTIONS,
                                            value='sample_type',
                                            allowClear=False,
                                            style={'width': 160},
                                        ),
                                    ],
                                    align='center',
                                    size='small',
                                ),
                                html.Div(
                                    fac.AntdSpace(
                                        [
                                            fac.AntdText("Target:", style={'fontWeight': 500}),
                                            fac.AntdSelect(
                                                id='qc-target-select',
                                                options=[],
                                                value=None,
                                                allowClear=False,
                                                optionFilterProp='label',
                                                optionFilterMode='case-insensitive',
                                                style={'width': 350},
                                            ),
                                        ],
                                        align='center',
                                        size='small',
                                    ),
                                    id='qc-target-wrapper',
                                    # This ID allows potential visibility toggling if needed later
                                    style={'display': 'flex'},
                                ),
                            ],
                            wrap=True,
                            gap='middle',
                            align='center',
                            id='analysis-metric-container',
                            style={'padding': '12px 16px', 'borderBottom': '1px solid #f0f0f0'},
                        ),
                        # QC content (first/default tab)
                        html.Div(
                            qc_content,
                            id='analysis-qc-container',
                            style={'display': 'block', 'padding': '16px'}
                        ),
                        # PCA content
                        html.Div(
                            pca_tab,
                            id='analysis-pca-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # t-SNE content
                        html.Div(
                            tsne_tab,
                            id='analysis-tsne-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # Violin content
                        html.Div(
                            [violin_content, violin_selected_sample_store],
                            id='analysis-violin-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # Bar content
                        html.Div(
                            [bar_content, bar_selected_sample_store],
                            id='analysis-bar-container',
                            style={'display': 'none', 'padding': '16px'}
                        ),
                        # Clustermap content
                        html.Div(
                            clustermap_tab,
                            id='analysis-clustermap-container',
                            style={'display': 'none', 'padding': '16px', 'height': 'calc(100vh - 150px)'}
                        ),
                    ],
                    className='ant-layout-content css-1v28nim',
                    style={'background': 'white', 'overflowY': 'auto', 'height': 'calc(100vh - 80px)'}
                ),
            ],
            style={'height': 'calc(100vh - 100px)', 'background': 'white'},
        ),
        # Hidden store for maintaining tab state compatibility with callbacks
        dcc.Store(id='analysis-tabs', data={'activeKey': 'qc'}),
        fac.AntdTour(
            locale='en-us',
            steps=[],
            id='analysis-tour',
            open=False,
            current=0,
        ),
        html.Div(id="analysis-notifications-container"),
    ]
)

_outputs = None


def layout():
    return _layout


allowed_metrics = {
    'peak_area',
    'peak_area_top3',
    'peak_max',
    'peak_mean',
    'peak_median',
    'scalir_conc',
}


def _parse_uploaded_standards(contents, filename):
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

def _plot_curve_fig(frame: pd.DataFrame, peak_label: str, units: pd.DataFrame = None, params_df: pd.DataFrame = None):
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
        title=dict(text=peak_label, x=0.5, xanchor='center'),
        xaxis=dict(
            title=xlabel,
            type="log",
            ticks="outside",
        ),
        yaxis=dict(
            title=f"{peak_label} intensity (AU)",
            type="log",
            tickformat="~s",
            tickmode="auto",
            nticks=6,
            ticks="outside",
        ),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=60, r=20, t=90, b=60),
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

def _analysis_tour_steps(active_tab: str):
    # Common steps for all views
    common_steps = [
        {
            'title': 'Analysis overview',
            'description': 'Use the controls above to choose a metric, transformation, and grouping.',
        },
        {
            'title': 'Metric',
            'description': 'Pick the quantitative metric to visualize.',
            'targetSelector': '#analysis-metric-select',
        },
        {
            'title': 'Transformations',
            'description': 'Normalize or transform the values before plotting.',
            'targetSelector': '#analysis-normalization-select',
        },
        {
            'title': 'Group by',
            'description': 'Group samples for coloring and summaries.',
            'targetSelector': '#analysis-grouping-select',
        },
        {
            'title': 'Analysis Types',
            'description': 'Switch between PCA, Violin, and Clustermap views.',
            'targetSelector': '#analysis-sidebar-menu',
        },
    ]
    
    # View-specific steps
    if active_tab == 'pca':
        view_steps = [
            {
                'title': 'PCA Axes',
                'description': 'Select which principal components to display on the X and Y axes.',
                'targetSelector': '#pca-x-comp',
            },
            {
                'title': 'PCA Plot',
                'description': 'Interactive scatter plot showing samples in PCA space. Hover for details, use the legend to filter groups.',
                'targetSelector': '#pca-graph',
            },

        ]
    elif active_tab == 'tsne':
        view_steps = [
            {
                'title': 't-SNE Axes',
                'description': 'Select which t-SNE dimensions to display.',
                'targetSelector': '#tsne-x-comp',
            },
            {
                'title': 't-SNE Plot',
                'description': 'Interactive scatter plot showing samples in t-SNE space.',
                'targetSelector': '#tsne-graph',
            },
        ]
    elif active_tab == 'raincloud':
        view_steps = [
            {
                'title': 'Target Selection',
                'description': 'Choose which target compound to display in the violin plot.',
                'targetSelector': '#violin-comp-checks',
            },
            {
                'title': 'Violin Plot',
                'description': 'Shows the distribution of values for each group. Click on individual points to see the chromatogram.',
                'targetSelector': '#violin-graphs',
            },
        ]
    elif active_tab == 'clustermap':
        view_steps = [
            {
                'title': 'Font Size Controls',
                'description': 'Adjust the font size for row and column labels.',
                'targetSelector': '#clustermap-fontsize-x-slider',
            },
            {
                'title': 'Regenerate',
                'description': 'Click to regenerate the clustermap with current settings.',
                'targetSelector': '#clustermap-regenerate-btn',
            },
            {
                'title': 'Clustermap',
                'description': 'Heatmap with hierarchical clustering of samples and targets.',
                'targetSelector': '#bar-graph-matplotlib',
            },
        ]
    else:
        view_steps = []
    
    return common_steps + view_steps


def _build_color_map(color_df: pd.DataFrame, group_col: str) -> dict:
    if not group_col or group_col not in color_df.columns or color_df.empty:
        return {}
    working = color_df[[group_col, 'color']].copy()
    working = working[working[group_col].notna()]
    working['color'] = working['color'].apply(
        lambda c: c if isinstance(c, str) and c.strip() and c.strip() != '#bbbbbb' else None
    )
    color_map = (
        working.dropna(subset=['color'])
        .drop_duplicates(subset=[group_col])
        .set_index(group_col)['color']
        .to_dict()
    )
    missing = [val for val in working[group_col].dropna().unique() if val not in color_map]
    if missing:
        palette = plotly_colors.qualitative.Plotly
        for val, color in zip(missing, cycle(palette)):
            color_map[val] = color
    return color_map

def _clean_numeric(numeric_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = numeric_df.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if cleaned.isna().any().any():
        cleaned = cleaned.fillna(0)
    return cleaned



def _create_pivot_custom(conn, value='peak_area', table='results'):
    """
    Local implementation of create_pivot to ensure table parameter is respected.
    """
    # Get ordered peak labels
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
        ORDER BY ms_type
    """
    df = conn.execute(query).df()
    if df.empty:
        return df
    meta_cols = ['ms_type', 'sample_type', *GROUP_COLUMNS, 'ms_file_label']
    # Start with metadata columns that exist in the dataframe
    final_cols = [col for col in meta_cols if col in df.columns]
    # Add pivot columns (targets) ONLY if they exist in the dataframe
    final_cols.extend([col for col in ordered_pl if col in df.columns])
    return df[final_cols]

def show_tab_content(section_context, tab_key, x_comp, y_comp, violin_comp_checks, bar_comp_checks, metric_value, norm_value,
                    group_by, regen_clicks, tsne_regen_clicks, cluster_rows, cluster_cols, fontsize_x, fontsize_y, wdir,
                    tsne_x_comp, tsne_y_comp, tsne_perplexity):
    if not section_context or section_context.get('page') != 'Analysis':
        raise PreventUpdate
    # Prevent double-firing when switching tabs forces a normalization update.
    # If the tab change triggered this, and the current norm isn't the tab's default,
    # we skip this and wait for the normalization update to trigger the callback again.
    from dash import callback_context
    triggered_props = [t['prop_id'] for t in callback_context.triggered] if callback_context.triggered else []
    if 'analysis-sidebar-menu.currentKey' in triggered_props:
        default_norm = TAB_DEFAULT_NORM.get(tab_key)
        if default_norm and norm_value != default_norm:
            raise PreventUpdate
    if not wdir:
        raise PreventUpdate
    # Early guard: if there are no results yet, return empty placeholders instead of erroring.
    # Create an invisible figure for PCA when there's no data
    invisible_fig = go.Figure()
    invisible_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
        height=10,  # Minimum height allowed by Plotly
    )
    from dash import callback_context
    ctx = callback_context
    triggered_prop = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    grouping_fields = GROUPING_FIELDS
    selected_group = group_by if group_by in grouping_fields else GROUPING_FIELDS[0]
    with duckdb_connection(wdir) as conn:
        if conn is None:
            return None, invisible_fig, invisible_fig, [], [], [], [], [], []
        results_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        if results_count == 0:
            return None, invisible_fig, invisible_fig, [], [], [], [], [], []
        # Robust metric selection
        metric = 'peak_area'
        if metric_value == 'scalir_conc' or metric_value in allowed_metrics:
            metric = metric_value
        
        logger.info(f"DEBUG: metric_value={metric_value}, metric={metric}")
        
        target_table = 'results'
        if metric == 'scalir_conc':
            scalir_path = Path(wdir) / "results" / "scalir" / "concentrations.csv"
            if not scalir_path.exists():
                return None, invisible_fig, invisible_fig, [], [], [], [], [], []
            try:
                conn.execute(f"CREATE OR REPLACE TEMP VIEW scalir_temp_conc AS SELECT * FROM read_csv_auto('{scalir_path}')")
                conn.execute("""
                    CREATE OR REPLACE TEMP VIEW scalir_results_view AS 
                    SELECT 
                        r.ms_file_label, 
                        r.peak_label, 
                        s.pred_conc AS scalir_conc 
                    FROM results r 
                    LEFT JOIN scalir_temp_conc s 
                    ON r.ms_file_label = CAST(s.ms_file AS VARCHAR) AND r.peak_label = s.peak_label
                """)
                target_table = 'scalir_results_view'
            except Exception as e:
                logger.error(f"Error preparing SCALiR data: {e}")
                return None, invisible_fig, invisible_fig, [], [], [], [], [], []

        df = _create_pivot_custom(conn, value=metric, table=target_table)
        df.set_index('ms_file_label', inplace=True)
        group_field = selected_group if selected_group in df.columns else (
            'sample_type' if 'sample_type' in df.columns else None
        )
        group_label = GROUP_LABELS.get(group_field, 'Group')
        missing_group_label = f"{group_label} (unset)"
        metadata_cols = [col for col in ['ms_type'] + grouping_fields if col in df.columns]
        order_df = conn.execute(
            "SELECT ms_file_label FROM samples ORDER BY ms_file_label"
        ).df()["ms_file_label"].tolist()
        ordered_labels = [lbl for lbl in order_df if lbl in df.index]
        leftover_labels = [lbl for lbl in df.index if lbl not in ordered_labels]
        df = df.loc[ordered_labels + leftover_labels]
        # Sort samples by group, then alphabetically within each group for clustermap display
        if group_field and group_field in df.columns:
            df = df.sort_values(by=[group_field, df.index.name or 'ms_file_label'], 
                                key=lambda x: x.str.lower() if x.dtype == 'object' else x)
            # Re-sort preserving group order but sorting index alphabetically within groups
            df['_sort_group'] = df[group_field].fillna('')
            df['_sort_index'] = df.index.str.lower()
            df = df.sort_values(by=['_sort_group', '_sort_index'])
            df = df.drop(columns=['_sort_group', '_sort_index'])
        group_series = df[group_field] if group_field else pd.Series(df.index, index=df.index, name='group')
        if isinstance(group_series, pd.Series):
            group_series = group_series.replace("", pd.NA)
        group_series.name = group_field or 'group'
        group_series = group_series.fillna(missing_group_label)
        colors_df = conn.execute(
            f"SELECT ms_file_label, color, sample_type, {', '.join(GROUP_COLUMNS)} FROM samples"
        ).df()
        color_map = _build_color_map(colors_df, group_field)
        if not color_map and group_field != 'sample_type':
            color_map = _build_color_map(colors_df, 'sample_type')
        if missing_group_label in group_series.values:
            if color_map is None:
                color_map = {}
            color_map.setdefault(missing_group_label, '#bbbbbb')
        raw_df = df.copy()
        df = df.drop(columns=[c for c in metadata_cols if c in df.columns], axis=1)
        compound_options = sorted(
            [
                {'label': c, 'value': c}
                for c in raw_df.columns
                if c not in metadata_cols
            ],
            key=lambda o: o['label'].lower(),
        )
        # Guard against NaN/inf and empty matrices (numeric only) before downstream plots
        df = _clean_numeric(df)
        raw_numeric_cols = [c for c in raw_df.columns if c not in metadata_cols]
        raw_numeric = _clean_numeric(raw_df[raw_numeric_cols])
        raw_df[raw_numeric_cols] = raw_numeric
        color_labels = group_series.reindex(df.index).fillna(missing_group_label)
        if df.empty or raw_numeric.empty:
            return None, invisible_fig, invisible_fig, [], [], [], [], [], []
        from ..pca import StandardScaler
        scaler = StandardScaler()
        provided_norm = norm_value  # keep the user-provided value (None on first layout pass)
        norm_value = norm_value or TAB_DEFAULT_NORM.get(tab_key, 'zscore')
        if norm_value == 'zscore':
            zdf = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            ndf = zdf
        elif norm_value == 'durbin':
            ndf = rocke_durbin(df, c=10)
            ndf = _clean_numeric(ndf)
            zdf = ndf
        elif norm_value == 'zscore_durbin':
            z_tmp = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
            ndf_tmp = rocke_durbin(z_tmp, c=10)
            ndf = _clean_numeric(ndf_tmp)
            zdf = ndf
        else:
            # raw/none: use df directly
            ndf = df
            zdf = df
        if ndf.empty or zdf.empty:
            raise PreventUpdate
        # Choose matrix for violin based on normalization selection
        if norm_value == 'zscore':
            violin_matrix = zdf
        elif norm_value in ('durbin', 'zscore_durbin'):
            violin_matrix = ndf
        else:
            violin_matrix = df
        if violin_matrix.empty:
            return dash.no_update, invisible_fig, invisible_fig, [], [], [], [], [], []
    if tab_key == 'clustermap':
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.patches as mpatches
        matplotlib.use('Agg')
        # Use base font_scale for other elements, apply specific sizes to tick labels
        sns.set_theme(style='white', font_scale=0.5)
        sample_colors = None
        if color_map:
            sample_colors = [color_map.get(lbl, '#bbbbbb') for lbl in color_labels]
        
        norm_label = next((o['label'] for o in NORM_OPTIONS if o['value'] == norm_value), norm_value)
        
        vmin = -2.00 if norm_value == 'zscore' else None
        vmax = 2.05 if norm_value == 'zscore' else None
        fig = sns.clustermap(
                             zdf.T,
                             method='ward', metric='euclidean', 
                             cmap='vlag', center=0, vmin=vmin, vmax=vmax,
                             standard_scale=None,
                             row_cluster=cluster_rows if cluster_rows is not None else True,
                             col_cluster=cluster_cols if cluster_cols is not None else False, 
                             dendrogram_ratio=0.1,
                             figsize=(8, 8),
                             cbar_kws={"orientation": "horizontal"},
                             cbar_pos=(0.00, 0.95, 0.075, 0.01),
                             col_colors=sample_colors,
                             row_colors=['#ffffff'] * len(zdf.T.index),
                             colors_ratio=(0.0015, 0.015)
                            )
        # Ensure white backgrounds across panels
        fig.fig.patch.set_facecolor('white')
        fig.ax_heatmap.set_facecolor('white')
        fig.ax_col_dendrogram.set_facecolor('white')
        fig.ax_row_dendrogram.set_facecolor('white')
        # Apply custom font sizes to x and y tick labels
        x_fontsize = fontsize_x if fontsize_x else 5
        y_fontsize = fontsize_y if fontsize_y else 5
        fig.ax_heatmap.tick_params(axis='x', labelsize=x_fontsize, length=0, rotation=90)
        fig.ax_heatmap.tick_params(axis='y', labelsize=y_fontsize, length=0)
        
        fig.ax_heatmap.set_xlabel('Samples', fontsize=7, labelpad=8)
        fig.ax_cbar.tick_params(which='both', axis='both', width=0.3, length=2, labelsize=4)
        fig.ax_cbar.set_title(norm_label, fontsize=6, pad=4)
        # Legend for grouping colors (top right)
        if color_map:
            used_types = [lbl for lbl in color_labels if lbl in color_map]
            handles = [
                mpatches.Patch(color=color_map[stype], label=stype)
                for stype in dict.fromkeys(used_types)  # preserve order, unique
                if stype in color_map
            ]
            if handles:
                fig.ax_heatmap.legend(
                    handles=handles,
                    title=group_label,
                    bbox_to_anchor=(-0.15, 1.025),
                    loc='upper right',
                    ncol=1,
                    frameon=False,
                    fontsize=5,
                    title_fontsize=5,
                    labelspacing=0.75,
                    
                )
        from io import BytesIO
        buf = BytesIO()
        # Save a high-resolution copy to disk for durability/exports
        try:
            # Avoid double-saving when the norm dropdown hasn't populated yet (provided_norm is None)
            # Also skip the immediate tab-change trigger; wait for the follow-up with the final norm value.
            if wdir and provided_norm is not None and triggered_prop != 'analysis-sidebar-menu':
                cm_dir = Path(wdir) / "analysis" / "clustermap"
                cm_dir.mkdir(parents=True, exist_ok=True)
                safe_metric = slugify_label(metric)
                file_name = f"{safe_metric}_{norm_value}_clustermap.png"
                save_path = cm_dir / file_name
                should_save = True
                if save_path.exists():
                    last_write = save_path.stat().st_mtime
                    should_save = (time.time() - last_write) > 30
                if should_save:
                    fig.savefig(save_path, format="png", dpi=600)
                    logger.info(f"Saved high-res Clustermap: {save_path}")
        except Exception:
            logger.error("Failed to save Clustermap image.", exc_info=True)
            pass
        fig.savefig(buf, format="png", dpi=300)
        # Avoid accumulating open figures across callbacks
        plt.close(fig.fig)
        # fig.savefig('test.png', format="png")
        import base64
        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'
        return fig_bar_matplotlib, dash.no_update, dash.no_update, dash.no_update, compound_options, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif tab_key == 'pca':
        logger.info(f"Generating PCA ({x_comp} vs {y_comp})...")
        # n_components should be <= min(n_samples, n_features)
        results = run_pca_samples_in_cols(ndf, n_components=min(ndf.shape[0], ndf.shape[1], 5))
        results['scores']['color_group'] = color_labels
        results['scores']['sample_label'] = results['scores'].index
        x_axis = x_comp or 'PC1'
        y_axis = y_comp or 'PC2'
        component_id = x_axis
        loading_bar = None
        loading_category_order = None
        loadings_df = results.get('loadings')
        has_component = isinstance(loadings_df, pd.DataFrame) and component_id in loadings_df.columns
        if has_component:
            ordered = loadings_df[component_id].abs().sort_values(ascending=False).head(10)
            loading_category_order = list(reversed(ordered.index.tolist()))
            loading_bar = go.Bar(
                x=ordered.loc[loading_category_order],
                y=loading_category_order,
                orientation='h',
                name=component_id,
                marker=dict(color='#bbbbbb'),
                showlegend=False,
                text=loading_category_order,
                textposition='outside',
                cliponaxis=False,
                hovertemplate='<b>%{y}</b><br>|Loading|=%{x:.4f}<extra></extra>',
            )
        variance_bar = go.Bar(
            x=results['explained_variance_ratio'].index,
            y=results['explained_variance_ratio'].values,
            width=0.5,
            showlegend=False,
            marker=dict(color='#bbbbbb'),
        )
        variance_line = go.Scatter(
            x=results['cumulative_variance_ratio'].index,
            y=results['cumulative_variance_ratio'].values,
            mode='lines+markers',
            marker=dict(color='#999999'),
            line=dict(color='#999999'),
            showlegend=False,
        )
        base_scatter = px.scatter(
            results['scores'],
            x=x_axis,
            y=y_axis,
            color='color_group',
            symbol='color_group',
            color_discrete_map=color_map if color_map else None,
            hover_data={'sample_label': True},
            title=f'PCA ({x_axis} vs {y_axis})'
        )
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{'rowspan': 2}, {}],
                   [None, {}]],
            column_widths=[0.6, 0.4],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
            subplot_titles=(
                f"PCA ({x_axis} vs {y_axis})",
                "Cummulative Variance",
                f"|Loadings| ({component_id})" if has_component else "Loadings",
            ),
        )
        for t in base_scatter.data:
            fig.add_trace(t, row=1, col=1)
        fig.add_trace(variance_bar, row=1, col=2)
        fig.add_trace(variance_line, row=1, col=2)
        # Label PCA scatter axes
        fig.update_xaxes(title_text=x_axis, row=1, col=1)
        fig.update_yaxes(title_text=y_axis, row=1, col=1)
        if loading_bar:
            fig.add_trace(loading_bar, row=2, col=2)
            fig.update_yaxes(
                categoryorder='array',
                categoryarray=loading_category_order,
                showticklabels=False,
                row=2,
                col=2,
            )
            # Nudge the subplot title away from the bars for a bit more breathing room
            for ann in fig['layout']['annotations']:
                if isinstance(ann.text, str) and ann.text.startswith("Loadings"):
                    ann.y = ann.y + 0.02
        fig.update_layout(
            autosize=True,
            margin=dict(l=140, r=80, t=60, b=50),
            legend_title_text=group_label,
            legend=dict(
                x=-0.06,
                y=1.05,
                xanchor='right',
                # yanchor='top',
                orientation='v',
                title=dict(text=f'{group_label}<br>', font=dict(size=12)),
                font=dict(size=11),
                itemsizing='constant',
                tracegroupgap=7.5,
            ),
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            xaxis_tickfont=dict(size=12),
            yaxis_tickfont=dict(size=12),
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
        return dash.no_update, fig, dash.no_update, dash.no_update, compound_options, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif tab_key == 'tsne':
        logger.info(f"Generating t-SNE...")
        
        # CPU limit logic
        n_jobs = max(1, min((os.cpu_count() or 4) // 2, get_physical_cores()))

        # t-SNE components (usually 2, sometimes 3)
        n_components = 3
        # Use random_state for reproducibility
        perplexity = tsne_perplexity if tsne_perplexity else 30
        
        # Ensure perplexity is valid for sample size
        n_samples = ndf.shape[0]
        if n_samples > 0:
             perplexity = min(perplexity, max(1, n_samples - 1))
             
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_jobs=n_jobs, random_state=42, init='pca')
        
        # Handle sparse/empty
        if ndf.empty or ndf.shape[0] < 2:
            return dash.no_update, dash.no_update, invisible_fig, dash.no_update, compound_options, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Fit t-SNE
        # For t-SNE, use the normalized data
        data_for_tsne = ndf.to_numpy()
        embedded = tsne.fit_transform(data_for_tsne)
        
        # Create results DataFrame
        tsne_cols = [f"t-SNE-{i+1}" for i in range(n_components)]
        scores_df = pd.DataFrame(embedded, index=ndf.index, columns=tsne_cols)
        
        # Add metadata for plotting
        scores_df['color_group'] = color_labels
        scores_df['sample_label'] = scores_df.index
        
        x_axis = tsne_x_comp or 't-SNE-1'
        y_axis = tsne_y_comp or 't-SNE-2'
        
        # Determine actual columns to use (fallback to available if selected not present)
        if x_axis not in scores_df.columns and len(scores_df.columns) > 0:
            x_axis = scores_df.columns[0]
        if y_axis not in scores_df.columns and len(scores_df.columns) > 1:
            y_axis = scores_df.columns[1]
            
        fig = px.scatter(
            scores_df,
            x=x_axis,
            y=y_axis,
            color='color_group',
            symbol='color_group',
            color_discrete_map=color_map if color_map else None,
            hover_data={'sample_label': True},
        )
        
        fig.update_layout(
            autosize=True,
            margin=dict(l=140, r=80, t=60, b=50),
            legend_title_text=group_label,
            legend=dict(
                x=-0.06,
                y=1.05,
                xanchor='right',
                orientation='v',
                title=dict(text=f'{group_label}<br>', font=dict(size=12)),
                font=dict(size=11),
                itemsizing='constant',
                tracegroupgap=7.5,
            ),
            xaxis_title_font=dict(size=16),
            yaxis_title_font=dict(size=16),
            xaxis_tickfont=dict(size=12),
            yaxis_tickfont=dict(size=12),
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
        return dash.no_update, dash.no_update, fig, dash.no_update, compound_options, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif tab_key == 'raincloud':
        logger.info("Generating Violin/Raincloud plots...")
        # Build options list; sort by absolute PC1 loading if available so the most
        # influential metabolites surface first.
        loadings_for_sort = None
        violin_options = compound_options
        # Default selection: highest absolute loading on PC1 (per current metric/norm).
        default_violin = None
        if violin_options:
            try:
                pca_results = run_pca_samples_in_cols(
                    violin_matrix,
                    n_components=min(violin_matrix.shape[0], violin_matrix.shape[1], 5)
                )
                loadings = pca_results.get('loadings')
                if loadings is not None and 'PC1' in loadings.columns:
                    loadings_for_sort = loadings
                    default_violin = loadings['PC1'].abs().idxmax()
            except Exception:
                if violin_options:
                    default_violin = violin_options[0]['value']
        
        if not default_violin and violin_options:
             default_violin = violin_options[0]['value']

        if loadings_for_sort is not None and 'PC1' in loadings_for_sort.columns:
            pc1_sorted = loadings_for_sort['PC1'].abs().sort_values(ascending=False)
            option_map = {opt['value']: opt for opt in compound_options}
            violin_options = [option_map[val] for val in pc1_sorted.index if val in option_map]
        
        from dash import callback_context
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        user_selected = triggered == 'violin-comp-checks'
        
        selected_compound = None
        if user_selected and violin_comp_checks:
             # Handle potential legacy list value or single value
             if isinstance(violin_comp_checks, list):
                  selected_compound = violin_comp_checks[0] if violin_comp_checks else default_violin
             else:
                  selected_compound = violin_comp_checks
        else:
             # Use current selection if valid and not triggering a reset, otherwise default
             if violin_comp_checks and not isinstance(violin_comp_checks, list) and violin_comp_checks in violin_matrix.columns:
                 selected_compound = violin_comp_checks
             else:
                 selected_compound = default_violin
        
        graphs = []
        if selected_compound and selected_compound in violin_matrix.columns:
            selected = selected_compound
            melt_df = violin_matrix[[selected]].join(group_series).reset_index().rename(columns={
                'ms_file_label': 'Sample',
                group_series.name: group_label,
                selected: 'Intensity',
            })
            
            # Use label from METRIC_OPTIONS instead of hardcoded strings
            metric_label = next((opt['label'] for opt in METRIC_OPTIONS if opt['value'] == metric), metric)

            melt_df['PlotValue'] = melt_df['Intensity']
            y_label = metric_label

            fig = px.violin(
                melt_df,
                x=group_label,
                y='PlotValue',
                color=group_label,
                color_discrete_map=color_map if color_map else None,
                box=False,
                points='all',
                hover_data={
                    'Sample': True,
                    group_label: False,  # redundant with x-axis
                    'PlotValue': False,  # redundant with Intensity
                    'Intensity': ':.2e'  # formatted intensity
                },
            )
            # Custom hover template to be very concise and clean
            fig.update_traces(
                hovertemplate="<b>%{customdata[0]}</b><br>Int: %{customdata[1]}<extra></extra>",
                selector=dict(type='violin')
            )
            fig.update_traces(jitter=0.25, meanline_visible=False, pointpos=-0.5, selector=dict(type='violin'))
            # Clamp KDE tails with spanmode='hard', similar to seaborn cut; use 1st-99th percentiles
            low, high = (
                melt_df['PlotValue'].quantile(0.01),
                melt_df['PlotValue'].quantile(0.99),
            )
            fig.update_traces(spanmode='hard', span=[low, high], side='positive', scalemode='width',
                              selector=dict(type='violin'))
            # Simple significance test: t-test for 2 groups, ANOVA for >2
            groups = [
                g['PlotValue'].dropna().to_numpy()
                for _, g in melt_df.groupby(group_label)
            ]
            groups = [g for g in groups if len(g) >= 2]
            method = None
            p_val = None
            if len(groups) == 2:
                method = "t-test"
                _, p_val = ttest_ind(groups[0], groups[1], equal_var=False, nan_policy='omit')
            elif len(groups) > 2:
                method = "ANOVA"
                _, p_val = f_oneway(*groups)
            
            title_text = f"{selected}"
            if method and p_val is not None and np.isfinite(p_val):
                display_p = f"{p_val:.3e}"
                title_text += f" <span style='font-size: 14px; font-weight: normal; color: #555;'>({method}, p={display_p})</span>"

            fig.update_layout(
                title=dict(text=title_text),
                title_font=dict(size=16),
                yaxis_title=y_label,
                xaxis_title=group_label,
                yaxis=dict(rangemode='tozero' if norm_value in ['none', 'durbin'] else 'normal', fixedrange=False),

                margin=dict(l=0, r=10, t=110, b=80),
                height=450,
                legend=dict(
                        title=dict(text=f"{group_label}: ", font=dict(size=13)),
                        font=dict(size=12),
                        orientation='h',
                        yanchor='top',
                        y=-0.3,
                        xanchor='left',
                        x=0,
                    ),
                xaxis_title_font=dict(size=16),
                yaxis_title_font=dict(size=16),
                xaxis_tickfont=dict(size=12),
                yaxis_tickfont=dict(size=12),
                template='plotly_white',
                paper_bgcolor='white',
                plot_bgcolor='white',
                clickmode='event'
            )
            graphs.append(dcc.Graph(
                id={'type': 'violin-plot', 'index': 'main'},
                figure=fig, 
                style={'height': '450px', 'width': '100%'},
                config=PLOTLY_HIGH_RES_CONFIG
            ))
        return dash.no_update, dash.no_update, dash.no_update, graphs, violin_options, selected_compound, dash.no_update, dash.no_update, dash.no_update
    elif tab_key == 'bar':
        default_bar = None
        bar_options = compound_options
        
        # Similar Logic for default selection as Violin
        if bar_options:
            try:
                pca_results = run_pca_samples_in_cols(
                    violin_matrix,
                    n_components=min(violin_matrix.shape[0], violin_matrix.shape[1], 5)
                )
                loadings = pca_results.get('loadings')
                if loadings is not None and 'PC1' in loadings.columns:
                    loadings_for_sort = loadings
                    default_bar = loadings['PC1'].abs().idxmax()
            except Exception:
                 if bar_options:
                    default_bar = bar_options[0]['value']
        
        if not default_bar and bar_options:
             default_bar = bar_options[0]['value']

        if loadings_for_sort is not None and 'PC1' in loadings_for_sort.columns:
            pc1_sorted = loadings_for_sort['PC1'].abs().sort_values(ascending=False)
            option_map = {opt['value']: opt for opt in compound_options}
            bar_options = [option_map[val] for val in pc1_sorted.index if val in option_map]
        
        from dash import callback_context
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        user_selected = triggered == 'bar-comp-checks'
        
        selected_compound = None
        if user_selected and bar_comp_checks:
             selected_compound = bar_comp_checks
        else:
             if bar_comp_checks and bar_comp_checks in violin_matrix.columns:
                 selected_compound = bar_comp_checks
             else:
                 selected_compound = default_bar
        
        graphs = []
        if selected_compound and selected_compound in violin_matrix.columns:
            selected = selected_compound
            melt_df = violin_matrix[[selected]].join(group_series).reset_index().rename(columns={
                'ms_file_label': 'Sample',
                group_series.name: group_label,
                selected: 'Intensity',
            })
            
            metric_label = next((opt['label'] for opt in METRIC_OPTIONS if opt['value'] == metric), metric)
            melt_df['PlotValue'] = melt_df['Intensity']
            y_label = metric_label

            # Calculate stats for Bar plot
            stats_df = melt_df.groupby(group_label)['PlotValue'].agg(['mean', 'sem', 'count']).reset_index()
            
            # Use numeric indexing for X-axis to allow custom jitter and width control
            unique_groups = stats_df[group_label].tolist()
            group_map = {g: i for i, g in enumerate(unique_groups)}
            
            # Dynamic bar width: narrower if few groups, standard otherwise
            n_groups = len(unique_groups)
            bar_width = 0.5 if n_groups <= 3 else None
            
            # Calculate jittered X for scatter
            x_jittered = []
            jitter_amount = 0.15 # Max offset from center
            for g in melt_df[group_label]:
                idx = group_map.get(g)
                if idx is not None:
                     # Deterministic pseudo-random jitter based on value for stability or just random?
                     # Random is standard for jitter
                     noise = np.random.uniform(-jitter_amount, jitter_amount)
                     x_jittered.append(idx + noise)
                else:
                     x_jittered.append(None)

            fig = go.Figure()

            # Add Bars with Error Bars
            fig.add_trace(go.Bar(
                x=[group_map[g] for g in stats_df[group_label]],
                y=stats_df['mean'],
                error_y=dict(type='data', array=stats_df['sem'], visible=True),
                marker_color=[color_map.get(g, '#333') for g in stats_df[group_label]] if color_map else None,
                name='Mean ± SE',
                opacity=0.7,
                showlegend=False,
                width=bar_width
            ))

            # Add individual points on top with jitter
            fig.add_trace(go.Scatter(
                x=x_jittered,
                y=melt_df['PlotValue'],
                mode='markers',
                marker=dict(
                    color='rgba(50, 50, 50, 0.7)',
                    line=dict(width=1, color='white'),
                    size=8
                ),
                customdata=melt_df[['Sample']].values,
                hovertemplate="<b>%{customdata[0]}</b><br>Val: %{y:.2e}<extra></extra>",
                name='Samples',
                showlegend=False
            ))

            # Significance tests (reuse logic)
            groups = [
                g['PlotValue'].dropna().to_numpy()
                for _, g in melt_df.groupby(group_label)
            ]
            groups = [g for g in groups if len(g) >= 2]
            method = None
            p_val = None
            if len(groups) == 2:
                method = "t-test"
                _, p_val = ttest_ind(groups[0], groups[1], equal_var=False, nan_policy='omit')
            elif len(groups) > 2:
                method = "ANOVA"
                _, p_val = f_oneway(*groups)
            
            title_text = f"{selected}"
            if method and p_val is not None and np.isfinite(p_val):
                display_p = f"{p_val:.3e}"
                title_text += f" <span style='font-size: 14px; font-weight: normal; color: #555;'>({method}, p={display_p})</span>"

            fig.update_layout(
                title=dict(text=title_text),
                title_font=dict(size=16),
                yaxis_title=y_label,
                xaxis_title=group_label,
                yaxis=dict(rangemode='tozero' if norm_value in ['none', 'durbin'] else 'normal', fixedrange=False),
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(unique_groups))),
                    ticktext=unique_groups,
                    range=[-0.5, len(unique_groups) - 0.5] # Ensure nicely centered view
                ),
                margin=dict(l=60, r=20, t=80, b=80),
                height=450,
                template='plotly_white',
                paper_bgcolor='white',
                plot_bgcolor='white',
                clickmode='event'
            )
            graphs.append(dcc.Graph(
                id={'type': 'bar-plot', 'index': 'main'},
                figure=fig, 
                style={'height': '450px', 'width': '100%'},
                config=PLOTLY_HIGH_RES_CONFIG
            ))
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, graphs, bar_options, selected_compound
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

def callbacks(app, fsc, cache):
    @app.callback(
        Output("analysis-notifications-container", "children"),
        Input('section-context', 'data'),
        Input("wdir", "data"),
        prevent_initial_call=True,
    )
    def warn_missing_workspace(section_context, wdir):
        if not section_context or section_context.get('page') != 'Analysis':
            return dash.no_update
        if not wdir:
            return fac.AntdNotification(
                message="Activate a workspace",
                description="Please select or create a workspace first.",
                type="warning",
                duration=4,
                placement='bottom',
                showProgress=True,
                stack=True,
            )
        
        # Check if results table is empty
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return []
            results_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            if results_count == 0:
                return fac.AntdNotification(
                    message="No data available",
                    description="Please run MINT processing first by navigating to the Processing tab and clicking Run MINT.",
                    type="info",
                    duration=6,
                    placement='bottom',
                    showProgress=True,
                    stack=True,
                )
        
        return []


    # Sidebar collapse toggle callback
    @app.callback(
        Output('analysis-sidebar', 'collapsed'),
        Output('analysis-sidebar-collapse-icon', 'icon'),
        Input('analysis-sidebar-collapse', 'nClicks'),
        State('analysis-sidebar', 'collapsed'),
        prevent_initial_call=True,
    )
    def toggle_analysis_sidebar(n_clicks, is_collapsed):
        if n_clicks is None:
            raise PreventUpdate
        new_collapsed = not is_collapsed
        icon = 'antd-right' if new_collapsed else 'antd-left'
        return new_collapsed, icon

    # Menu selection -> content visibility + sync to analysis-tabs store
    @app.callback(
        Output('analysis-qc-container', 'style'),
        Output('analysis-pca-container', 'style'),
        Output('analysis-tsne-container', 'style'),
        Output('analysis-violin-container', 'style'),
        Output('analysis-bar-container', 'style'),
        Output('analysis-clustermap-container', 'style'),
        Output('analysis-tabs', 'data'),
        Input('analysis-sidebar-menu', 'currentKey'),
        prevent_initial_call=False,
    )
    def update_analysis_content_visibility(current_key):
        # Default to 'qc' if no key selected (QC is now the first tab)
        active_key = current_key or 'qc'
        
        qc_style = {'display': 'block', 'padding': '16px'} if active_key == 'qc' else {'display': 'none', 'padding': '16px'}
        pca_style = {'display': 'block', 'padding': '16px'} if active_key == 'pca' else {'display': 'none', 'padding': '16px'}
        tsne_style = {'display': 'block', 'padding': '16px'} if active_key == 'tsne' else {'display': 'none', 'padding': '16px'}
        violin_style = {'display': 'block', 'padding': '16px'} if active_key == 'raincloud' else {'display': 'none', 'padding': '16px'}
        bar_style = {'display': 'block', 'padding': '16px'} if active_key == 'bar' else {'display': 'none', 'padding': '16px'}
        # Clustermap needs explicit height for spinner centering to work on first access
        clustermap_style = {
            'display': 'block' if active_key == 'clustermap' else 'none',
            'padding': '16px',
            'height': 'calc(100vh - 150px)',  # Explicit height for flexbox centering
        }
        
        # Sync to legacy analysis-tabs store for backward compatibility with other callbacks
        tabs_data = {'activeKey': active_key}
        
        return qc_style, pca_style, tsne_style, violin_style, bar_style, clustermap_style, tabs_data


    @app.callback(
        Output('analysis-metric-wrapper', 'style'),
        Output('analysis-normalization-wrapper', 'style'),
        Input('analysis-sidebar-menu', 'currentKey'),
        prevent_initial_call=False,
    )
    def toggle_metric_visibility(active_tab):
        # For QC tab, hide Metric and Transformations (show only Group by)
        if active_tab == 'qc':
            hidden_style = {'display': 'none'}
            return hidden_style, hidden_style
        # For other tabs, show all controls
        visible_style = {'display': 'flex'}
        return visible_style, visible_style


    @app.callback(
        Output('analysis-normalization-select', 'value', allow_duplicate=True),
        Input('analysis-sidebar-menu', 'currentKey'),
        prevent_initial_call=True,
    )
    def set_norm_default_for_tab(active_tab):
        return TAB_DEFAULT_NORM.get(active_tab, 'zscore')


    @app.callback(
        Output('analysis-tour', 'current'),
        Output('analysis-tour', 'open'),
        Output('analysis-tour', 'steps'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('analysis-tour-icon', 'nClicks'),
        State('analysis-tour', 'open'),
        prevent_initial_call=False,
    )
    def analysis_tour_open(active_tab, n_clicks, is_open):
        ctx = dash.callback_context
        steps = _analysis_tour_steps(active_tab)
        if not ctx.triggered:
            return 0, False, steps
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'analysis-tour-icon':
            return 0, True, steps
        if trigger == 'analysis-sidebar-menu' and is_open:
            return 0, True, steps
        return dash.no_update, dash.no_update, steps

    app.callback(
        Output('bar-graph-matplotlib', 'src'),
        Output('pca-graph', 'figure'),
        Output('tsne-graph', 'figure'),
        Output('violin-graphs', 'children'),
        Output('violin-comp-checks', 'options'),
        Output('violin-comp-checks', 'value'),
        Output('bar-graphs', 'children'),
        Output('bar-comp-checks', 'options'),
        Output('bar-comp-checks', 'value'),

        Input('section-context', 'data'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('pca-x-comp', 'value'),
        Input('pca-y-comp', 'value'),
        Input('violin-comp-checks', 'value'),
        Input('bar-comp-checks', 'value'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('clustermap-regenerate-btn', 'nClicks'),
        Input('tsne-regenerate-btn', 'nClicks'),
        Input('clustermap-cluster-rows', 'checked'),
        Input('clustermap-cluster-cols', 'checked'),
        State('clustermap-fontsize-x-slider', 'value'),
        State('clustermap-fontsize-y-slider', 'value'),
        State('wdir', 'data'),
        State('tsne-x-comp', 'value'),
        State('tsne-y-comp', 'value'),
        State('tsne-perplexity-slider', 'value'),
        prevent_initial_call=True,

    )(show_tab_content)


    @app.callback(
        Output('clustermap-spinner', 'spinning'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('bar-graph-matplotlib', 'src'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        prevent_initial_call=False,
    )
    def toggle_clustermap_spinner(active_tab, bar_src, metric_value, norm_value):
        from dash import callback_context

        if active_tab != 'clustermap':
            return False

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        # When user switches to clustermap or changes metric/normalization, force spinner on even if previous image exists
        if trigger in ('analysis-sidebar-menu', 'analysis-metric-select', 'analysis-normalization-select'):
            return True

        # Otherwise, keep spinning until image src is set
        return bar_src is None

    @app.callback(
        Output('clustermap-download', 'data'),
        Input('clustermap-save-png-btn', 'nClicks'),
        State('bar-graph-matplotlib', 'src'),
        prevent_initial_call=True,
    )
    def save_clustermap_png(n_clicks, img_src):
        if not n_clicks or not img_src:
            raise PreventUpdate
        
        # Extract base64 data from src
        if ',' in img_src:
            img_data = img_src.split(',')[1]
        else:
            img_data = img_src
            
        return dict(
            content=img_data,
            filename='clustermap.png',
            type='image/png',
            base64=True,
        )

    @app.callback(
        Output('violin-spinner', 'spinning'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('violin-graphs', 'children'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        prevent_initial_call=True,
    )
    def toggle_violin_spinner(active_tab, violin_children, metric_value, norm_value, group_value):
        from dash import callback_context

        if active_tab != 'raincloud':
            return False

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        # When user switches to raincloud or changes parameters, force spinner on
        if trigger in ('analysis-sidebar-menu', 'analysis-metric-select', 'analysis-normalization-select', 'analysis-grouping-select'):
            return True

        # Otherwise, stop spinning once content is loaded
        return not violin_children



    @app.callback(
        Output('violin-chromatogram', 'figure'),
        Output('violin-chromatogram-container', 'style'),
        Output('violin-selected-sample', 'data'),
        Input({'type': 'violin-plot', 'index': ALL}, 'clickData'),
        Input('violin-comp-checks', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('violin-log-scale-switch', 'checked'),
        State('violin-selected-sample', 'data'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def update_chromatogram_on_click(clickData_list, peak_label, group_by_col, metric, normalization, log_scale, current_selection, wdir):
        import random
        from dash import ALL
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update

        # Extract clickData safely
        clickData = clickData_list[0] if clickData_list else None

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Reset triggers - when these change, auto-select a random sample
        reset_triggers = [
            'violin-comp-checks', 
            'analysis-grouping-select', 
            'analysis-metric-select', 
            'analysis-normalization-select'
        ]

        ms_file_label = None
        
        # If triggered by a click, extract the sample from clickData
        if 'violin-plot' in ctx.triggered[0]['prop_id']:
            try:
                ms_file_label = clickData['points'][0]['customdata'][0]
            except (KeyError, IndexError, TypeError):
                pass
        
        # If triggered by log scale toggle, keep current selection
        elif trigger_id == 'violin-log-scale-switch' and current_selection:
            ms_file_label = current_selection

        # If no sample (or reset triggered), auto-select a random sample with valid signal
        if not ms_file_label and wdir and peak_label:
            with duckdb_connection(wdir) as conn:
                if conn:
                    # Get top 5 samples with highest intensity contrast (max - min)
                    # This ensures we pick a sample with actual peaks, not flat lines
                    top_samples = conn.execute("""
                        SELECT ms_file_label 
                        FROM chromatograms
                        WHERE peak_label = ?
                        ORDER BY (list_max(intensity) - list_min(intensity)) DESC
                        LIMIT 5
                    """, [peak_label]).fetchall()
                    
                    if top_samples:
                        import random
                        ms_file_label = random.choice(top_samples)[0]
                    else:
                        # Fallback: if all samples are flat, just pick any
                        random_sample = conn.execute("""
                            SELECT DISTINCT c.ms_file_label 
                            FROM chromatograms c
                            WHERE c.peak_label = ?
                            ORDER BY random()
                            LIMIT 1
                        """, [peak_label]).fetchone()
                        if random_sample:
                            ms_file_label = random_sample[0]

        if not ms_file_label or not wdir or not peak_label:
            # Return empty placeholder figure
            fig = go.Figure()
            # fig.add_annotation(
            #     text="Select a sample from the violin plot",
            #     xref="paper", yref="paper",
            #     x=0.5, y=0.5,
            #     showarrow=False,
            #     font=dict(size=14, color="gray")
            # )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                template="plotly_white",
                margin=dict(l=0, r=0, t=0, b=0),
            )
            return fig, {'display': 'block', 'width': 'calc(43% - 6px)', 'height': '450px'}, None

        # Fetch data
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update, dash.no_update, dash.no_update
            
            # 1. Get RT span info for the target
            rt_info = conn.execute("SELECT rt_min, rt_max FROM targets WHERE peak_label = ?", [peak_label]).fetchone()
            rt_min, rt_max = rt_info if rt_info else (None, None)

            # 2. Identify neighbors if a grouping column is selected
            neighbor_files = []
            group_val = None
            
            # Determine group label
            group_label = GROUP_LABELS.get(group_by_col, group_by_col or 'Sample')

            if group_by_col:
                # Get the group value for the clicked sample
                try:
                    group_val_query = f'SELECT "{group_by_col}" FROM samples WHERE ms_file_label = ?'
                    row = conn.execute(group_val_query, [ms_file_label]).fetchone()
                    
                    if row:
                        group_val = row[0]
                        # Fetch up to 10 random other samples from the same group
                        if group_val is None:
                             neighbors_query = f"""
                                SELECT ms_file_label, color 
                                FROM samples 
                                WHERE "{group_by_col}" IS NULL AND ms_file_label != ? 
                                ORDER BY random() 
                                LIMIT 10
                            """
                             neighbor_files = conn.execute(neighbors_query, [ms_file_label]).fetchall()
                        else:
                            neighbors_query = f"""
                                SELECT ms_file_label, color 
                                FROM samples 
                                WHERE "{group_by_col}" = ? AND ms_file_label != ? 
                                ORDER BY random() 
                                LIMIT 10
                            """
                            neighbor_files = conn.execute(neighbors_query, [group_val, ms_file_label]).fetchall()
                except Exception as e:
                    logger.warning(f"Failed to fetch neighbors: {e}")
                    pass

            # Determine display value for legend
            display_val = group_val
            if group_by_col and not group_val:
                 display_val = f"{group_label} (unset)"

            # 3. Fetch chromatograms
            # We need the clicked sample + neighbors
            files_to_fetch = [ms_file_label] + [n[0] for n in neighbor_files]
            placeholders = ','.join(['?'] * len(files_to_fetch))
            
            chrom_query = f"""
                SELECT c.ms_file_label, c.scan_time, c.intensity, s.color
                FROM chromatograms c
                JOIN samples s ON c.ms_file_label = s.ms_file_label
                WHERE c.peak_label = ? AND c.ms_file_label IN ({placeholders})
            """
            
            chrom_data = conn.execute(chrom_query, [peak_label] + files_to_fetch).fetchall()
            
            if not chrom_data:
                fig = go.Figure()
                fig.add_annotation(text="No chromatogram data found", showarrow=False)
                return fig, {'display': 'block', 'width': 'calc(43% - 6px)', 'height': '450px'}, ms_file_label

            # Organize data
            # chrom_data: [(ms_file_label, scan_time, intensity, color), ...]
            data_map = {row[0]: row for row in chrom_data}
            
            fig = go.Figure()

            # Plot neighbors first (background)
            # Plot neighbors first (background)
            for n_label, n_color in neighbor_files:
                if n_label in data_map:
                    _, scan_times, intensities, _ = data_map[n_label]
                    
                    # Check for valid signal (skip flat lines)
                    if len(intensities) > 0 and min(intensities) == max(intensities):
                        continue

                    # If grouping is active but value is missing, use the "unset" color (gray)
                    if group_by_col and group_val is None:
                        n_color = '#bbbbbb'
                    
                    if log_scale:
                         intensities = np.log2(np.array(intensities) + 1)

                    fig.add_trace(go.Scatter(
                        x=scan_times,
                        y=intensities,
                        mode='lines',
                        name=str(display_val),
                        legendgroup=str(display_val),
                        showlegend=False,
                        line=dict(width=1, color=n_color),
                        opacity=0.4,
                        hovertemplate=f"<b>{n_label}</b><br>Scan Time: %{{x:.2f}}<br>Intensity: %{{y:.2e}}<extra>{display_val}</extra>"
                    ))

            # Plot clicked sample (foreground)
            if ms_file_label in data_map:
                _, scan_times, intensities, main_color = data_map[ms_file_label]
                
                # Check for valid signal
                is_flat = len(intensities) > 0 and min(intensities) == max(intensities)
                
                if is_flat:
                     fig.add_annotation(
                        text="Selected sample has no valid signal (flat line)",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="gray")
                    )
                else:
                    # If grouping is active but value is missing, use the "unset" color (gray)
                    if group_by_col and not group_val:
                        main_color = '#bbbbbb'

                    legend_name = str(display_val) if group_by_col else ms_file_label
                    
                    if log_scale:
                         intensities = np.log2(np.array(intensities) + 1)

                    fig.add_trace(go.Scatter(
                        x=scan_times,
                        y=intensities,
                        mode='lines',
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=True,
                        hovertemplate=f"<b>{ms_file_label}</b><br>Scan Time: %{{x:.2f}}<br>Intensity: %{{y:.2e}}<extra>{legend_name}</extra>",
                        line=dict(width=2, color=main_color),
                        fill='tozeroy',
                        opacity=1.0
                    ))
            
            # Add RT span
            if rt_min is not None and rt_max is not None:
                fig.add_vrect(
                    x0=rt_min, x1=rt_max,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                )

            # Set initial X-axis range to show RT span with padding (like optimization modal)
            # Show one span width on each side of the RT span
            x_range_min = None
            x_range_max = None
            if rt_min is not None and rt_max is not None:
                span_width = rt_max - rt_min
                # Use 5 seconds as padding since at this point this is optimized
                padding = 5
                x_range_min = rt_min - padding
                x_range_max = rt_max + padding
                
                # Clamp to actual data range
                all_x = []
                for trace in fig.data:
                    if hasattr(trace, 'x') and trace.x is not None:
                        all_x.extend(trace.x)
                if all_x:
                    data_x_min = min(all_x)
                    data_x_max = max(all_x)
                    x_range_min = max(x_range_min, data_x_min)
                    x_range_max = min(x_range_max, data_x_max)

            # Calculate Y-axis range for the visible X range
            y_range = None
            if x_range_min is not None and x_range_max is not None:
                traces_data = [{'x': list(t.x), 'y': list(t.y)} for t in fig.data if hasattr(t, 'x') and hasattr(t, 'y')]
                y_range = _calc_y_range_numpy(traces_data, x_range_min, x_range_max, is_log=False)

            # Truncate title if too long
            title_label = ms_file_label
            if len(title_label) > 50:
                 title_label = title_label[:20] + "..." + title_label[-20:]

            # Update layout
            y_title = "Intensity (Log2)" if log_scale else "Intensity"
            fig.update_layout(
                title=dict(text=f"{peak_label} | {title_label}", font=dict(size=14)),
                xaxis_title="Scan Time (s)",
                yaxis_title=y_title,
                xaxis_title_font=dict(size=16),
                yaxis_title_font=dict(size=16),
                xaxis_tickfont=dict(size=12),
                yaxis_tickfont=dict(size=12),
                template="plotly_white",
                margin=dict(l=50, r=20, t=110, b=80),
                height=450,
                showlegend=True,
                legend=dict(
                    title=dict(text=f"{group_label}: ", font=dict(size=13)),
                    font=dict(size=12),
                    orientation='h',
                    yanchor='top',
                    y=-0.3,
                    xanchor='left',
                    x=0,
                ),
                xaxis=dict(
                    range=[x_range_min, x_range_max] if x_range_min is not None else None,
                    autorange=x_range_min is None,
                ),
                yaxis=dict(
                    range=y_range if y_range else None,
                    autorange=y_range is None,
                ),
            )
            return fig, {'display': 'block', 'width': 'calc(43% - 6px)', 'height': '450px'}, ms_file_label

    @app.callback(
        Output('bar-chromatogram', 'figure'),
        Output('bar-chromatogram-container', 'style'),
        Output('bar-selected-sample', 'data'),
        Input({'type': 'bar-plot', 'index': ALL}, 'clickData'),
        Input('bar-comp-checks', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('analysis-metric-select', 'value'),
        Input('analysis-normalization-select', 'value'),
        Input('bar-log-scale-switch', 'checked'),
        State('bar-selected-sample', 'data'),
        State("wdir", "data"),
        prevent_initial_call=True,
    )
    def update_bar_chromatogram_on_click(clickData_list, peak_label, group_by_col, metric, normalization, log_scale, current_selection, wdir):
        import random
        from dash import ALL
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update

        # Extract clickData safely
        clickData = clickData_list[0] if clickData_list else None

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Reset triggers - when these change, auto-select a random sample
        reset_triggers = [
            'bar-comp-checks', 
            'analysis-grouping-select', 
            'analysis-metric-select', 
            'analysis-normalization-select'
        ]

        ms_file_label = None
        
        # If triggered by a click, extract the sample from clickData
        if 'bar-plot' in ctx.triggered[0]['prop_id']:
            try:
                ms_file_label = clickData['points'][0]['customdata'][0]
            except (KeyError, IndexError, TypeError):
                pass
        
        # If triggered by log scale toggle, keep current selection
        elif trigger_id == 'bar-log-scale-switch' and current_selection:
            ms_file_label = current_selection

        # If no sample (or reset triggered), auto-select a random sample with valid signal
        if not ms_file_label and wdir and peak_label:
            with duckdb_connection(wdir) as conn:
                if conn:
                    # Get top 5 samples with highest intensity contrast (max - min)
                    # This ensures we pick a sample with actual peaks, not flat lines
                    top_samples = conn.execute("""
                        SELECT ms_file_label 
                        FROM chromatograms
                        WHERE peak_label = ?
                        ORDER BY (list_max(intensity) - list_min(intensity)) DESC
                        LIMIT 5
                    """, [peak_label]).fetchall()
                    
                    if top_samples:
                        import random
                        ms_file_label = random.choice(top_samples)[0]
                    else:
                        # Fallback: if all samples are flat, just pick any
                        random_sample = conn.execute("""
                            SELECT DISTINCT c.ms_file_label 
                            FROM chromatograms c
                            WHERE c.peak_label = ?
                            ORDER BY random()
                            LIMIT 1
                        """, [peak_label]).fetchone()
                        if random_sample:
                            ms_file_label = random_sample[0]

        if not ms_file_label or not wdir or not peak_label:
            # Return empty placeholder figure
            fig = go.Figure()
            # fig.add_annotation(
            #     text="Select a sample from the bar plot",
            #     xref="paper", yref="paper",
            #     x=0.5, y=0.5,
            #     showarrow=False,
            #     font=dict(size=14, color="gray")
            # )
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                template="plotly_white",
                margin=dict(l=0, r=0, t=0, b=0),
            )
            return fig, {'display': 'block', 'width': 'calc(43% - 6px)', 'height': '450px'}, None

        # Fetch data
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return dash.no_update, dash.no_update, dash.no_update
            
            # 1. Get RT span info for the target
            rt_info = conn.execute("SELECT rt_min, rt_max FROM targets WHERE peak_label = ?", [peak_label]).fetchone()
            rt_min, rt_max = rt_info if rt_info else (None, None)

            # 2. Identify neighbors if a grouping column is selected
            neighbor_files = []
            group_val = None
            
            # Determine group label
            group_label = GROUP_LABELS.get(group_by_col, group_by_col or 'Sample')

            if group_by_col:
                # Get the group value for the clicked sample
                try:
                    group_val_query = f'SELECT "{group_by_col}" FROM samples WHERE ms_file_label = ?'
                    row = conn.execute(group_val_query, [ms_file_label]).fetchone()
                    
                    if row:
                        group_val = row[0]
                        # Fetch up to 10 random other samples from the same group
                        if group_val is None:
                             neighbors_query = f"""
                                SELECT ms_file_label, color 
                                FROM samples 
                                WHERE "{group_by_col}" IS NULL AND ms_file_label != ? 
                                ORDER BY random() 
                                LIMIT 10
                            """
                             neighbor_files = conn.execute(neighbors_query, [ms_file_label]).fetchall()
                        else:
                            neighbors_query = f"""
                                SELECT ms_file_label, color 
                                FROM samples 
                                WHERE "{group_by_col}" = ? AND ms_file_label != ? 
                                ORDER BY random() 
                                LIMIT 10
                            """
                            neighbor_files = conn.execute(neighbors_query, [group_val, ms_file_label]).fetchall()
                except Exception as e:
                    logger.warning(f"Failed to fetch neighbors: {e}")
                    pass

            # Determine display value for legend
            display_val = group_val
            if group_by_col and not group_val:
                 display_val = f"{group_label} (unset)"

            # 3. Fetch chromatograms
            # We need the clicked sample + neighbors
            files_to_fetch = [ms_file_label] + [n[0] for n in neighbor_files]
            placeholders = ','.join(['?'] * len(files_to_fetch))
            
            chrom_query = f"""
                SELECT c.ms_file_label, c.scan_time, c.intensity, s.color
                FROM chromatograms c
                JOIN samples s ON c.ms_file_label = s.ms_file_label
                WHERE c.peak_label = ? AND c.ms_file_label IN ({placeholders})
            """
            
            chrom_data = conn.execute(chrom_query, [peak_label] + files_to_fetch).fetchall()
            
            if not chrom_data:
                fig = go.Figure()
                fig.add_annotation(text="No chromatogram data found", showarrow=False)
                return fig, {'display': 'block', 'width': 'calc(43% - 6px)', 'height': '450px'}, ms_file_label

            # Organize data
            # chrom_data: [(ms_file_label, scan_time, intensity, color), ...]
            data_map = {row[0]: row for row in chrom_data}
            
            fig = go.Figure()

            # Plot neighbors first (background)
            # Plot neighbors first (background)
            for n_label, n_color in neighbor_files:
                if n_label in data_map:
                    _, scan_times, intensities, _ = data_map[n_label]
                    
                    # Check for valid signal (skip flat lines)
                    if len(intensities) > 0 and min(intensities) == max(intensities):
                        continue

                    # If grouping is active but value is missing, use the "unset" color (gray)
                    if group_by_col and group_val is None:
                        n_color = '#bbbbbb'
                    
                    if log_scale:
                         intensities = np.log2(np.array(intensities) + 1)

                    fig.add_trace(go.Scatter(
                        x=scan_times,
                        y=intensities,
                        mode='lines',
                        name=str(display_val),
                        legendgroup=str(display_val),
                        showlegend=False,
                        line=dict(width=1, color=n_color),
                        opacity=0.4,
                        hovertemplate=f"<b>{n_label}</b><br>Scan Time: %{{x:.2f}}<br>Intensity: %{{y:.2e}}<extra>{display_val}</extra>"
                    ))

            # Plot clicked sample (foreground)
            if ms_file_label in data_map:
                _, scan_times, intensities, main_color = data_map[ms_file_label]
                
                # Check for valid signal
                is_flat = len(intensities) > 0 and min(intensities) == max(intensities)
                
                if is_flat:
                     fig.add_annotation(
                        text="Selected sample has no valid signal (flat line)",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="gray")
                    )
                else:
                    # If grouping is active but value is missing, use the "unset" color (gray)
                    if group_by_col and not group_val:
                        main_color = '#bbbbbb'

                    legend_name = str(display_val) if group_by_col else ms_file_label
                    
                    if log_scale:
                         intensities = np.log2(np.array(intensities) + 1)

                    fig.add_trace(go.Scatter(
                        x=scan_times,
                        y=intensities,
                        mode='lines',
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=True,
                        hovertemplate=f"<b>{ms_file_label}</b><br>Scan Time: %{{x:.2f}}<br>Intensity: %{{y:.2e}}<extra>{legend_name}</extra>",
                        line=dict(width=2, color=main_color),
                        fill='tozeroy',
                        opacity=1.0
                    ))
            
            # Add RT span
            if rt_min is not None and rt_max is not None:
                fig.add_vrect(
                    x0=rt_min, x1=rt_max,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                )

            # Set initial X-axis range to show RT span with padding (like optimization modal)
            # Show one span width on each side of the RT span
            x_range_min = None
            x_range_max = None
            if rt_min is not None and rt_max is not None:
                span_width = rt_max - rt_min
                # Use 5 seconds as padding since at this point this is optimized
                padding = 5
                x_range_min = rt_min - padding
                x_range_max = rt_max + padding
                
                # Clamp to actual data range
                all_x = []
                for trace in fig.data:
                    if hasattr(trace, 'x') and trace.x is not None:
                        all_x.extend(trace.x)
                if all_x:
                    data_x_min = min(all_x)
                    data_x_max = max(all_x)
                    x_range_min = max(x_range_min, data_x_min)
                    x_range_max = min(x_range_max, data_x_max)

            # Calculate Y-axis range for the visible X range
            y_range = None
            if x_range_min is not None and x_range_max is not None:
                traces_data = [{'x': list(t.x), 'y': list(t.y)} for t in fig.data if hasattr(t, 'x') and hasattr(t, 'y')]
                y_range = _calc_y_range_numpy(traces_data, x_range_min, x_range_max, is_log=False)

            # Truncate title if too long
            title_label = ms_file_label
            if len(title_label) > 50:
                 title_label = title_label[:20] + "..." + title_label[-20:]

            # Update layout
            y_title = "Intensity (Log2)" if log_scale else "Intensity"
            fig.update_layout(
                title=dict(text=f"{peak_label} | {title_label}", font=dict(size=14)),
                xaxis_title="Scan Time (s)",
                yaxis_title=y_title,
                xaxis_title_font=dict(size=16),
                yaxis_title_font=dict(size=16),
                xaxis_tickfont=dict(size=12),
                yaxis_tickfont=dict(size=12),
                template="plotly_white",
                margin=dict(l=50, r=20, t=110, b=80),
                height=450,
                showlegend=True,
                legend=dict(
                    title=dict(text=f"{group_label}: ", font=dict(size=13)),
                    font=dict(size=12),
                    orientation='h',
                    yanchor='top',
                    y=-0.3,
                    xanchor='left',
                    x=0,
                ),
                xaxis=dict(
                    range=[x_range_min, x_range_max] if x_range_min is not None else None,
                    autorange=x_range_min is None,
                ),
                yaxis=dict(
                    range=y_range if y_range else None,
                    autorange=y_range is None,
                ),
            )
            return fig, {'display': 'block', 'width': 'calc(43% - 6px)', 'height': '450px'}, ms_file_label


    @app.callback(
        Output('violin-chromatogram', 'figure', allow_duplicate=True),
        Input('violin-chromatogram', 'relayoutData'),
        State('violin-chromatogram', 'figure'),
        prevent_initial_call=True
    )
    def update_violin_chromatogram_zoom(relayout, figure_state):
        """
        When user zooms on X-axis, auto-fit Y-axis to visible data range.
        This matches the behavior in the optimization modal chromatogram.
        """
        if not relayout or not figure_state:
            raise PreventUpdate

        # Check for X-axis zoom event
        x_range = (relayout.get('xaxis.range[0]'), relayout.get('xaxis.range[1]'))
        y_range = (relayout.get('yaxis.range[0]'), relayout.get('yaxis.range[1]'))
        
        # If user explicitly set Y range, respect it
        if y_range[0] is not None and y_range[1] is not None:
            raise PreventUpdate
        
        # Only act when X-axis zoom happens
        if x_range[0] is None or x_range[1] is None:
            raise PreventUpdate
        
        # Auto-fit Y-axis to visible data in the X range
        from dash import Patch
        fig_patch = Patch()
        
        traces = figure_state.get('data', [])
        y_calc = _calc_y_range_numpy(traces, x_range[0], x_range[1], is_log=False)
        
        if y_calc:
            fig_patch['layout']['xaxis']['range'] = [x_range[0], x_range[1]]
            fig_patch['layout']['xaxis']['autorange'] = False
            fig_patch['layout']['yaxis']['range'] = y_calc
            fig_patch['layout']['yaxis']['autorange'] = False
            return fig_patch
        
        raise PreventUpdate

    @app.callback(
        Output({'type': 'violin-plot', 'index': MATCH}, 'figure'),
        Input({'type': 'violin-plot', 'index': MATCH}, 'clickData'),
        State({'type': 'violin-plot', 'index': MATCH}, 'figure'),
        prevent_initial_call=True
    )
    def highlight_selected_point(clickData, fig_dict):
        """Draw a red circle around the selected sample point."""
        if not clickData:
            raise PreventUpdate
        
        # Parse clickData
        point = clickData['points'][0]
        curve_number = point['curveNumber']
        point_index = point['pointNumber']
        
        # Reconstruct figure
        fig = go.Figure(fig_dict)
        
        # Clear any previous selection styling
        for i, trace in enumerate(fig.data):
            # Reset selectedpoints for all traces
            if hasattr(trace, 'selectedpoints'):
                trace.selectedpoints = None
        
        # Apply selection to the clicked trace/point
        if curve_number < len(fig.data):
            fig.data[curve_number].selectedpoints = [point_index]
            
            # Style selected point with red color and larger size, keep unselected fully visible
            fig.data[curve_number].selected = dict(
                marker=dict(
                    color='red',
                    size=14,
                    opacity=1.0
                )
            )
            fig.data[curve_number].unselected = dict(
                marker=dict(opacity=1.0)  # Keep unselected points fully visible
            )
        
        return fig

    # QC Tab callbacks
    @app.callback(
        Output('qc-target-select', 'options'),
        Output('qc-target-select', 'value'),
        Input('analysis-sidebar-menu', 'currentKey'),
        Input('wdir', 'data'),
        State('qc-target-select', 'value'),
        prevent_initial_call=False,
    )
    def update_qc_target_options(current_key, wdir, current_value):
        """Populate QC target dropdown when QC tab is selected."""
        if current_key != 'qc' or not wdir:
            raise PreventUpdate
        
        with duckdb_connection(wdir) as conn:
            if conn is None:
                return [], None
            
            # Get targets that have results
            targets = conn.execute("""
                SELECT DISTINCT t.peak_label 
                FROM targets t
                JOIN results r ON t.peak_label = r.peak_label
                ORDER BY t.peak_label COLLATE NOCASE
            """).fetchall()
            
            if not targets:
                return [], None
            
            options = [{'label': t[0], 'value': t[0]} for t in targets]
            value = current_value if current_value in [t[0] for t in targets] else targets[0][0]
            
            return options, value
            
    @app.callback(
        Output('qc-target-wrapper', 'style'),
        Input('analysis-sidebar-menu', 'currentKey'),
    )
    def toggle_header_visibility(current_key):
        """Toggle visibility of the QC target selector in the header."""
        if current_key == 'qc':
            return {'display': 'flex'}
        return {'display': 'none'}

    @app.callback(
        Output('qc-rt-graph', 'figure'),
        Output('qc-mz-graph', 'figure'),
        Output('qc-spinner', 'spinning'),
        Input('qc-target-select', 'value'),
        Input('analysis-grouping-select', 'value'),
        Input('wdir', 'data'),
        prevent_initial_call=False,
    )
    def generate_qc_plots(peak_label, group_by, wdir):
        """Generate QC plots: RT and m/z in separate figures."""
        if not peak_label or not wdir:
            raise PreventUpdate
        
        from plotly.subplots import make_subplots
        
        with duckdb_connection(wdir) as conn:
            if conn is None:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No data available", paper_bgcolor='white', plot_bgcolor='white')
                return empty_fig, empty_fig, False
            
            group_col = group_by if group_by else 'sample_type'
            
            # Check if peak_mz_of_max exists
            has_peak_mz = False
            try:
                res_cols = [c[0] for c in conn.execute("DESCRIBE results").fetchall()]
                if 'peak_mz_of_max' in res_cols:
                    has_peak_mz = True
            except:
                pass

            mz_col_sql = "r.peak_mz_of_max," if has_peak_mz else "NULL as peak_mz_of_max,"

            query = f"""
                SELECT 
                    r.ms_file_label,
                    r.peak_rt_of_max,
                    {mz_col_sql}
                    r.peak_area,
                    t.rt_min,
                    t.rt_max,
                    t.mz_mean,
                    t.mz_width,
                    COALESCE(s.{group_col}, 'unset') as group_val,
                    s.color,
                    s.acquisition_datetime,
                    ROW_NUMBER() OVER (ORDER BY s.acquisition_datetime NULLS LAST, s.ms_file_label) as sample_order
                FROM results r
                JOIN targets t ON r.peak_label = t.peak_label
                JOIN samples s ON r.ms_file_label = s.ms_file_label
                WHERE r.peak_label = ?
                ORDER BY s.acquisition_datetime NULLS LAST, s.ms_file_label
            """
            
            try:
                df = conn.execute(query, [peak_label]).df()
            except Exception as e:
                logger.error(f"QC query error: {e}")
                err_fig = go.Figure()
                err_fig.update_layout(title=f"Error: {e}", paper_bgcolor='white', plot_bgcolor='white')
                return err_fig, err_fig, False
            
            if df.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No results for selected target", paper_bgcolor='white', plot_bgcolor='white')
                return empty_fig, empty_fig, False
            
            # Use colors from database (samples table)
            unique_groups = df['group_val'].unique()
            
            color_discrete_map = {}
            default_colors = px.colors.qualitative.Plotly
            
            for i, group in enumerate(unique_groups):
                # meaningful color from database?
                group_color = df[df['group_val'] == group]['color'].iloc[0]
                
                if group == 'unset':
                     color_discrete_map[group] = '#bbbbbb'
                elif group_color and group_color != '#BBBBBB':
                     color_discrete_map[group] = group_color
                else:
                     color_discrete_map[group] = default_colors[i % len(default_colors)]

            # X-axis configuration
            x_col = 'sample_order'
            x_title = 'Sample Order (by Acquisition Time)'
            
            # Identify valid datetime data
            if 'acquisition_datetime' in df.columns:
                 # Ensure datetime type
                 try:
                     # Attempt to parse. Coerce errors to NaT
                     dt_series = pd.to_datetime(df['acquisition_datetime'], errors='coerce')
                     if dt_series.notna().any():
                         # We have valid dates. Create a formatted string column for display
                         df['acquisition_time_str'] = dt_series.dt.strftime('%Y-%m-%d %H:%M')
                         x_col = 'acquisition_time_str'
                         x_title = 'Acquisition Time'
                 except Exception:
                     pass
            
            # Get bounds
            mz_mean = df['mz_mean'].iloc[0] if pd.notna(df['mz_mean'].iloc[0]) else None
            
            # === RT FIGURE ===
            fig_rt = go.Figure()
            
            # === PEAK AREA FIGURE ===
            fig_mz = go.Figure()
            
            # Add traces for each group
            for group in unique_groups:
                group_df = df[df['group_val'] == group]
                base_color = color_discrete_map.get(group, '#888888')
                
                # Use per-sample color for scatter points ONLY if grouping by sample_type
                if (not group_by or group_by == 'sample_type') and 'color' in group_df.columns and group_df['color'].notna().all():
                    scatter_colors = group_df['color']
                else:
                    scatter_colors = base_color

                # === RT SCATTER ===
                fig_rt.add_trace(
                    go.Scatter(
                        x=group_df[x_col],
                        y=group_df['peak_rt_of_max'],
                        mode='markers',
                        name=str(group),
                        marker=dict(color=scatter_colors, size=6),
                        text=group_df['ms_file_label'],
                        hovertemplate='%{text}<br>RT: %{y:.2f}s<extra></extra>',
                        legendgroup=str(group),
                        showlegend=True,
                    )
                )
                
                # === PEAK AREA SCATTER ===
                fig_mz.add_trace(
                    go.Scatter(
                        x=group_df[x_col],
                        y=group_df['peak_area'],
                        mode='markers',
                        name=str(group),
                        marker=dict(color=scatter_colors, size=6),
                        text=group_df['ms_file_label'],
                        hovertemplate='%{text}<br>Int: %{y:.2e}<extra></extra>',
                        legendgroup=str(group),
                        showlegend=True, 
                    )
                )

            # Common layout updates
            layout_common = dict(
                legend=dict(
                    orientation='v',
                    yanchor='top',
                    y=0.95,
                    xanchor='right',
                    x=-0.15,
                    font=dict(size=10),
                ),
                hovermode='closest',
                margin=dict(l=0, r=10, t=50, b=40),
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=280,
            )

            # Update RT Figure Layout
            fig_rt.update_layout(
                **layout_common,
                yaxis_title='RT (sec)',
                xaxis_title="", # Separation of axes, might prefer empty here
                xaxis=dict(showticklabels=True)
            )

            # Update Peak Area Figure Layout
            fig_mz.update_layout(
                **layout_common,
                yaxis_title='Peak Area',
                xaxis_title=x_title,
                xaxis=dict(showticklabels=True)
            )
            
            return fig_rt, fig_mz, False

    @app.callback(
        Output('qc-chromatogram', 'figure', allow_duplicate=True),
        Input('qc-chromatogram', 'relayoutData'),
        State('qc-chromatogram', 'figure'),
        prevent_initial_call=True
    )
    def update_qc_chromatogram_zoom(relayout, figure_state):
        """
        When user zooms on X-axis, auto-fit Y-axis to visible data range.
        This matches the behavior in the optimization modal chromatogram.
        """
        if not relayout or not figure_state:
            raise PreventUpdate

        # Check for X-axis zoom event
        x_range = (relayout.get('xaxis.range[0]'), relayout.get('xaxis.range[1]'))
        y_range = (relayout.get('yaxis.range[0]'), relayout.get('yaxis.range[1]'))
        
        # If user explicitly set Y range, respect it
        if y_range[0] is not None and y_range[1] is not None:
            raise PreventUpdate
        
        # Only act when X-axis zoom happens
        if x_range[0] is None or x_range[1] is None:
             # handle autosize button which resets axes
             if 'xaxis.autorange' in relayout and relayout['xaxis.autorange']:
                  # Ideally we'd recalculate full range, but preventing update is safer than erroring
                  # or we can let the full redraw happen via other means.
                  raise PreventUpdate
             raise PreventUpdate
        
        # Auto-fit Y-axis to visible data in the X range
        from dash import Patch
        fig_patch = Patch()
        
        traces = figure_state.get('data', [])
        y_calc = _calc_y_range_numpy(traces, x_range[0], x_range[1], is_log=False)
        
        if y_calc:
            fig_patch['layout']['xaxis']['range'] = [x_range[0], x_range[1]]
            fig_patch['layout']['xaxis']['autorange'] = False
            fig_patch['layout']['yaxis']['range'] = y_calc
            fig_patch['layout']['yaxis']['autorange'] = False
            return fig_patch
        
        raise PreventUpdate
    

    @app.callback(
        Output('qc-selected-sample', 'data'),
        Input('qc-rt-graph', 'clickData'),
        Input('qc-mz-graph', 'clickData'),
        Input('qc-target-select', 'value'),
        State('wdir', 'data'),
        prevent_initial_call=False,
    )
    def update_qc_selected_sample_store(rt_click, mz_click, peak_label, wdir):
        """
        Coordinator callback: Updates the selected sample store.
        Triggered by graph clicks or target change.
        """
        from dash import callback_context, no_update
        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
        
        ms_file_label = None
        
        # 1. Handle Graph Clicks
        if triggered_id in ['qc-rt-graph', 'qc-mz-graph']:
            clickData = rt_click if triggered_id == 'qc-rt-graph' else mz_click
            if clickData:
                try:
                    point = clickData['points'][0]
                    if 'text' in point:
                        ms_file_label = point['text']
                    elif 'customdata' in point:
                        ms_file_label = point['customdata'][0] if isinstance(point['customdata'], list) else point['customdata']
                except Exception:
                    pass
            # If a click happened but we failed to parse, or if we successfully parsed, return that label
            # (even if None, though usually it won't be if clickData exists)
            return ms_file_label
            
        # 2. Handle Target Change / Initial Load (Default selection logic)
        # Only run this if we didn't get a click (i.e. triggered by dropdown or initial load)
        if wdir and peak_label:
             with duckdb_connection(wdir) as conn:
                if conn:
                    try:
                        # Pick a sample with high contrast (likely to have a peak)
                        top_sample = conn.execute("""
                            SELECT ms_file_label 
                            FROM chromatograms
                            WHERE peak_label = ?
                            ORDER BY (list_max(intensity) - list_min(intensity)) DESC
                            LIMIT 1
                        """, [peak_label]).fetchone()
                        if top_sample:
                            ms_file_label = top_sample[0]
                    except:
                        pass
        
        return ms_file_label

    @app.callback(
        Output('qc-chromatogram', 'figure'),
        Output('qc-chromatogram-container', 'style'),
        Input('qc-selected-sample', 'data'),
        Input('qc-log-scale-switch', 'checked'),
        State('qc-target-select', 'value'),
        State('analysis-grouping-select', 'value'),
        State('wdir', 'data'),
        State('qc-rt-graph', 'figure'),
        prevent_initial_call=False,
    )
    def update_qc_chromatogram(ms_file_label, log_scale, peak_label, group_by_col, wdir, rt_fig):
        """Update the chromatogram plot based on the selected sample store."""
        
        if not ms_file_label or not wdir or not peak_label:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                xaxis=dict(visible=False), 
                yaxis=dict(visible=False), 
                paper_bgcolor='white', 
                plot_bgcolor='white'
            )
            return empty_fig, {'display': 'block', 'width': 'calc(43% - 6px)'}

         # Fetch data
        with duckdb_connection(wdir) as conn:
            if conn is None:
                 return dash.no_update, dash.no_update
            
            # 1. Get RT span info for the target
            rt_info = conn.execute("SELECT rt_min, rt_max FROM targets WHERE peak_label = ?", [peak_label]).fetchone()
            rt_min, rt_max = rt_info if rt_info else (None, None)

            # 2. Identify neighbors if a grouping column is selected
            neighbor_files = []
            group_val = None
            group_label = GROUP_LABELS.get(group_by_col, group_by_col or 'Sample')

            if group_by_col:
                try:
                    group_val_query = f'SELECT "{group_by_col}" FROM samples WHERE ms_file_label = ?'
                    row = conn.execute(group_val_query, [ms_file_label]).fetchone()
                    if row:
                        group_val = row[0]
                        if group_val is None:
                             neighbors_query = f"""
                                SELECT ms_file_label, color 
                                FROM samples 
                                WHERE "{group_by_col}" IS NULL AND ms_file_label != ? 
                                ORDER BY random() LIMIT 10
                            """
                             neighbor_files = conn.execute(neighbors_query, [ms_file_label]).fetchall()
                        else:
                            neighbors_query = f"""
                                SELECT ms_file_label, color 
                                FROM samples 
                                WHERE "{group_by_col}" = ? AND ms_file_label != ? 
                                ORDER BY random() LIMIT 10
                            """
                            neighbor_files = conn.execute(neighbors_query, [group_val, ms_file_label]).fetchall()
                except Exception:
                    pass

            display_val = group_val
            if group_by_col and not group_val:
                 display_val = f"{group_label} (unset)"

            # Resolve color from QC RT graph if available (to match scatter plot colors)
            group_color_override = None
            if rt_fig:
                # Determine the group name as it appears in the scatter plot legend
                # In generate_qc_plots, name=str(group_val) where None becomes 'unset'
                lookup_name = str(group_val) if group_val is not None else 'unset'
                
                for trace in rt_fig.get('data', []):
                    if trace.get('name') == lookup_name:
                         # Found the group trace, grab its color
                         marker_color = trace.get('marker', {}).get('color')
                         if isinstance(marker_color, str):
                             group_color_override = marker_color
                         break

            # 3. Fetch chromatograms
            files_to_fetch = [ms_file_label] + [n[0] for n in neighbor_files]
            placeholders = ','.join(['?'] * len(files_to_fetch))
            
            chrom_query = f"""
                SELECT c.ms_file_label, c.scan_time, c.intensity, s.color
                FROM chromatograms c
                JOIN samples s ON c.ms_file_label = s.ms_file_label
                WHERE c.peak_label = ? AND c.ms_file_label IN ({placeholders})
            """
            
            chrom_data = conn.execute(chrom_query, [peak_label] + files_to_fetch).fetchall()
            
            if not chrom_data:
                fig = go.Figure()
                fig.add_annotation(text="No chromatogram data found", showarrow=False)
                return fig, {'display': 'block', 'width': 'calc(43% - 6px)', 'height': '100%'}

            data_map = {row[0]: row for row in chrom_data}
            
            fig = go.Figure()

            # Plot neighbors (background)
            for n_label, n_color in neighbor_files:
                if n_label in data_map:
                    _, scan_times, intensities, _ = data_map[n_label]
                    if len(intensities) > 0 and min(intensities) == max(intensities): continue
                    
                    if group_color_override:
                        n_color = group_color_override
                    elif group_by_col and group_val is None: 
                        n_color = '#bbbbbb'
                    
                    if log_scale: intensities = np.log2(np.array(intensities) + 1)

                    fig.add_trace(go.Scatter(
                        x=scan_times, y=intensities, mode='lines',
                        name=str(display_val), legendgroup=str(display_val), showlegend=False,
                        line=dict(width=1, color=n_color), opacity=0.4,
                        hovertemplate=f"<b>{n_label}</b><br>Scan Time: %{{x:.2f}}<br>Intensity: %{{y:.2e}}<extra>{display_val}</extra>"
                    ))

            # Plot main sample
            if ms_file_label in data_map:
                _, scan_times, intensities, main_color = data_map[ms_file_label]
                
                # Check for flat line only within RT window of the target
                check_intensities = intensities
                if rt_min is not None and rt_max is not None:
                     # Filter checking logic to just the RT span
                     check_intensities = [i for t, i in zip(scan_times, intensities) if rt_min <= t <= rt_max]
                     # If the RT span is very narrow or empty, check_intensities might be empty.
                     # In that case, we might fallback to checking everything or assume it's flat.
                     # Let's assume if we have no data in the window, it's effectively "no signal".
                
                is_flat = False
                if check_intensities:
                    is_flat = min(check_intensities) == max(check_intensities)
                elif rt_min is not None:
                    # No data points in the window?
                    is_flat = True
                
                if is_flat:
                     fig.add_annotation(text="Selected sample has no valid signal (flat line)", showarrow=False)
                else:
                    if group_color_override:
                        main_color = group_color_override
                    elif group_by_col and not group_val: 
                        main_color = '#bbbbbb'
                    legend_name = str(display_val) if group_by_col else ms_file_label
                    if log_scale: intensities = np.log2(np.array(intensities) + 1)

                    fig.add_trace(go.Scatter(
                        x=scan_times, y=intensities, mode='lines',
                        name=legend_name, legendgroup=legend_name, showlegend=True,
                        hovertemplate=f"<b>{ms_file_label}</b><br>Scan Time: %{{x:.2f}}<br>Intensity: %{{y:.2e}}<extra>{legend_name}</extra>",
                        line=dict(width=2, color=main_color), fill='tozeroy', opacity=1.0
                    ))
            
            # Add RT span
            if rt_min is not None and rt_max is not None:
                fig.add_vrect(x0=rt_min, x1=rt_max, fillcolor="green", opacity=0.1, layer="below", line_width=0)

            # Auto-range logic
            x_range_min, x_range_max = None, None
            if rt_min is not None and rt_max is not None:
                padding = 5
                x_range_min, x_range_max = rt_min - padding, rt_max + padding
                # Clamp
                all_x = []
                for t in fig.data:
                    if hasattr(t, 'x') and t.x: all_x.extend(t.x)
                if all_x:
                    x_range_min = max(x_range_min, min(all_x))
                    x_range_max = min(x_range_max, max(all_x))

            # Y-range calculation
            y_range = None
            if x_range_min is not None and x_range_max is not None:
                traces_data = [{'x': list(t.x), 'y': list(t.y)} for t in fig.data if hasattr(t, 'x') and hasattr(t, 'y')]
                y_range = _calc_y_range_numpy(traces_data, x_range_min, x_range_max, is_log=False)

            title_label = ms_file_label
            if len(title_label) > 50: title_label = title_label[:20] + "..." + title_label[-20:]

            y_title = "Intensity (Log2)" if log_scale else "Intensity"
            fig.update_layout(
                title=dict(text=f"{title_label}", font=dict(size=12)),
                xaxis_title="Scan Time (s)", yaxis_title=y_title,
                xaxis_title_font=dict(size=16), yaxis_title_font=dict(size=16),
                xaxis_tickfont=dict(size=12), yaxis_tickfont=dict(size=12),
                template="plotly_white", margin=dict(l=50, r=20, t=90, b=80),
                height=450, showlegend=True,
                legend=dict(title=dict(text=f"{group_label}: ", font=dict(size=13)), font=dict(size=12), orientation='h', y=-0.3, x=0),
                xaxis=dict(range=[x_range_min, x_range_max] if x_range_min else None, autorange=x_range_min is None),
                yaxis=dict(range=y_range if y_range else None, autorange=y_range is None),
            )
            return fig, {'display': 'block', 'width': 'calc(43% - 6px)', 'marginTop': '75px'}

    @app.callback(
        Output('qc-rt-graph', 'figure', allow_duplicate=True),
        Output('qc-mz-graph', 'figure', allow_duplicate=True),
        Input('qc-selected-sample', 'data'),
        State('qc-rt-graph', 'figure'),
        State('qc-mz-graph', 'figure'),
        prevent_initial_call=True
    )
    def highlight_qc_sample(ms_file_label, fig_rt_dict, fig_mz_dict):
        """Draw a red circle around the selected sample point in both graphs."""
        from dash import callback_context, no_update
        
        if not ms_file_label:
            return no_update, no_update
        
        # We search matching text in traces. This is safer than relying on clickData curve/point numbers
        # which aren't available if the update comes from the Store (initial load or previous click).
        # We need to find where this label is.
        # But wait - we don't have easy point index mapping unless we iterate data.
        # Efficient approach: we know 'text' or 'customdata' holds the label.
        
        figs = []
        for fig_data in [fig_rt_dict, fig_mz_dict]:
             if not fig_data:
                  figs.append(no_update)
                  continue

             fig = go.Figure(fig_data)
             
             found = False
             # Clear previous
             for i, trace in enumerate(fig.data):
                  if hasattr(trace, 'selectedpoints'):
                       trace.selectedpoints = None
                  
                  # Search for point
                  if not found and hasattr(trace, 'text') and trace.text:
                      # If text is tuple/list
                      try:
                          # trace.text might be a tuple of strings if there are many points
                          if ms_file_label in trace.text:
                               # Find index
                               # Assuming trace.text is an iterable of strings
                               p_index = list(trace.text).index(ms_file_label)
                               
                               trace.selectedpoints = [p_index]
                               trace.selected = dict(marker=dict(color='red', size=10, opacity=1.0))
                               trace.unselected = dict(marker=dict(opacity=0.6))
                               found = True
                      except:
                          pass

             figs.append(fig)
             
        return figs[0], figs[1]
