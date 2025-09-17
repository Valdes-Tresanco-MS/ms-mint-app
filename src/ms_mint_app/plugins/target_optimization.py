import json
import uuid
import math
import logging

from plotly_resampler import FigureResampler
from tqdm import tqdm

import dash
from dash import html, dcc, no_update, Patch
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import feffery_antd_components as fac

from .. import tools as T
from ..plugin_interface import PluginInterface

_label = "Optimization"

class TargetOptimizationPlugin(PluginInterface):
    def __init__(self):
        self._label = _label
        self._order = 6
        print(f'Initiated {_label} plugin')

    def layout(self):
        return _layout

    def callbacks(self, app, fsc, cache):
        callbacks(app, fsc, cache)
    
    def outputs(self):
        return _outputs


info_txt = """
Creating chromatograms from mzXML/mzML files can last 
a long time the first time. Try converting your files to 
_feather_ format first.'
"""

def create_preview_peakshape_plotly(
    ms_files, mz_mean, mz_width, rt, rt_min, rt_max, wdir, peak_label, colors, log
):
    """Create peak shape previews."""
    logging.info(f'Create_preview_peakshape {peak_label}')

    fig = FigureResampler(go.Figure())
    y_max = -999999
    y_min = 10e10


    for ms_file in ms_files:
        color = colors[T.filename_to_label(ms_file)]
        if color is None or color == "":
            color = "grey"
        fn_chro = T.get_chromatogram(ms_file, mz_mean, mz_width, wdir)
        y_max = max(y_max, fn_chro["intensity"].max())
        y_min = min(y_min, fn_chro["intensity"].min())

        fn_chro = fn_chro[(rt_min < fn_chro["scan_time"]) & (fn_chro["scan_time"] < rt_max)]
        fig.add_trace(
            go.Scatter(
                       mode='lines', line=dict(color=color, width=1),
                       name=T.filename_to_label(ms_file),
                       hoverinfo='skip'), hf_x=fn_chro["scan_time"], hf_y=fn_chro["intensity"]
        )

    fig.add_vline(rt, line=dict(color='black', width=2))
    fig.update_yaxes(range=[y_min, y_max])

    if log:
        fig.update_yaxes(type="log")

    fig.update_layout(
        title=dict(
            text=peak_label,
            font={"size": 12},
            subtitle=dict(
                text=f"m/z={mz_mean:.2f}",
                font=dict(color="gray", size=10),
            ),
        ),
        margin=dict(l=30, r=0, t=50, b=15),
        # height=100,
        hovermode=False,
        dragmode=False,
        xaxis=dict(
            # visible=False,
            fixedrange=True,
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            # visible=False,
            fixedrange=True,
            tickfont=dict(size=9)
        ),
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None,
    )
    return fig


def create_plot(*, ms_files, ms_files_selection, checkedkeys, mz_mean, mz_width, wdir, rt_min, rt, rt_max, label,
                log, colors):
    """Crear gráfico con rango y línea central"""
    fig = FigureResampler(go.Figure())
    fig.layout.hovermode = "closest"

    fig.update_layout(
        yaxis_title="Intensity",
        xaxis_title="Scan Time [s]",
        # xaxis=dict(rangeslider=dict(visible=True, thickness=0.33)),
        title=label
    )
    if log:
        fig.update_yaxes(type="log")

    fig.add_vline(
        rt, line=dict(color='black', width=3), annotation=dict(text="RT", showarrow=True, bgcolor="white", font=dict(
                color="black", size=14, weight="bold",
            ),)
    )
    fig.add_vrect(
        x0=rt_min, x1=rt_max, line_width=0, fillcolor="green", opacity=0.1
    )

    slider_min = 999999
    slider_max = -999999
    n_files = len(ms_files_selection)
    for i, ms_file in tqdm(enumerate(ms_files), total=n_files, desc="PKO-figure"):
        color = colors[T.filename_to_label(ms_file)]
        if color is None or color == "":
            color = "grey"
        ms_label = T.filename_to_label(ms_file)
        chrom = T.get_chromatogram(ms_file, mz_mean, mz_width, wdir)
        slider_min = min(slider_min, chrom["scan_time"].min())
        slider_max = max(slider_max, chrom["scan_time"].max())

        print(f"{ms_label = }")
        print(f"{checkedkeys = }")

        fig.add_trace(
            go.Scattergl(name=ms_label,
                       visible='legendonly' if ms_label not in checkedkeys else True,
                         line_color=color,
                         ), hf_x=chrom["scan_time"], hf_y=chrom["intensity"],
        )
    fig.update_layout(hoverlabel=dict(namelength=-1))
    fig.layout.xaxis.range = [slider_min, slider_max]

    return fig, math.floor(slider_min), math.ceil(slider_max)


config = {
    'scrollZoom': True,             # allows scroll wheel zooming
    'displayModeBar': True,         # show toolbar
    'modeBarButtonsToAdd': [],      # no need to add zoom/pan – already present
    'displaylogo': False
}

_layout = dbc.Container([
    dbc.Row([
        # Side Panel for Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sample Type"),
                dbc.CardBody([
                    html.Div([
                        # Tree que se expande
                        html.Div([
                            fac.AntdTree(
                                id='sample-type-tree',
                                treeData=[],
                                multiple=True,
                                checkable=True,
                                defaultExpandAll=True,
                            )],
                            style={
                                'flex': '1',
                                'overflow': 'auto',
                                'minHeight': '0'
                            }
                        ),
                        html.Div([
                            html.Label("Selection"),
                            dcc.Dropdown(
                                id="card-plot-selection",
                                options={
                                    'all': 'All',
                                    'bookmarked': 'Bookmarked',
                                    'unmarked': 'Unmarked'
                                },
                                value='all',
                                className="mt-2",
                                clearable=False,
                                style={'marginBottom': '80px'}
                            ),
                            dcc.Checklist(
                                id="pko-figure-options",
                                options=[{"value": "log", "label": "  Logarithmic y-scale"}],
                                value=[],
                                className="mt-4 mb-2",
                            )],
                            style={
                                'flex': '0 0 auto',
                                'borderTop': '1px solid #f0f0f0',
                                'paddingTop': '10px'
                            }
                        )],
                        style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'height': '100%'  # Usar toda la altura disponible del CardBody
                        }
                    )],
                    style={
                        'height': '94vh',  # Ajusta según la altura del header
                        'padding': '10px'
                    }
                    # className="overflow-auto",
                )],
                # className="mb-5",
                style={'maxHeight': '98vh'}
            ),
        ], width=2),  # Reduced width for side panel
        
        # Main Content Column
        dbc.Col([
            # Peak Preview Images (Now directly above the main figure)
            html.Div(
                children=[
                    html.Div([
                        html.H6("Getting chromatograms..."),
                        fac.AntdProgress(
                            id='chromatograms-progress',
                            percent=0,
                        )],
                        id='chromatograms-progress-container',
                        style={
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "flexDirection": "column",
                            "minWidth": "200px",
                            "maxWidth": "500px",
                            "margin": "auto",
                        }
                    ),
                    fac.AntdSpin(
                        html.Div([
                            fac.AntdSpace(
                                id="pko-peak-preview-images",
                                style={
                                    "width": "100%",
                                    'overflowY': 'auto',
                                    "height": "94vh",
                                },
                                wrap=True,
                                align='start'
                            ),
                            fac.AntdPagination(
                                id='plot-preview-pagination',
                                defaultPageSize=30,
                                locale='en-us',
                                align='center',
                                showTotalSuffix='targets',
                                hideOnSinglePage=True
                            )]
                        ),
                        style={"height": "100%", "alignContent": "center"},
                        text="Loading plots...",
                        size="large",
                        delay=500,
                        includeProps=['pko-peak-preview-images.children']
                    ),


                    # main modal
                    fac.AntdModal(
                        id="pko-info-modal",
                        visible=False,
                        width="90vw",
                        centered=True,
                        destroyOnClose=True,
                        closable=False,
                        maskClosable=False,
                        children=[
                            html.Div([
                                dcc.Graph(
                                    id='pko-plot',
                                    config={'displayModeBar': True},
                                    style={'width': '100%', 'height': '600px'}
                                ),
                                html.Div(
                                    [
                                        html.Div([
                                            dcc.RangeSlider(
                                                id='rt-range-slider',
                                                step=1,
                                                tooltip={"placement": "bottom", "always_visible": False},
                                                pushable=1,
                                                allowCross=False,
                                                updatemode="drag",
                                            )],
                                            id='rslider',
                                            style={'alignItems': 'center',
                                                   "alignContent": 'center',
                                                   "width": "100%"
                                                   }
                                        ),
                                        html.Div(
                                            id="rt-values-span",
                                            style={"display": "flex",
                                                   "flexDirection": "column",
                                                   "fontWeight": "bold",
                                                   "fontSize": "16px",
                                                   "minWidth": "100px"
                                                   }
                                        )
                                    ],
                                    style={"display": "flex",}
                                ),
                            ]),
                            html.Hr(style={'margin': '10px 0 10px 0'}),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Row([
                                        dbc.Col(
                                            fac.AntdAlert(
                                                message="RT values have been changed. Save or reset the changes.",
                                                type="warning",
                                                showIcon=True,
                                            ),
                                            width=8
                                        ),
                                        dbc.Col(
                                            fac.AntdButton(
                                                "Reset",
                                                id="reset-btn",
                                                icon=fac.AntdIcon(icon='antd-reload'),
                                            ),
                                            width=2,
                                            style={'textAlign': 'right',
                                                   "alignContent": "center"}
                                        ),
                                        dbc.Col(
                                            fac.AntdButton(
                                                "Save",
                                                id="save-btn",
                                                type="primary",
                                                icon=fac.AntdIcon(icon="antd-save")
                                            ),
                                            width=2,
                                            style={'textAlign': 'left',
                                                   "alignContent": "center"}
                                        )],
                                        id='action-buttons-container',
                                        style={
                                            "visibility": "hidden",
                                            'opacity': '0',
                                            'transition': 'opacity 0.3s ease-in-out',
                                        }
                                    ),
                                    width=11
                                ),
                                dbc.Col(
                                    fac.AntdButton(
                                        "Close",
                                        id="close-modal-btn",
                                    ),
                                    width=1,
                                    style={"alignContent": "center",
                                           "textAlign": "right"}
                                ),
                            ]),
                        ]
                    ),
                    # confirmation modal
                    fac.AntdModal(
                        id="confirm-modal",
                        title="Confirm close without saving",
                        visible=False,
                        width=400,
                        children=[
                            fac.AntdParagraph(
                                "Are you sure you want to close this window without saving your changes?"),
                            html.Div([
                                fac.AntdSpace([
                                    fac.AntdButton(
                                        "Cancel",
                                        id="stay-btn"
                                    ),
                                    fac.AntdButton(
                                        "Close without saving",
                                        id="close-without-save-btn",
                                        type="primary",
                                        danger=True
                                    )
                                ])
                            ], style={'textAlign': 'right', 'marginTop': '20px'})
                        ]
                    ),
                    fac.AntdModal(
                        id="delete-modal",
                        title="Delete target",
                        visible=False,
                        closable=False,
                        width=400,
                        renderFooter=True,
                        okText="Delete",
                        okButtonProps={"danger": True},
                        cancelText="Cancel"
                    )

                ],
                style={
                    'height': '98vh',
                    'margin': '0',
                    "alignItems": "center",
                    "alignContent": "center",
                }
            ),
            # Hidden div for image click tracking
            html.Div(id="pko-image-clicked", style={'display': 'none'}),
            html.Div(id="delete-target-clicked", style={'display': 'none'}),
        ], width=10),  # Expanded width for main content
        
    ]),
], fluid=True)

pko_layout_no_data = html.Div(
    [
        dcc.Markdown(
            """### No targets found.
    You did not genefrate a targets yet.
    """
        )
    ]
)

_outputs = html.Div(
    id="pko-outputs",
    children=[
        html.Div(id={"index": "pko-set-rt-output", "type": "output"}),
        html.Div(id={"index": "pko-confirm-rt-output", "type": "output"}),
        html.Div(id={"index": "pko-detect-rt-for-all-output", "type": "output"}),
        html.Div(id={"index": "pko-detect-rtspan-for-all-output", "type": "output"}),
        html.Div(id={"index": "pko-detect-rt-output", "type": "output"}),
        html.Div(id={"index": "pko-detect-rtspan-output", "type": "output"}),
        html.Div(id={"index": "pko-drop-target-output", "type": "output"}),
        html.Div(id={"index": "pko-remove-low-intensity-output", "type": "output"}),
        dcc.Store(id='saved-range', data=[None, None, None]),
        dcc.Store(id='current-range', data=[None, None, None]),
        dcc.Store(id='has-unsaved-changes', data=False),
        dcc.Store(id='chromatograms', data={}),
    ],
)

def layout():
    return _layout


def callbacks(app, fsc, cache, cpu=None):

    # clientside callback for set width rangeslider == plot bglayer
    app.clientside_callback(
        """
        function(relayoutData) {
            const root = document.getElementById("pko-plot");
            if (!root) return window.dash_clientside.no_update;
            console.log(root);
            // plotly DOM: tres divs -> svg -> g.bglayer -> rect
            const bg = root.querySelector("div > div > div svg > g.draglayer > g.xy > rect");
            console.log(bg);
            if (!bg) return window.dash_clientside.no_update;

            const rw = root.offsetWidth;
            const w = bg.width.baseVal.value;
            const pl = bg.x.baseVal.value - 25;
            const pr = rw - pl - w - 150;
            console.log("rw:", rw, "w:", w, "pl:", pl, "pr:", pr);

            if (!isFinite(w) || w <= 0) return window.dash_clientside.no_update;


            return {"paddingLeft": pl + "px", "paddingRight": pr + "px", "width": "100%"};
        }
        """,
        Output("rslider", "style"),
        Input("pko-plot", "relayoutData"),
        prevent_initial_call=True
    )

    @app.callback(
        Output('sample-type-tree', 'treeData'),
        Output('sample-type-tree', 'checkedKeys'),
        Output('sample-type-tree', 'expandedKeys'),
        Input("wdir", "children"),
    )
    def update_sample_type_tree(wdir):
        metadata = T.get_metadata(wdir)
        if metadata is None or 'sample_type' not in metadata.columns:
            return []

        sample_type_dict = {}
        for ms_file,  sample_type in metadata[['ms_file_label', 'sample_type']].values:
            if sample_type in sample_type_dict:
                sample_type_dict[sample_type].append(ms_file)
            else:
                sample_type_dict[sample_type] = [ms_file]

        checkedkeys = metadata[metadata['use_for_optimization']]['ms_file_label'].tolist()

        tree_data = []
        for k, v in sample_type_dict.items():
            children = [{'title': ms, 'key': ms} for ms in v if ms in checkedkeys]
            if children:
                tree_data.append({'title': k, 'key': k, 'children': children})

        expandedKeys = [
            k
            for k, v in sample_type_dict.items()
            if any(ms in checkedkeys for ms in v)
        ]
        return tree_data, checkedkeys, expandedKeys

    @app.callback(
        Output('pko-info-modal', 'visible'),
        Output('pko-info-modal', 'title'),
        Output('has-unsaved-changes', 'data'),

        Input("pko-image-clicked", "children"),
        Input('close-modal-btn', 'nClicks'), # nClicks (antd) != n_clicks (dash)
        Input('close-without-save-btn', 'nClicks'),
        Input('sample-type-tree', 'checkedKeys'),
        State('has-unsaved-changes', 'data'),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def handle_modal_open_close(image_clicked, close_clicks, close_without_save_clicks,
                                checkedkeys, has_changes, wdir):
        """
        Handle opening and closing of the modal dialog for display the selected target plot.
        :param image_clicked:
        :param close_clicks:
        :param close_without_save_clicks:
        :param saved_values:
        :param has_changes:
        :return:
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'pko-image-clicked':

            return (True, image_clicked,
                    # no_update,
                    # saved_values,
                    False)
        elif trigger_id == 'close-without-save-btn':
            # Close modal without saving changes
            return (False, no_update,
                    # no_update,
                    # no_update,
                    False)
        elif trigger_id == 'close-modal-btn':
            # if not has_changes, close it
            if not has_changes:
                return (False, no_update,
                        # no_update,
                        # no_update,
                        False)
            # if it has_changes, don't close it
            return (no_update, no_update,
                    # no_update,
                    # no_update,
                    no_update)
        else:
            print(f"modal {trigger_id = }")

        return (no_update, no_update,
                # no_update,
                # no_update,
                no_update)

    @app.callback(
        Output('confirm-modal', 'visible'),
        Input('close-modal-btn', 'nClicks'),
        State('has-unsaved-changes', 'data'),
        prevent_initial_call=True
    )
    def show_confirm_modal(close_clicks, has_changes):
        return bool(close_clicks and has_changes)

    @app.callback(
        Output('pko-plot', 'figure'),
        Output('action-buttons-container', 'style'),
        Output("rt-range-slider", "min"),
        Output("rt-range-slider", "max"),
        Output("rt-range-slider", "value"),
        Output("rt-range-slider", "marks"),
        Output("rt-range-slider", "tooltip"),
        Output('has-unsaved-changes', 'data', allow_duplicate=True),

        Input("pko-image-clicked", "children"),
        Input("pko-figure-options", "value"),
        Input('rt-range-slider', 'value'),
        Input('sample-type-tree', 'checkedKeys'),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def update_from_slider(image_clicked, options, slider_values, checkedkeys, wdir):

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # reset slider values when is not an update to avoid use stored values from other plots
        if trigger_id != 'rt-range-slider':
            slider_values = None
        if image_clicked is None:
            raise PreventUpdate
        targets = T.get_targets(wdir)
        cols = ["mz_mean", "mz_width", "rt", "rt_min", "rt_max", "peak_label"]
        mz_mean, mz_width, rt, rt_min, rt_max, label = targets.loc[targets['peak_label'] == image_clicked, cols].iloc[0]
        # round values to int because the slider steps are int
        orig_values = [int(rt_min), int(rt), int(rt_max)]

        if trigger_id == 'pko-image-clicked':
            ms_files_fs = {T.filename_to_label(f): f for f in T.get_ms_fns(wdir)}
            metadata = T.get_metadata(wdir)
            samples_selected = [
                ms_files_fs[ss]
                for ss in metadata[metadata['use_for_optimization'] == True]['ms_file_label'].tolist()
                if ss in ms_files_fs
            ]

            ms_files_selection = []
            for ms_name in checkedkeys:
                if ms_name in ms_files_fs:
                    ms_files_selection.append(ms_files_fs[ms_name])

            rt_slider_min, st_slider,rt_slider_max = slider_values or orig_values
            file_colors = T.file_colors(wdir)

            fig, slider_min, slider_max = create_plot(
                ms_files=samples_selected,
                ms_files_selection=ms_files_selection,
                checkedkeys=checkedkeys,
                mz_mean=mz_mean,
                mz_width=mz_width,
                wdir=wdir,
                rt_min=rt_slider_min,
                rt=st_slider,
                rt_max=rt_slider_max,
                label=label,
                log='log' in options,
                colors=file_colors
            )
            slider_marks = {i: str(i) for i in range(slider_min, slider_max, (int(slider_max - slider_min)//5))}
            has_changes = slider_values != orig_values if slider_values else False

            buttons_style = {
                'visibility': 'visible' if has_changes else 'hidden',
                'opacity': '1' if has_changes else '0',
                'transition': 'opacity 0.3s ease-in-out'
            }
            return (fig, buttons_style, slider_min, slider_max,
                    [rt_slider_min, st_slider, rt_slider_max],
                    slider_marks,
                    {"placement": "bottom", "always_visible": False},
                    has_changes,
                    )

        elif trigger_id == 'rt-range-slider':
            patch_fig = Patch()
            if slider_values[0]:
                patch_fig['layout']['shapes'][1]['x0'] = slider_values[0]
            if slider_values[2]:
                patch_fig['layout']['shapes'][1]['x1'] = slider_values[2]
            if slider_values[1]:
                patch_fig['layout']['shapes'][0]['x0'] = slider_values[1]
                patch_fig['layout']['shapes'][0]['x1'] = slider_values[1]
                patch_fig['layout']['annotations'][0]['x'] = slider_values[1]

            has_changes = slider_values != orig_values if slider_values else False
            buttons_style = {
                'visibility': 'visible' if has_changes else 'hidden',
                'opacity': '1' if has_changes else '0',
                'transition': 'opacity 0.3s ease-in-out'
            }

            return (patch_fig, buttons_style, no_update, no_update, no_update, no_update, no_update, has_changes)
        return (no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    @app.callback(
        Output('rt-values-span', 'children'),
        Input('rt-range-slider', 'value'),
        prevent_initial_call=True
    )
    def rt_representation(values):
        rt_min, rt, rt_max = values

        return [html.Span(f"RT-min: {rt_min}", className="rt-value"),
                html.Span(f"RT: {rt}", className="rt-value"),
                html.Span(f"RT-max: {rt_max}", className="rt-value")]


    @app.callback(
        Output('saved-range', 'data', allow_duplicate=True),
        Output('pko-info-modal', 'visible', allow_duplicate=True),
        Output('notifications-container', 'children'),
        Output('has-unsaved-changes', 'data', allow_duplicate=True),
        Output('action-buttons-container', 'style', allow_duplicate=True),

        Input("pko-image-clicked", "children"),
        Input('save-btn', 'nClicks'),
        Input("rt-range-slider", "value"),
        State("has-unsaved-changes", "data"),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def save_changes(image_clicked, save_clicks, current_values, unsaved_changes, wdir):

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'save-btn' and unsaved_changes:
            targets = T.get_targets(wdir)
            rt_min, rt, rt_max = current_values
            peak_label = targets.loc[targets['peak_label'] == image_clicked, "peak_label"].iloc[0]
            T.update_targets(wdir, peak_label, rt_min, rt_max, rt)

            buttons_style = {
                'visibility': 'hidden',
                'opacity': '0',
                'transition': 'opacity 0.3s ease-in-out'
            }

            return (
                current_values,  # Actualizar valores guardados
                False,  # Cerrar modal
                fac.AntdNotification(message=f"RT values saved...\n"
                                             f"RT-min: {current_values[0]:.2f}<br>"
                                             f"RT: {current_values[1]:.2f}<br> "
                                             f"RT-max: {current_values[2]:.2f}",
                                     type="success",
                                     placement='bottom',
                                     showProgress=True,
                                     stack=True
                                     ),
                False,
                buttons_style
            )
        raise PreventUpdate

    @app.callback(
        Output('rt-range-slider', 'value', allow_duplicate=True),
        Output('has-unsaved-changes', 'data', allow_duplicate=True),
        Output('action-buttons-container', 'style', allow_duplicate=True),

        Input("pko-image-clicked", "children"),
        Input('reset-btn', 'nClicks'),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def reset_changes(image_clicked, reset_clicks, wdir):
        """
        Reset the slider values to the saved RT values
        :param image_clicked:
        :param reset_clicks:
        :param wdir:
        :return:
        """
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'reset-btn' and reset_clicks:
            targets = T.get_targets(wdir)
            rt_min, rt, rt_max = targets.loc[targets['peak_label'] == image_clicked, ["rt_min", "rt", "rt_max"]].iloc[0]

            buttons_style = {
                'visibility': 'hidden',
                'opacity': '0',
                'transition': 'opacity 0.3s ease-in-out'
            }
            return [int(rt_min), int(rt), int(rt_max)], False, buttons_style
        return no_update, no_update, no_update

    @app.callback(
        Output('confirm-modal', 'visible', allow_duplicate=True),
        Input('stay-btn', 'nClicks'),
        Input('close-without-save-btn', 'nClicks'),
        prevent_initial_call=True
    )
    def handle_confirm_modal(stay_clicks, close_clicks):
        return False

    # callback to compute or read the chromatograms
    @app.callback(
        Output('chromatograms', 'data'),
        # Output('chromatograms-progress-container', 'style'),
        Input('tab', 'value'),
        Input('chromatograms', 'data'),
        State('wdir', 'children'),
        background=True,
        running=[
            (
                    Output("chromatograms-progress-container", "style"),
                    {"display": "flex",
                     "justifyContent": "center",
                     "alignItems": "center",
                     "flexDirection": "column",
                     "minWidth": "200px",
                     "maxWidth": "500px",
                     "margin": "auto",},
                    {"display": "none"},
            ),
        ],
        progress=[Output("chromatograms-progress", "percent"),
                  ],
        prevent_initial_call=True
    )
    def compute_chromatograms(set_progress, tab, chromatograms, wdir):

        ctx = dash.callback_context
        prop_id = ctx.triggered[0]['prop_id']
        print(f"{prop_id = }")

        if tab != 'Optimization':
            raise PreventUpdate


        metadata = T.get_metadata(wdir)
        samples_selected = metadata[metadata['use_for_optimization'] == True]['ms_file_label'].tolist()
        ms_files_fs = {T.filename_to_label(f): f for f in T.get_ms_fns(wdir)}
        targets = T.get_targets(wdir)
        ms_files_selection = []
        for ms_name in samples_selected:
            if ms_name in ms_files_fs:
                ms_files_selection.append(ms_files_fs[ms_name])

        total = 0
        for target in targets['peak_label']:
            if target not in chromatograms:
                chromatograms[target] = []
            total += len(set(samples_selected) - set(chromatograms[target]))
        data = {}
        count = 0
        for ti, target in targets.iterrows():
            peak_label, mz_mean, mz_width, rt, rt_min, rt_max, bookmark, score = target[
                ["peak_label", "mz_mean", "mz_width", "rt", "rt_min", "rt_max", 'bookmark', 'score']
            ]
            if peak_label not in data:
                data[peak_label] = []
            for i, sample in enumerate(ms_files_selection, start=1):
                if sample not in data[peak_label]:
                    T.get_chromatogram(sample, mz_mean, mz_width, wdir)
                    data[peak_label].append(sample)
                    count += 1
                    set_progress((round(count / total * 100, 2)))


        return data


    @app.callback(
        Output("pko-peak-preview-images", "children"),
        Output('plot-preview-pagination', 'total'),

        Input('chromatograms', 'data'),

        Input('plot-preview-pagination', 'current'),
        Input('plot-preview-pagination', 'pageSize'),

        Input('sample-type-tree', 'checkedKeys'),
        Input("pko-figure-options", "value"),
        Input('card-plot-selection', 'value'),
        Input({"index": "pko-drop-target-output", "type": "output"}, "children"),
        State("wdir", "children"),
    )
    def peak_preview(chromatograms, current_page, page_size, checkedkeys, options, selection, dropped_target, wdir):
        logging.info(f'Create peak previews {wdir}')

        if not chromatograms:
            raise PreventUpdate

        ms_files_fs = {T.filename_to_label(f): f  for f in T.get_ms_fns(wdir)}

        ms_files_selection = []
        for ms_name in checkedkeys:
            if ms_name in ms_files_fs:
                ms_files_selection.append(ms_files_fs[ms_name])

        if not ms_files_selection:
            return (fac.AntdNotification(
                message='No files selected for peak optimization in MS-Files tab. Please, mark some files in the '
                        'Sample Type tree. "use_for_optimization" is use as the initial selection only',
                type="error", duration=None,
                placement='bottom',
                showProgress=True,
                stack=True
            ), no_update)
        logging.info(f"Using {len(ms_files_selection)} files for peak preview. ({ms_files_selection = })")

        targets = T.get_targets(wdir)
        file_colors = T.file_colors(wdir)

        if selection == 'all':
            target_selection = targets
        elif selection == 'bookmarked':
            target_selection = targets.query('bookmark == 1')
        else:
            target_selection = targets.query('bookmark == 0')


        start = (current_page - 1) * page_size
        end = min(current_page * page_size, len(target_selection) + 1)

        plots = []
        for _, row in target_selection.iloc[start:end].iterrows():
            peak_label, mz_mean, mz_width, rt, rt_min, rt_max, bookmark, score = row[
                ["peak_label", "mz_mean", "mz_width", "rt", "rt_min", "rt_max", 'bookmark', 'score']
            ]
            fig = create_preview_peakshape_plotly(
                ms_files_selection,
                mz_mean,
                mz_width,
                rt,
                rt_min,
                rt_max,
                wdir,
                peak_label=peak_label,
                colors=file_colors,
                log='log' in options
            )

            plots.append(
                fac.AntdCard([
                    dcc.Graph(
                        id={'type': 'graph-card-preview', 'index': f"{peak_label}-{uuid.uuid4().hex[:6]}"},
                        figure=fig,
                        style={'height': '150px', 'width': '200px', 'margin': '0px'},
                        config={
                            'displayModeBar': False,
                            'staticPlot': True,  # Totalmente estático para máxima performance
                            'doubleClick': False,
                            'showTips': False,
                            'responsive': False  # Tamaño fijo
                        },
                    ),
                    fac.AntdTooltip(
                        fac.AntdButton(
                            icon=fac.AntdIcon(icon='antd-delete'),
                            type='text',
                            size='small',
                            id={'type': 'delete-target-btn', 'index': peak_label},
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
                                'zIndex': 10,
                                'opacity': '0',
                                'transition': 'opacity 0.3s ease'
                            },
                            className='peak-action-button',
                        ),
                        title='Delete target',
                        color='red',
                        placement='bottom',
                    ),
                    fac.AntdRate(
                        id={'type': 'rate-target-card', 'index': peak_label},
                        count=1,
                        defaultValue=0,
                        value=targets.loc[targets['peak_label'] == peak_label, 'bookmark'].values[0],
                        allowHalf=False,
                        tooltips=['Bookmark this target'],
                        style={'position': 'absolute', 'top': '8px', 'right': '8px', 'zIndex': 10},
                    )],
                    id={'type': 'plot-card-preview', 'index': peak_label},
                    style={'cursor': 'pointer'},
                    styles={'header': {'display': 'none'}, 'body': {'padding': '5px'}},
                    hoverable=True,
                    className='peak-card-container',
                )
            )
        return plots, len(target_selection)

    @app.callback(
        Output("pko-image-clicked", "children"),
        [Input({"type": "plot-card-preview", "index": ALL}, "nClicks")],
        [Input({"type": "delete-target-btn", "index": ALL}, "nClicks")],
        [Input({"type": "rate-target-card", "index": ALL}, "value")],
        prevent_initial_call=True,
    )
    def pko_image_clicked(ndx, dt_ndx, b_ndx):

        if ndx is None or len(ndx) == 0:
            raise PreventUpdate

        ctx = dash.callback_context

        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']
        trigger_type = ctx_trigger['type']

        if len(ctx.triggered) > 1 or trigger_type != "plot-card-preview":
            raise PreventUpdate
        return trigger_id

    @app.callback(
        Output('delete-modal', 'visible'),
        Output('delete-modal', 'children'),
        Output('delete-target-clicked', 'children'),
        Input({"type": "delete-target-btn", "index": ALL}, 'nClicks'),
        prevent_initial_call=True
    )
    def show_delete_modal(delete_clicks):

        ctx = dash.callback_context
        if not delete_clicks or not ctx.triggered:
            raise PreventUpdate
        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']

        if len(dash.callback_context.triggered) > 1:
            raise PreventUpdate

        return True, fac.AntdParagraph(f"Are you sure you want to delete `{trigger_id}` target?"), trigger_id

    @app.callback(
        Output({"index": "pko-drop-target-output", "type": "output"}, "children"),
        Input('delete-modal', 'okCounts'),
        Input('delete-modal', 'cancelCounts'),
        Input('delete-modal', 'closeCounts'),
        Input('delete-target-clicked', 'children'),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def plk_delete(okCounts, cancelCounts, closeCounts, target, wdir):
        if not okCounts or cancelCounts or closeCounts:
            raise PreventUpdate
        targets = T.get_targets(wdir)
        targets = targets[targets['peak_label'] != target]

        T.write_targets(targets, wdir)
        return fac.AntdNotification(message=f"{target} deleted",
                                    type="success",
                                    duration=3,
                                    placement='bottom',
                                    showProgress=True,
                                    stack=True
                                    )

    @app.callback(
        Output('notifications-container', "children", allow_duplicate=True),
        Input({"type": "rate-target-card", "index": ALL}, "value"),
        State('wdir', 'children'),
        prevent_initial_call=True
    )
    def bookmark_target(value, wdir):

        ctx = dash.callback_context
        if not ctx.triggered or len(dash.callback_context.triggered) > 1:
            raise PreventUpdate

        targets = T.get_targets(wdir)

        ctx_trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        trigger_id = ctx_trigger['index']

        if targets['bookmark'].values.tolist() == value:
            raise PreventUpdate

        targets['bookmark'] = value
        T.write_targets(targets, wdir)
        new_value = targets.loc[targets['peak_label'] == trigger_id, 'bookmark'].values[0]

        return fac.AntdNotification(message=f"Target {trigger_id} {'' if new_value else 'un'}bookmarked", duration=3,
                                    placement='bottom', type="success", showProgress=True, stack=True)




