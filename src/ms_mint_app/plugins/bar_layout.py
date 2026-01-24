
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
