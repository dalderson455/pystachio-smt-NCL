#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

#import dash
#import dash_html_components as html

from dash import html
from dash import dcc

from dash_ui.app import app
import dash_ui.navbar
import dash_ui.img_pane
import dash_ui.info_pane
import dash_ui.sidebar
import dash_ui.footer

def launch_app(params):
    app.title = "PySTACHIO"
    app.layout = html.Div(id='dash-app',
            children=[
                dash_ui.navbar.layout(),
                    dash_ui.sidebar.layout(params),
                html.Div(id='main-content', children=[
                dash_ui.img_pane.layout(params),
                    dash_ui.info_pane.layout(),
                ]),
                dash_ui.footer.layout()
                ])

    app.run_server(debug=True)

