#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Edward Higgins <ed.higgins@york.ac.uk>
#
# Distributed under terms of the MIT license.

"""

"""
from dash import html
from dash import dcc

def layout():
    footer = html.Div(id="footer",
        children=[
            """
            PySTACHIO-SMT was developed by Edward Higgins and Jack Shepherd in
            collaboration with the Physics of Life group at the University of
            York. For more information, go to the
            """,
            dcc.Link('Github page', href='https://www.github.com/ejh516/pystachio-smt'), 
            "."
            ])
    return footer
