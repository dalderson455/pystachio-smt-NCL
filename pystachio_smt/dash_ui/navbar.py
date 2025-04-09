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

def layout():
    navbar = html.Div(id="navbar",
        children=[
            html.H1("PySTACHIO"), ])

    return navbar
