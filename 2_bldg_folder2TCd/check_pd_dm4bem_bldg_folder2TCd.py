#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:39:54 2023

@author: cghiaus
"""
# import pandas as pd
# import os
import pd_dm4bem

folder_path = "bldg_cube1"
TCd = pd_dm4bem.bldg2TCd(folder_path)

for key in TCd.keys():
    print('Thermal circuit:', key)
    pd_dm4bem.print_TC(TCd[key])
