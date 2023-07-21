#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:39:54 2023

@author: cghiaus
"""
import pandas as pd
import os
import pd_dm4bem

folder_path = "bldg_cube1"

file_path = os.path.join(folder_path, "wall_types.csv")
wall_types = pd.read_csv(file_path)

file_path = os.path.join(folder_path, "walls_generic.csv")
walls = pd.read_csv(file_path)

# Thermal circuits from data & type files of walls
TCd_generic = pd_dm4bem.wall2TC(wall_types, walls, prefix="g")

# Print TCd_generic
# for key in TCd_generic.keys():
#     print('Wall:', key)
#     pd_dm4bem.print_TC(TCd_generic[key])

file_path = os.path.join(folder_path, "walls_in.csv")
walls = pd.read_csv(file_path)

# Thermal circuits from data & type files of walls
TCd_in = pd_dm4bem.wall2TC(wall_types, walls, prefix="i")

# Print TCd_in
# for key in TCd_generic.keys():
#     print('Wall:', key)
#     pd_dm4bem.print_TC(TCd_in[key])

# Put TCd together
TCd = TCd_generic.copy()
TCd.update(TCd_in)

for key in TCd.keys():
    print('Wall:', key)
    pd_dm4bem.print_TC(TCd[key])
