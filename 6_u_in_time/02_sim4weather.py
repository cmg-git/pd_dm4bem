#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:24:46 2023

@author: cghiaus

Simulation for weather data

Based on
`4_tc2ss/check_pd_dm4bem_tc2ss_sym2num.py`
"""

import pandas as pd
import pd_dm4bem


# Dissasembled thermal circuits
folder_path = "bldg/sym"
TCd = pd_dm4bem.bldg2TCd(folder_path, TC_auto_number=True)

# For non auto-numbering of thermal circuits TC
# TCd = pd_dm4bem.bldg2TCd(folder_path, TC_auto_number=False)

# Assembled thermal circuit from assembly_matrix.csv
ass_mat = pd.read_csv(folder_path + '/assembly_matrix.csv')
TCmat = pd_dm4bem.assemble_TCd_matrix(TCd, ass_mat)
# pd_dm4bem.print_TC(TCm)

# Assembled thermal circuit from assembly_lists.csv')
ass_lists = pd.read_csv(folder_path + '/assembly_lists.csv')
ass_mat = pd_dm4bem.assemble_lists2matrix(ass_lists)
TClist = pd_dm4bem.assemble_TCd_matrix(TCd, ass_mat)
# pd_dm4bem.print_TC(TCl)

print('\nState-space from TC')
[As, Bs, Cs, Ds, us] = pd_dm4bem.tc2ss(TClist)
