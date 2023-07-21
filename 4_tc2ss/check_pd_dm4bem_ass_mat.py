#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:02:47 2023

@author: cghiaus
"""
import pandas as pd
import pd_dm4bem

print('\nDissasembled thermal circuits')
folder_path = "bldg"
TCd = pd_dm4bem.bldg2TCd(folder_path)

for key in TCd.keys():
    print('Thermal circuit:', key)
    pd_dm4bem.print_TC(TCd[key])

print('\nAssembled thermal circuit from assembly_matrix.csv')
ass_mat = pd.read_csv(folder_path + '/assembly_matrix.csv')
TC = pd_dm4bem.assemble_TCd_matrix(TCd, ass_mat)
pd_dm4bem.print_TC(TC)

print('\nAssembled thermal circuit from assembly_lists.csv')
ass_lists = pd.read_csv(folder_path + '/assembly_lists.csv')
ass_mat = pd_dm4bem.assemble_lists2matrix(ass_lists)
TC = pd_dm4bem.assemble_TCd_matrix(TCd, ass_mat)
pd_dm4bem.print_TC(TC)
