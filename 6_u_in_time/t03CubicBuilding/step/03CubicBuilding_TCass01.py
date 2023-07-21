#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 13:57:13 2023

@author: cghiaus

Example from 03CubicBuilding.ipynb and 041tc2ss_ass.ipynb
treated with:
    dm4bem.tc2ss: without assembling
    pd_dm4bem.tc2ss: with assembling

Follows the notatons from 03CubicBuilding.ipynb

"""
import numpy as np
import pandas as pd
import dm4bem
import pd_dm4bem

"""
dm4bem.tc2ss: without assembling
"""
TC0 = pd.read_csv("./bldg/TC0.csv")
TC1 = pd.read_csv("./bldg/TC1.csv")
TC2 = pd.read_csv("./bldg/TC2.csv")
TC3 = pd.read_csv("./bldg/TC3.csv")

wS = pd.read_csv("./bldg/walls_out.csv")

w = pd.read_csv("./bldg/wall_types.csv")
w0 = w[w['type'] == 0]
w0.set_index('Material', inplace=True)

A = np.zeros([12, 8])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
A[5, 4], A[5, 5] = -1, 1    # branch 5: node 4 -> node 5
A[6, 4], A[6, 6] = -1, 1    # branch 6: node 4 -> node 6
A[7, 5], A[7, 6] = -1, 1    # branch 7: node 5 -> node 6
A[8, 7] = 1                 # branch 8: -> node 7
A[9, 5], A[9, 7] = 1, -1    # branch 9: node 5 -> node 7
A[10, 6] = 1                # branch 10: -> node 6
A[11, 6] = 1                # branch 11: -> node 6


# G = np.array([1125.0, 30.375, 30.375, 630.0, 630.0, 44.78682363, 360.0,
#               72.0, 165.78947368, 630.0, 9.0, 0.0])

G = np.zeros(A.shape[0])
G[0] = wS.loc[0]['h0'] * wS.loc[0]['Area']
G[1] = G[2] = w0.loc[
    'Insulation']['Conductivity'] / (w0.loc[
        'Insulation']['Width'] / 2) * wS.loc[0]['Area']
G[3] = G[4] = w0.loc[
    'Concrete']['Conductivity'] / (w0.loc[
        'Concrete']['Width'] / 2) * wS.loc[0]['Area']
G[5] = TC0.loc[0]['G']
G[6] = wS.loc[0]['h1'] * wS.loc[0]['Area']
G[7] = TC1.loc[2]['G']
G[8] = TC1.loc[0]['G']
G[9] = TC1.loc[1]['G']
G[10] = TC2.loc[0]['G']
G[11] = TC3.loc[0]['G']

# C = np.array([0., 239580., 0., 18216000., 0., 0., 32400., 1089000.])

C = np.zeros(A.shape[1])
C[1] = w0.loc[
    'Insulation']['Density'] * w0.loc[
        'Insulation']['Specific heat'] * w0.loc[
            'Insulation']['Width'] * wS.loc[0]['Area']

C[3] = w0.loc[
    'Concrete']['Density'] * w0.loc[
        'Concrete']['Specific heat'] * w0.loc[
            'Concrete']['Width'] * wS.loc[0]['Area']

C[6] = TC2.loc[TC2['A'] == 'C', 'θ0']
C[7] = TC1.loc[TC1['A'] == 'C', 'θg']

b = np.zeros(12)        # branches
b[[0, 8, 10, 11]] = 1   # branches with temperature sources

f = np.zeros(8)         # nodes
f[[0, 4, 6, 7]] = 1     # nodes with heat-flow sources

y = np.zeros(8)         # nodes
y[[4, 6]] = 1           # nodes: in wall surface, in air

[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, np.diag(G), np.diag(C), b, f, y)
print('As = \n', As, '\n')
print('Bs = \n', Bs, '\n')
print('Cs = \n', Cs, '\n')
print('Ds = \n', Ds, '\n')

"""
pd_dm4bem.tc2ss: with assembling
"""
# Dissasembled thermal circuits
folder_path = "bldg"
TCd = pd_dm4bem.bldg2TCd(folder_path,
                         TC_auto_number=True)
# For non auto-numbering of thermal circuits TC
# TCd = pd_dm4bem.bldg2TCd(folder_path, TC_auto_number=False)

# Assembled thermal circuit
# from assembly_matrix.csv')
ass_mat = pd.read_csv(folder_path + '/assembly_matrix.csv')
TCm = pd_dm4bem.assemble_TCd_matrix(TCd, ass_mat)

# from assembly_lists.csv'
ass_lists = pd.read_csv(folder_path + '/assembly_lists.csv')
ass_mat = pd_dm4bem.assemble_lists2matrix(ass_lists)
TCl = pd_dm4bem.assemble_TCd_matrix(TCd, ass_mat)

# State-space from TC
[Al, Bl, Cl, Dl, ul] = pd_dm4bem.tc2ss(TCl)
