#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:46:19 2023

@author: cghiaus

Example from 03CubicBuilding.ipynb and 041tc2ss_ass.ipynb
treated with:
    dm4bem.tc2ss: without assembling
    pd_dm4bem.tc2ss: with assembling

Use the same notations figure TC.svg

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem
import pd_dm4bem


def input_data_step_response(duration, dt):
    """
    Parameters
    ----------
    n : int
        DESCRIPTION.

    Returns
    -------
    None.

    """
    To = 10 * np.ones(n)
    Ti_sp = 20 * np.ones(n)
    Φ = Qa = Qo = Qi = np.zeros(n)

"""
Obtain the state-space
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

"""
Step response
"""
# time step smaller than dtmax = min(−2/𝜆𝑖)
λ = np.linalg.eig(Al)[0]    # eigenvalues of matrix As
dtmax = 2 * min(-1. / λ)
dt = np.floor(dtmax / 60) * 60   # s

# settling time = 4 * max time constant
time_const = np.array([int(x) for x in sorted(-1 / λ)])
t_settle = 4 * max(-1 / λ)

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
n = int(np.floor(duration / dt))        # number of time steps
t = np.arange(0, n * dt, dt)            # time vector for n time steps

# inputs
To = 10 * np.ones(n)
Ti_sp = 20 * np.ones(n)
Φ = Qa = Qo = Qi = np.zeros(n)

# construct input vector from "ul"
# u = pd.concat([pd.Series(eval(col), name=col) for col in ul], axis=1)

input_data = {'To': To, 'Ti_sp': Ti_sp, 'Φ': Φ, 'Qa': Qa, 'Qo': Qo, 'Qi': Qi}
u = pd_dm4bem.inputs_in_time(ul, input_data)

# initial conditions
n_s = Al.shape[0]                       # number of state variables
θ_exp = np.zeros([n_s, t.shape[0]])     # explicit Euler in time t
θ_imp = np.zeros([n_s, t.shape[0]])     # implicit Euler in time t


I = np.eye(n_s)                         # identity matrix

Bl = Bl.rename(columns=ul)
Dl = Dl.rename(columns=ul)

# time integration
for k in range(n - 1):
    θ_exp[:, k + 1] = (I + dt * Al) @\
        θ_exp[:, k] + dt * Bl @ u.iloc[k]
    θ_imp[:, k + 1] = np.linalg.inv(I - dt * Al) @\
        (θ_imp[:, k] + dt * Bl @ u.iloc[k])

# outputs
y_exp = Cl @ θ_exp + Dl @  u.T
y_imp = Cl @ θ_imp + Dl @  u.T

# plot
fig, ax = plt.subplots()
ax.plot(t / 3600, y_exp.values.T, t / 3600, y_imp.values.T)
ax.set(xlabel='Time, $t$ / h',
       ylabel='Temperatue, $θ_i$ / °C',
       title='Step input: outdoor temperature $T_o$')
ax.legend(['θia - explicit', 'θwi - explicit',
           'θia - implicit', 'θwi - implicit'])
ax.grid()
plt.show()
