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

"""
dm4bem.tc2ss: without assembling
# """
TC0 = pd.read_csv("./bldg/TC0.csv")
TC1 = pd.read_csv("./bldg/TC1.csv")
TC2 = pd.read_csv("./bldg/TC2.csv")
TC3 = pd.read_csv("./bldg/TC3.csv")

wS = pd.read_csv("./bldg/walls_out.csv")

w = pd.read_csv("./bldg/wall_types.csv")

w0 = w[w['type'] == 0]
w0.set_index('Material', inplace=True)



A = np.zeros([12, 8])       # n° of branches X n° of nodes
A[0, 7], A[0, 1] = -1, 1
A[1, 0] = 1
A[2, 0], A[2, 1] = -1, 1
A[3, 1], A[3, 2] = -1, 1
A[4, 2] = 1
A[5, 2] = 1
A[6, 3] = 1
A[7, 3], A[7, 4] = -1, 1
A[8, 4], A[8, 5] = -1, 1
A[9, 5], A[9, 6] = -1, 1
A[10, 6], A[10, 7] = -1, 1
A[11, 7], A[11, 2] = -1, 1


G = np.zeros(A.shape[0])

G[0] = TC0.loc[0]['G']
G[1] = TC1.loc[0]['G']
G[2] = TC1.loc[1]['G']
G[3] = TC1.loc[2]['G']
G[4] = TC2.loc[0]['G']
G[5] = TC3.loc[0]['G']
G[6] = wS.loc[0]['h0'] * wS.loc[0]['Area']
G[7] = G[8] = w0.loc[
    'Insulation']['Conductivity'] / (w0.loc[
        'Insulation']['Width'] / 2) * wS.loc[0]['Area']
G[9] = G[10] = w0.loc[
    'Concrete']['Conductivity'] / (w0.loc[
        'Concrete']['Width'] / 2) * wS.loc[0]['Area']
G[11] = wS.loc[0]['h1'] * wS.loc[0]['Area']


# C = np.array([0., 239580., 0., 18216000., 0., 0., 32400., 1089000.])

C = np.zeros(A.shape[1])

C[0] = TC1.loc[TC1['A'] == 'C', 'θg']
C[2] = TC2.loc[TC2['A'] == 'C', 'θ0']
C[4] = w0.loc[
    'Insulation']['Density'] * w0.loc[
        'Insulation']['Specific heat'] * w0.loc[
            'Insulation']['Width'] * wS.loc[0]['Area']
C[6] = w0.loc[
    'Concrete']['Density'] * w0.loc[
        'Concrete']['Specific heat'] * w0.loc[
            'Concrete']['Width'] * wS.loc[0]['Area']

b = np.zeros(12)        # all branches
b[[1, 4, 5, 6]] = 1     # branches with temperature sources

f = np.zeros(8)         # all nodes
f[[0, 2, 3, 7]] = 1     # nodes with heat-flow sources

y = np.zeros(8)         # nodes
y[[2, 7]] = 1           # nodes: in wall surface, in air

[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, np.diag(G), np.diag(C), b, f, y)
print('As = \n', As, '\n')
print('Bs = \n', Bs, '\n')
print('Cs = \n', Cs, '\n')
print('Ds = \n', Ds, '\n')

# ------------------
# Dynamic simulation
# ------------------
# time step smaller than dtmax = min(−2/𝜆𝑖)
λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As
dtmax = 2 * min(-1. / λ)
dt = np.floor(dtmax / 60) * 60   # s
print(f'dt = {dt} s = {dt / 60:.0f} min')

# settling time
time_const = np.array([int(x) for x in sorted(-1 / λ)])
t_settle = 4 * max(-1 / λ)
print(f'Settling time: \
{t_settle:.0f} s = \
{t_settle / 60:.1f} min = \
{t_settle / (3600):.2f} h = \
{t_settle / (3600 * 24):.2f} days')

# # Step response
# # -------------
# # duration: next multiple of 3600 s that is larger than t_settle
# duration = np.ceil(t_settle / 3600) * 3600
# n = int(np.floor(duration / dt))        # number of time steps
# t = np.arange(0, n * dt, dt)            # time vector for n time steps

# # input vector
# u = np.zeros([8, n])                    # u = [To To Ti_sp To Φ Qa Qo Qi]
# u[[0, 1, 3], :] = 10 * np.ones([3, n])  # To = 10 for n time steps
# u[2, :] = 20 * np.ones([1, n])          # Ti_sp

# # initial conditions
# n_s = As.shape[0]                       # number of state variables
# θ_exp = np.zeros([n_s, t.shape[0]])     # explicit Euler in time t
# θ_imp = np.zeros([n_s, t.shape[0]])     # implicit Euler in time t

# # time integration
# I = np.eye(n_s)                         # identity matrix

# for k in range(n - 1):
#     θ_exp[:, k + 1] = (I + dt * As) @\
#         θ_exp[:, k] + dt * Bs @ u[:, k]
#     θ_imp[:, k + 1] = np.linalg.inv(I - dt * As) @\
#         (θ_imp[:, k] + dt * Bs @ u[:, k])

# # outputs
# y_exp = Cs @ θ_exp + Ds @  u
# y_imp = Cs @ θ_imp + Ds @  u

# # plot
# fig, ax = plt.subplots()
# ax.plot(t / 3600, y_exp.T, t / 3600, y_imp.T)
# ax.set(xlabel='Time, $t$ / h',
#        ylabel='Temperatue, $θ_i$ / °C',
#        title='Step input: outdoor temperature $T_o$')
# ax.legend(['θia - explicit', 'θwi - explicit',
#            'θia - implicit', 'θwi - implicit'])
# ax.grid()
# plt.show()

# ### Simulation with weather data

# In[40]:


start_date = '02-01 12:00:00'
end_date = '02-07 18:00:00'


# In[41]:


start_date = '2000-' + start_date
end_date = '2000-' + end_date
print(f'{start_date} \tstart date')
print(f'{end_date} \tend date')


# #### Inputs
# ##### Read weather data

# In[42]:


filename = './weather_data/FRA_Lyon.074810_IWEC.epw'
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data


# In[43]:


weather.index = weather.index.map(lambda t: t.replace(year=2000))
weather = weather.loc[start_date:end_date]


# In[44]:


surface_orientation = {'slope': 90,
                       'azimuth': 0,
                       'latitude': 45}
albedo = 0.2
rad_surf = dm4bem.sol_rad_tilt_surf(
    weather, surface_orientation, albedo)
# pd.DataFrame(rad_surf)


# The total solar [irradiance](https://en.m.wikipedia.org/wiki/Solar_irradiance)  $E_{tot}$, in W/m², is the sum of direct, diffuse, and reflected components.  

# In[45]:


rad_surf['Φtot'] = rad_surf.sum(axis=1)


# In[46]:


# resample weather data
data = pd.concat([weather['temp_air'], rad_surf['Φtot']], axis=1)
data = data.resample(str(dt) + 'S').interpolate(method='linear')
data = data.rename(columns={'temp_air': 'To'})
data = data.rename_axis('Time')
# pd.DataFrame(data)


# In[47]:


data['Ti'] = 20 * np.ones(data.shape[0])
data['Qa'] = 0 * np.ones(data.shape[0])
# pd.DataFrame(data)


# In[48]:


# input vector
To = data['To']
Ti = data['Ti']
Φo = wS.loc[0]['α0'] * wS.loc[0]['Area'] * data['Φtot']
Φi = τ_gSW * α_wSW * wall['Surface']['Glass'] * data['Φtot']
Qa = data['Qa']
Φa = α_gSW * wall['Surface']['Glass'] * data['Φtot']

u = pd.concat([To, To, To, Ti, Φo, Φi, Qa, Φa], axis=1)
u.columns.values[[4, 5, 7]] = ['Φo', 'Φi', 'Φa']
# pd.DataFrame(u)


# In[49]:


θ_exp = 20 * np.ones([As.shape[0], u.shape[0]])


# In[50]:


for k in range(u.shape[0] - 1):
    θ_exp[:, k + 1] = (I + dt * As) @ θ_exp[:, k]        + dt * Bs @ u.iloc[k, :]


# In[51]:


y_exp = Cs @ θ_exp + Ds @ u.to_numpy().T
q_HVAC = Kp * (data['Ti'] - y_exp[0, :])


# In[52]:


data['θi_exp'] = y_exp.T
data['q_HVAC'] = q_HVAC.T


# In[53]:


fig, axs = plt.subplots(2, 1)

data[['To', 'θi_exp']].plot(ax=axs[0],
                            xticks=[],
                            ylabel='Temperature, $θ$ / °C')
axs[0].legend(['$θ_{outdoor}$', '$θ_{indoor}$'],
              loc='upper right')

data[['Φtot', 'q_HVAC']].plot(ax=axs[1],
                              ylabel='Heat rate, $q$ / W')
axs[1].set(xlabel='Time')
axs[1].legend(['$Φ_{total}$', '$q_{HVAC}$'],
             loc='upper right')
plt.show()


# In[54]:


t = dt * np.arange(data.shape[0])   # time vector

fig, axs = plt.subplots(2, 1)
# plot outdoor and indoor temperature
axs[0].plot(t / 3600 / 24, data['To'], label='$θ_{outdoor}$')
axs[0].plot(t / 3600 / 24, y_exp[0, :], label='$θ_{indoor}$')
axs[0].set(ylabel='Temperatures, $θ$ / °C',
           title='Simulation for weather')
axs[0].legend(loc='upper right')

# plot total solar radiation and HVAC heat flow
axs[1].plot(t / 3600 / 24, data['Φtot'], label='$Φ_{total}$')
axs[1].plot(t / 3600 / 24, q_HVAC, label='$q_{HVAC}$')
axs[1].set(xlabel='Time, $t$ / day',
           ylabel='Heat flows, $q$ / W')
axs[1].legend(loc='upper right')

fig.tight_layout()


"""
pd_dm4bem.tc2ss: with assembling
"""
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
