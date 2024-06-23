# -*- coding: utf-8 -*-
"""
Author: Jane Cohen
Date: March 20th, 20024

1d FDTD method to simulate the propagation of a electromagnetic field across a dielectric.
Look for "User controls" to set parameters before running each cell
"""

#%% Imports, parameters and functions

"User controls"
# specify path to save plots to
cd = None # use "None" to use current directory
cycle = 50 # for animation updates 
time_pause  = 0.5 # for animation



import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.constants as constants
import os
plt.rcParams.update({'font.size': 20})

"Basic geometry and dielectric parameters"
Xmax = 801  # number of FDTD cells in x
nsteps = 1000 # number of FDTD time steps
c = constants.c # speed of light in vacuum
ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step
fs = constants.femto # 1.e-15 - useful for pulses 
tera = constants.tera # 1.e12 - used for optical frequencues 
dielectric_cnst = 9 # dielectric constant 


"Pulse parameters and points per wavelength"
isource = 200 # source position (launch point)
spread = 2.* fs/dt # 2 fs for this example
X1 = int(Xmax/2) # center position
t0 = spread*6
freq_in = 2*math.pi*200*tera # incident (angular) frequency
w_scale = freq_in*dt
lam = 2*math.pi*c/freq_in # near 1.5 microns
ppw = int(lam/ddx) # will round down
print('points per wavelength',ppw, 'should be > 15') # checking stability condition

# an array for spatial points (without first and last points)
xs = np.arange(1,Xmax-1)  

t = 0# initial time
t_list = np.arange(0,nsteps)
     

"Time-dependent pulse"           
def pulse(t_source):
    pulse = -np.exp(-0.5 * (t_source - t0)**2 / spread**2) * np.cos(t_source * w_scale)
    return pulse

"Apply periodic boundary conditions"
def boundary_conditions(Ex, Ex_minus_2):
    Ex[0] = Ex_minus_2[1]
    Ex[Xmax-1] = Ex_minus_2[Xmax-2]
    return Ex

"Initialize arrays for simulation"
def initialize_arrays():
    Ex = np.zeros((Xmax),float) # E array  
    Hy = np.zeros((Xmax),float) # H array 
    full_Ex = np.zeros((nsteps, Xmax),float) # Ex array for all time steps
    return Ex, Hy, full_Ex

"FDTD loop iterated nsteps times with ABC"
def ABC_loop(nsteps):
    Ex, Hy, full_Ex_ABC =  initialize_arrays()
    full_Ex_ABC[0,:] = Ex
    
    # loop over all time steps
    for i in range(0, nsteps):
        t = i-1  # iterative time-dependent pulse as source

        # update Ex 
        Ex[1:Xmax-1] += 0.5 * (Hy[:Xmax-2] - Hy[1:Xmax-1])
        Ex[isource] -= pulse(t) * 0.5
        
        # apply boundary conditions
        Ex_update = boundary_conditions(Ex, full_Ex_ABC[i-2, :])

        # update H 
        Hy[:Xmax-1] += 0.5 * (Ex_update[:Xmax-1] - Ex_update[1:Xmax])
        
        # store time step
        full_Ex_ABC[i] = Ex_update
        
        # check and show animation
        if show_animation == 1:
            if (i % cycle == 0):
                animate_simple(Ex_update, i, cd, 1, 1)
                
    return full_Ex_ABC
        
        
"FDTD loop iterated nsteps times with TFSF"
def TFSF_loop(nsteps):
    Ex, Hy, full_Ex_TFSF =  initialize_arrays() 
    full_Ex_TFSF[0,:] = Ex
    
    
    # loop over all time steps
    for i in range(0, nsteps):
        t = i-1  # iterative time-dependent pulse as source
        
        Ex_source = pulse(t)
        Hy_source = pulse(t+1/2)

        # update Ex         
        Ex[1:Xmax-1] += 0.5 * (Hy[:Xmax-2] - Hy[1:Xmax-1])
        Ex[isource] = Ex[isource] - 0.5 * Hy_source 
        
        # apply boundary conditions
        Ex_update = boundary_conditions(Ex, full_Ex_TFSF[i-2, :])

        # update H 
        Hy[:Xmax-1] += 0.5 * (Ex_update[:Xmax-1] - Ex_update[1:Xmax])
        Hy[isource-1] = Hy[isource-1] - 0.5 * Ex_source
        
        # save time step
        full_Ex_TFSF[i] = Ex_update
        
        # check and show animation
        if show_animation == 1:
            if (i % cycle == 0):
                animate_simple(Ex_update, i, cd, 1, 1)
    
    return full_Ex_TFSF
        
        
        
"FDTD loop iterated nsteps times with dielectric film"        
def film_loop(nsteps, x_film_start, x_film_end):
    Ex, Hy, full_Ex_film =  initialize_arrays() 

    # initialize arrays for incident, reflected and transmitted fields
    time = np.linspace(0,nsteps, nsteps)
    Ex_in = pulse(time)
    Ex_r = np.empty((nsteps))
    Ex_t = np.empty((nsteps))
    
    # set up dielectric film
    epsilon = np.empty((Xmax), float)
    epsilon.fill(1)
    epsilon[film_x_start:film_x_end] = dielectric_cnst
    
    # loop over all time steps
    for i in range(0, nsteps):
        t = i-1  # iterative time-dependent pulse as source
        
        Ex_source = pulse(t)
        Hy_source = pulse(t+1/2)

        # update Ex         
        Ex[1:Xmax-1] += 0.5 * (1/epsilon[1:Xmax-1]) * (Hy[:Xmax-2] - Hy[1:Xmax-1])
        Ex[isource] = Ex[isource] - 0.5 * Hy_source 
        
        # apply boundary conditions
        Ex_update = boundary_conditions(Ex, full_Ex_film[i-2, :])

        # update H 
        Hy[:Xmax-1] += 0.5 * (Ex_update[:Xmax-1] - Ex_update[1:Xmax])
        Hy[isource-1] = Hy[isource-1] - 0.5 * Ex_source
        
        # store time step
        full_Ex_film[i] = Ex_update
        
        # store reflected and transmitted wave
        Ex_r[i] = full_Ex_film[i,r_x]
        Ex_t[i] = full_Ex_film[i,t_x]
        
        # check and show animation
        if show_animation == 1:
            if (i % cycle == 0):
                animate_film(Ex_update, i, cd, 1, 1)
        
    return Ex_in, Ex_r, Ex_t, time

"Compute Fourier transform of three fields"
def compute_fourier(Ex_in, Ex_r, Ex_t, time):
    
    w = np.fft.fftfreq(len(time), dt)  # fft of dt
    w = np.fft.fftshift(w) # shift frequency axis
    w = w/10**12 # to THz
    
    # incident field
    Ex_in_w = np.fft.fft(Ex_in) # fft 
    Ex_in_w = np.fft.fftshift(Ex_in_w) # shift frequency axis

    # reflected field
    Ex = Ex_r[:]
    Ex_r_w = np.fft.fft(Ex) # fft 
    Ex_r_w = np.fft.fftshift(Ex_r_w) # shift frequency axis

    # transmitted field
    Ex = Ex_t[:]
    Ex_t_w = np.fft.fft(Ex) # fft 
    Ex_t_w = np.fft.fftshift(Ex_t_w) # shift frequency axis
    
    return Ex_in_w, Ex_r_w, Ex_t_w, w

"Compute numerical reflection and transmission coefficients"
def compute_coefficients(Ex_in_w, Ex_r_w, Ex_t_w):
    T = np.zeros((nsteps))
    R = np.zeros((nsteps))

    for i in range(nsteps):
        T[i] = np.abs(Ex_t_w[i])**2 / np.abs(Ex_in_w[i])**2
        R[i] = np.abs(Ex_r_w[i])**2 / np.abs(Ex_in_w[i])**2
        
    return T, R

"Compute analytical reflection and transmission coefficients"
def analytic_coefficients(max_w):
    w_an = np.linspace(100e12, max_w*1e12, 1000) # frequency range
    w_an = np.pi*2 * w_an # convert from angular frequency to frequency bc FT is in frequency
    n = np.sqrt(dielectric_cnst)
    L = 1e-6 # width of dielectric
    k0 = w_an/c
    r1 = (1-n)/(1+n)
    r2 = (n-1)/(n+1)
    
    r = ( r1 + r2*np.exp(2*1j*k0*L*n) ) / (1 + r1*r2*np.exp(2*1j*k0*L*n))
    t = (1+r1)*(1+r2)*np.exp(1j*k0*L*n) / (1+r1*r2*np.exp(2*1j*k0*L*n))
    
    T_an = (np.abs(t))**2 # analytical T
    R_an = (np.abs(r))**2 # analytical R
    
    w_an_THz = w_an / (1e12 * np.pi*2) # convert back to angular frequency
    
    return w_an_THz, T_an, R_an

"Plot and save snapshots all on one plot"
def save_timesnaps_one_plot(E_array, snapshots, cd, name, source_x, film_flag):
    
    # set up plot
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['figure.dpi']= 120
    plt.figure(figsize=(10,8))
    
    # plot snapshots
    for t in snapshots:
        plt.plot(xs, E_array[t, 1:Xmax-1], linewidth=2, label=f"Time step = {t}")
        
    # configure plot
    plt.xlabel('z')
    plt.ylabel('$E_x$')
    plt.grid(True)
    plt.xlim(xlim_lower, xlim_upper)
    plt.ylim(ylim_lower, ylim_upper)
    plt.axvline(source_x, linestyle='--', color='red', label='z$_{source}$') # source 
    plt.tight_layout()
    
    # plot film
    if (film_flag == 1):
        plt.axvline(film_x_start)
        plt.axvline(film_x_end)
     
    # place legend above the current axis
    plt.subplots_adjust(top=0.7)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=False, shadow=False, fontsize=20)
    
    # save plot
    if (cd == None):
        cd = os.getcwd()
    plt.savefig(f"{cd}/{name}.pdf", dpi=300)
        
"Plot and save snapshots on individual plots"
def save_timesnaps(E_array, snapshots, cd, source_x, film_flag):
    for t in snapshots:
        
        # set up plot
        plt.rcParams.update({'font.size': 26})
        plt.rcParams['figure.dpi']= 120
        plt.figure(figsize=(9, 6))
        
        plt.plot(xs, E_array[t, 1:Xmax-1], color='cornflowerblue', linewidth=2)
        plt.xlabel('z')
        plt.ylabel('$E_x$')
        plt.title(f"Time step: {t}")
        plt.grid(True)
        plt.xlim(xlim_lower, xlim_upper)
        plt.ylim(ylim_lower, ylim_upper)
        plt.axvline(source_x, linestyle='--', color='red') # source
        
        # plot film
        if (film_flag == 1):
            plt.axvline(film_x_start)
            plt.axvline(film_x_end)
            
        # save plot
        if (cd == None):
            cd = os.getcwd()
        plt.savefig(f"{cd}/snapshot_{t}.pdf", dpi=300)
        
"Plot zoomed in portion of a plot"
def plot_zoomed(E_array, cd, t, name):
    
    # set up plot
    plt.rcParams.update({'font.size': 26})
    plt.rcParams['figure.dpi']= 120
    plt.figure(figsize=(9, 6))
    
    plt.plot(xs, E_array[t, 1:Xmax-1], color='cornflowerblue', linewidth=2)
    plt.xlabel('z')
    plt.ylabel('$E_x$')
    plt.title(f"Time step: {t}")
    plt.grid(True)
    plt.xlim(xlim_lower, xlim_upper) # strict axis limits
    plt.ylim(ylim_lower, ylim_upper) # strict axis limits
    plt.tight_layout()
        
    # save plot
    if (cd == None):
        cd = os.getcwd()
    plt.savefig(f"{cd}/{name}_{t}.pdf", dpi=300)
    
"Simple animation for ABCs"
def animate_simple(Ex_update, t, cd, ylim_flag, xlim_flag):
    plt.clf() # close each time for new update graph
    plt.plot(Ex_update[1:Xmax-1])
    plt.title("frame time {}".format(t))
    if (ylim_flag == 1):
        plt.ylim(ylim_lower, ylim_upper) # set y axis limits
    if (xlim_flag == 1):
        plt.xlim(xlim_lower, xlim_upper) # set x axis limits
    plt.axvline(isource-1, color='red', linestyle='dashed')
    plt.show()
    plt.pause(time_pause) # pause to watch simple animation 

"Animation for dielectric film"
def animate_film(Ex_update, t, cd, ylim_flag, xlim_flag):
    plt.clf() # close each time for new update graph
    plt.plot(Ex_update[1:Xmax-1])
    plt.title("frame time {}".format(t))
    if (ylim_flag == 1):
        plt.ylim(ylim_lower, ylim_upper) # set y axis limits
    if (xlim_flag == 1):
        plt.xlim(xlim_lower, xlim_upper) # set x axis limits
        
    # plot film
    plt.axvline(film_x_start, color='blue')
    plt.axvline(film_x_end, color='blue')
    plt.axvline(isource-1, color='red', linestyle='dashed') # source
    plt.xlabel("Grid cells (z)")
    plt.ylabel("$E_x$")
    plt.tight_layout()
    plt.show()
    plt.pause(time_pause) # pause to watch animation 
    
"Plot and save incident, refelcted and transmitted fields"
def plot_fields(Ex_in, Ex_r, Ex_t, time, cd):
    # set up plot
    plt.rcParams.update({'font.size': 26})
    plt.rcParams['figure.dpi']= 120
    plt.figure(figsize=(9, 6))

    plt.plot(time, Ex_in, linewidth=2, label='$E_{in}$') # incident field
    plt.plot(time, Ex_r, linewidth=2, label='$E_r$') # reflected field
    plt.plot(time, Ex_t, linewidth=2, label='$E_t$') # transmitted field
    plt.xlim(0, xlim_upper)
    plt.xlabel('time step')
    plt.ylabel('$E_x$')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # save plot
    if (cd == None):
        cd = os.getcwd()
    plt.savefig(f"{cd}/fields.pdf", dpi=300) # save plot plt.savefig(f"{cd}/{name}_{t}.pdf", dpi=300)
    
"Plot numerical transmission and reflection coefficients"
def plot_coefficients(T, R, w, cd):
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['figure.dpi']= 120
    plt.figure(figsize=(8, 5))
    plt.plot(w, T, label='$T$') # plot T
    plt.plot(w, R, label='$R$') # plot R
    plt.plot(w, R+T, label='$Sum$') # plot sum
    plt.xlim(xlim_lower, xlim_upper)
    plt.ylim(ylim_lower, ylim_upper)
    plt.xlabel('$\omega$ $\pi/2$ (THz)')
    plt.ylabel('$R, T$')
    plt.grid()
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # save plot
    if (cd == None):
        cd = os.getcwd()
    plt.savefig(f"{cd}/coefficients.pdf", dpi=300) # save plot
    
    
"Plot comparison of analytical and numerical coefficients"
def plot_coefficients_comparison(T, R, w, T_an, R_an, w_an_THz, cd):
    plt.rcParams.update({'font.size': 24})
    plt.rcParams['figure.dpi']= 120
    plt.figure(figsize=(9, 6))
    
    # plot numerical coefficients
    plt.plot(w, T, label='$T$')
    plt.plot(w, R, label='$R$')
    
    # plot analytical coefficients
    plt.plot(w_an_THz, T_an, label='$T_{an}$', linestyle='dashed')
    plt.plot(w_an_THz, R_an, label='$R_{an}$', linestyle='dashed')
    
    # plot sum
    plt.plot(w, T+R, color='k', label='T+R')
    
    plt.xlim(xlim_lower, xlim_upper)
    plt.ylim(ylim_lower, ylim_upper)
    plt.xlabel('$\omega$ $\pi/2$ (THz)')
    plt.ylabel('$R, T$')
    plt.grid()
    
    # place legend above the current axis
    plt.subplots_adjust(top=0.8)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=5, fancybox=False, shadow=False, fontsize=22)
    
    plt.tight_layout()
    
    # save plot
    if (cd == None):
        cd = os.getcwd()
    plt.savefig(f"{cd}/coefficients_combined.pdf", dpi=300) # save plot

                
    
#%% 1 (a) Simple absorbing boundary conditions   

"USER CONTROLS"
save_boundary_condition_snapshots = 1 # save plot of ABC snapshotS
save_numerical_reflection = 1 # save plot of reflected field
show_animation = 0 



if show_animation == 1:
    fig = plt.figure(figsize=(8,6))
    ylim_lower = -0.7; ylim_upper = 0.7; xlim_lower = 0; xlim_upper = Xmax
    
full_Ex_ABC = ABC_loop(nsteps)

if (save_boundary_condition_snapshots == 1):
    snapshots = [700,950]
    ylim_lower = -0.7; ylim_upper = 0.7; xlim_lower = 0; xlim_upper = Xmax
    save_timesnaps_one_plot(full_Ex_ABC, snapshots, cd, 'ABCs', isource, 0)
    
if (save_numerical_reflection == 1):
    ylim_lower = -0.0005; ylim_upper = 0.0005; xlim_lower = 1; xlim_upper = 30
    plot_zoomed(full_Ex_ABC, cd, 950, 'ABC_reflection')
           

#%% 1 (b) TFSF

"USER CONTROLS"
save_snapshots = 1 # save snapshots of source injection
save_backscatter = 1 # save plot of backscatter
show_animation = 0



if (show_animation == 1):
    fig = plt.figure(figsize=(8,6))
    ylim_lower = -1; ylim_upper = 0.7; xlim_lower = 0; xlim_upper = Xmax

full_Ex_TFSF = TFSF_loop(nsteps)

if (save_snapshots == 1):
    snapshots = [300, 700]
    ylim_lower = -1; ylim_upper = 0.7; xlim_lower = 0; xlim_upper = Xmax
    save_timesnaps_one_plot(full_Ex_TFSF, snapshots, cd, 'TFSFs', isource, 0)
    
if (save_backscatter == 1):
    ylim_lower = -0.00001; ylim_upper = 0.00001; xlim_lower = isource-100; xlim_upper = isource
    plot_zoomed(full_Ex_TFSF, cd, 500, 'TSFS_backscatter')

    
#%% 1 (c) dielectric film

"USER CONTROLS"
save_time_plot = 1 # save plot of incident, reflected and transmitted field
show_animation = 0

# measuring points for each field
in_x = 300 # incident field
r_x = 150 # reflected field
t_x = 500 # transmitted field

# location and width of film
film_width = 1/1e6/ddx # for one micron
film_x_start = 400
film_x_end = int(film_x_start+film_width)

nsteps = 1500
if (show_animation == 1):
    fig = plt.figure(figsize=(8,6))
    ylim_lower = -1.5; ylim_upper = 1; xlim_lower = 0; xlim_upper = Xmax

Ex_in, Ex_r, Ex_t, time = film_loop(nsteps, film_x_start, film_x_end)

if (save_time_plot == 1):
    xlim_upper = nsteps
    plot_fields(Ex_in, Ex_r, Ex_t, time, cd)


#%% 1 (d) Fourier transforms
"MUST run cell above first"

"USER CONTROLS"
save_coefficient_plot = 1 # save plot of numerical coefficients
save_coefficient_comparison = 1 # save plot of analytical and numerical coefficients



show_animation = 0 # do not change -  no animation available for this section
nsteps = 100000
Ex_in, Ex_r, Ex_t, time = film_loop(nsteps, film_x_start, film_x_end)
Ex_in_w, Ex_r_w, Ex_t_w, w = compute_fourier(Ex_in, Ex_r, Ex_t, time)

T, R = compute_coefficients(Ex_in_w, Ex_r_w, Ex_t_w)

max_w = 350
w_an_THz, T_an, R_an = analytic_coefficients(max_w)

if (save_coefficient_plot ==  1):
    ylim_lower = 0; ylim_upper = 1.1; xlim_lower =100; xlim_upper = 300
    plot_coefficients(T, R, w, cd)
    
if (save_coefficient_comparison ==  1):
    ylim_lower = 0; ylim_upper = 1.1; xlim_lower = 125; xlim_upper = 325
    plot_coefficients_comparison(T, R, w, T_an, R_an, w_an_THz, cd)





