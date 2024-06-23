# -*- coding: utf-8 -*-
"""
Late edited: March 2024

Author: Jane Cohen

Based on code written by S. Hughes

Simple 2d FDTD Example Simulation for TM: Ez,Dz,Hy,Hx,

Look for "User controls" to set parameters before running each cell

Set your graphics to Auto, e.g., within Spyder or %matplotlib auto
"""

#%% Imports, parameters and functions

"User Controls"
# specify path to save plots to"
cd = None # use "None" to use current directory
cycle = 80 # for animation
time_pause = 0.01 # pause for animation


import numpy as np
from matplotlib import pyplot as plt
import math as m
import scipy.constants as constants
import timeit
import numba
import os
plt.rcParams.update({'font.size': 17}) 
plt.rcParams['figure.dpi'] = 120 


"Basic Geometry and Dielectric Parameters"
Xmax = 500  # number of FDTD cells in x
Ymax = 500  # number of FDTD cells in y
nsteps = 1000 # total number of FDTD time steps
c = constants.c # speed of light in vacuum
ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step
fs = constants.femto # 1.e-15 - useful for pulses 
tera = constants.tera # 1.e12 - used for optical frequencues 

# dipole source position, at center
isource = int(Ymax/2)
jsource = int(Xmax/2)

# simple fixed dielectric box coordinates
X1=isource+10; X2=X1+40
Y1=jsource+10; Y2=Y1+40

"Pulse parameters and points per wavelength"
spread=2.* fs/dt # 2 fs for this example
t0=spread*6
freq_in = 2*m.pi*200*tera # incident (angular) frequency
w_scale = freq_in*dt
lam = 2*m.pi*c/freq_in # near 1.5 microns
eps2 = 4 # dielectric box (so 1 is just free space)
ppw = int(lam/ddx/eps2**0.5) #  rounded down
print('points per wavelength:',ppw, '(should be > 15)')


"Clean all arrays and variables"
def initialize_arrays():
    "2d Arrays"
    Ez = np.zeros([Xmax,Ymax],float)
    Hx = np.zeros([Xmax,Ymax],float)
    Hy = np.zeros([Xmax,Ymax],float) 
    Dz = np.zeros([Xmax,Ymax],float)
    ga=np.ones([Xmax,Ymax],float) # for spatially varying dielectric constant
    
    "Time Dependent Field Monitors"
    EzMonTime1 = np.zeros((nsteps),float)
    PulseMonTime = np.zeros((nsteps),float)

    "3d Array"
    Ez_full = np.zeros([nsteps,Xmax,Ymax],float) 
    
    "Dielectric box"
    for j in range (0,Ymax): 
        for i in range (0,Xmax):
            if i>X1 and i<X2+1 and j>Y1 and j<X2+1:   
                ga[i,j] = 1./eps2

    
    return Ez, Hx, Hy, Dz, EzMonTime1, PulseMonTime, Ez_full, ga

"Polarization pulse"
@numba.jit(nopython=True)
def pol_pulse(t):
    return np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))

"Update Ez and Dz"
@numba.jit(nopython=True)
def update_Ez(Dz, Hx, Hy, Ez, ga, pulse):
    for x in range (1,Xmax-1): 
        for y in range (1,Ymax-1):
            Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1]) 
            Ez[x,y] =  ga[x,y]*(Dz[x,y])
    
    Dz[isource,jsource] =  Dz[isource,jsource] + pulse 
    Ez[isource,jsource] =  ga[isource,jsource]*(Dz[isource,jsource])
    return Ez, Dz

"Update Hz and Hy"
@numba.jit(nopython=True)
def update_H(Hx, Hy, Ez):
    for x in range (0,Ymax-1): 
        for y in range (0,Xmax-1): 
            Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])                       
            Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y])  
    return Hx, Hy
    
"Main FDTD loop iterated over nsteps"
def FDTD_loop(nsteps, Dz, Hx, Hy, Ez, ga):
    # loop over all time steps
    for t in range (0,nsteps):
        pulse = pol_pulse(t)

        # calculate Dz, Ez         
        Ez, Dz = update_Ez(Dz, Hx, Hy, Ez, ga, pulse)

        # save one point in time just to see the transient
        EzMonTime1[t] = Ez[isource,jsource]
        PulseMonTime[t] = pulse

        # update H 
        Hx, Hy = update_H(Hx, Hy, Ez)
        
        # save time step
        Ez_full[t] = Ez
        
        # animation
        if (t % cycle == 0 and animate == 1): 
            animation(t)
            
    return Ez_full

"Main FDTD loop - FAST"
def FDTD_loop_speed(nsteps, Dz, Hx, Hy, Ez, ga):
    # loop over all time steps
    for t in range (0,nsteps):
        pulse = pol_pulse(t)

        # calculate Dz, Ez         
        Ez, Dz = update_Ez(Dz, Hx, Hy, Ez, ga, pulse)

        # update H 
        Hx, Hy = update_H(Hx, Hy, Ez)
            
"Animation function"     
def animation(t):
    plt.clf() # close each time for new update graph/colormap
    ax = fig.add_axes([.2, .15, .7, .7])   
    ax2 = fig.add_axes([.035, .79, .15, .15])   
  

    # 2d plot
    img = ax.contourf(Ez)
    cbar=plt.colorbar(img, ax=ax)
    cbar.set_label('$Ez$ (arb. units)')

    # add labels to axes
    ax.set_xlabel('Grid Cells ($x$)')
    ax.set_ylabel('Grid Cells ($y$)')
     
    # dielectric box 
    ax.vlines(X1,Y1,Y2,colors='r')
    ax.vlines(X2,Y1,Y2,colors='r')
    ax.hlines(Y1,X1,X2,colors='r')
    ax.hlines(Y2,X1,X2,colors='r')

    # add title with current simulation time step
    ax.set_title("frame time {}".format(t))
    plt.show()

    # small graph to see time development as a single point
    PulseNorm = np.asarray(PulseMonTime)*0.2;
    ax2.plot(PulseNorm,'r',linewidth=1.6)
    ax2.plot(EzMonTime1,'b',linewidth=1.6)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_title('$E_{src}(t)$, pulse')
    plt.show()
    plt.pause(time_pause) # pause value to watch what is happening


"Plot and save time stamps"    
def plot_and_save(t, cd):
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['figure.dpi']= 120
    fig = plt.figure(figsize=(8,6))
    
    ax = fig.add_axes([.2, .15, .7, .7])   
    ax2 = fig.add_axes([.05, .79, .15, .15])   

    # 2d plot 
    t = int(t//1) # make integer index
    img = ax.contourf(Ez_full[t])
    cbar=plt.colorbar(img, ax=ax)
    cbar.set_label('$Ez$ (arb. units)')

    # add labels to axes
    ax.set_xlabel('Grid Cells ($x$)')
    ax.set_ylabel('Grid Cells ($y$)')
     
    # dielectric box
    ax.vlines(X1,Y1,Y2,colors='r')
    ax.vlines(X2,Y1,Y2,colors='r')
    ax.hlines(Y1,X1,X2,colors='r')
    ax.hlines(Y2,X1,X2,colors='r')

    # add title with current simulation time step
    ax.set_title("frame time {}".format(t))

    # small graph to see time development as a single point
    PulseNorm = np.asarray(PulseMonTime[:t])*0.2;
    ax2.plot(PulseNorm,'r',linewidth=1.6)
    ax2.plot(EzMonTime1[:t],'b',linewidth=1.6)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_title('$E_{src}(t)$, pulse')
    
    # save plot
    if (cd == None):
        cd = os.getcwd()
    plt.savefig(f"{cd}/snapshot_{t}.pdf", dpi=300)


#%% Run method and display results

"User controls"
speed_time = 0 # run fastest version of solver and display time
animate = 0 # show animation
save_snapshots = 1 # save and display snapshots of plots to specified path

# timer
if (speed_time == 1): 
    Ez, Hx, Hy, Dz, EzMonTime1, PulseMonTime, Ez_full, ga = initialize_arrays()
    start = timeit.default_timer()      
    FDTD_loop_speed(nsteps,Dz, Hx, Hy, Ez, ga)
    stop = timeit.default_timer()
    print ("Time for FDTD simulation", stop - start)

if animate==1: 
    fig = plt.figure(figsize=(8,6))
    
Ez, Hx, Hy, Dz, EzMonTime1, PulseMonTime, Ez_full, ga = initialize_arrays()
FDTD_loop(nsteps,Dz, Hx, Hy, Ez, ga)

timeList = np.linspace(100, 950, 4) # time stamps to save
if save_snapshots==1:
        for t in timeList:
            plot_and_save(t, cd)









