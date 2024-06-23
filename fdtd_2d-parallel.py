import numpy as np
from matplotlib import pyplot as plt
import math as m
import scipy.constants as constants
import timeit
import numba
import os
from mpi4py import MPI

plt.rcParams.update({'font.size': 17}) # keep those graph fonts readable!
plt.rcParams['figure.dpi'] = 120 # plot resolution

##### USER CONTROLS ######
"Select speed or save_snapshot. If both are selected, speed will be default."
speed = 1
save_snapshots = 1

timeList = (100, 300, 700, 900) # time stamps for save
##########################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"Basic Geometry and Dielectric Parameters"
Xmax =  3008  # number of FDTD cells in x
Ymax =  3008  # number of FDTD cells in y
nsteps = 1000 # total number of FDTD time steps
cycle = 80 # for animation
time_pause = 0.01 # pause for animation
c = constants.c # speed of light in vacuum
ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step
fs = constants.femto # 1.e-15 - useful for pulses 
tera = constants.tera # 1.e12 - used for optical frequencues 

# split along y axis
num_rows = Ymax // size
start_y = rank * num_rows
end_y = start_y + num_rows

# dipole source position, at center
isource = int(Ymax/2)+1
jsource = int(Xmax/2)

# simple fixed dielectric box coordinates
X1=isource+10; X2=X1+40
Y1=jsource+10; Y2=Y1+40

"Pulse parameters and points per wavelength"
spread=2.* fs/dt # 2 fs for this example
t0=spread*6
freq_in = 2*np.pi*200*tera # incident (angular) frequency
w_scale = freq_in*dt
lam = 2*np.pi*c/freq_in # near 1.5 microns
eps2 = 4 # dielectric box (so 1 is just free space)
ppw = int(lam/ddx/eps2**0.5) #  rounded down

if rank == 0:
    print('points per wavelength:',ppw, '(should be > 15)')

"Polarization pulse"
@numba.jit(nopython=True)
def pol_pulse(t):
    return np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))

"Update Ez and Dz"
@numba.jit(nopython=True)
def update_Ez(Dz, Hx, Hy, Ez, ga, pulse):
    for x in range (1,(num_rows+2)-1): 
        for y in range (1,Xmax-1):
            Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1]) 
            Ez[x,y] =  ga[x,y]*(Dz[x,y])
    
    # check if the source is within the local domain
    if start_y <= isource < end_y:
        Dz[isource_local,jsource] =  Dz[isource_local,jsource] + pulse 
        Ez[isource_local,jsource] =  ga[isource_local,jsource]*(Dz[isource_local,jsource])
        
    return Ez, Dz

"Update Hz and Hy"
@numba.jit(nopython=True)
def update_H(Hx, Hy, Ez):
    for x in range (0,(num_rows+2)-1): 
        for y in range (0,Xmax-1): 
            Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])                       
            Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y])  
    return Hx, Hy

"Boundary communication between processes"
def boundary_communication(array):
    requests = []
    if rank > 0:
        # top boundary
        req = comm.Isend(array[1, :], dest=rank-1, tag=1)
        requests.append(req)
        req = comm.Irecv(array[0,:], source=rank-1, tag=2)
        requests.append(req)
    
    if rank < size - 1:
        # bottom boundary
        req = comm.Irecv(array[-1,:], source=rank+1, tag=1)
        requests.append(req)
        req = comm.Isend(array[-2, :], dest=rank+1, tag=2)
        requests.append(req)
        
    MPI.Request.Waitall(requests)
    

"Main FDTD loop iterated over nsteps"
def FDTD_loop(nsteps, Dz, Hx, Hy, Ez, ga, EzMonTime1, PulseMonTime):
    # loop over all time steps
    for t in range (0,nsteps):
        boundary_communication(Ez)
        boundary_communication(Dz)
        boundary_communication(Hx)
        boundary_communication(Hy)
        
        pulse = pol_pulse(t)

        # calculate Dz, Ez 
        Ez, Dz = update_Ez(Dz, Hx, Hy, Ez, ga, pulse)
        boundary_communication(Ez)
        
        # update H 
        Hx, Hy = update_H(Hx, Hy, Ez)
        
        final_Ez = np.empty((Ymax,Xmax), dtype=np.float64)  # full array to gather into for one time step
        comm.Gather(Ez[1:-1,:], final_Ez, root=0) # gather all processes
        
        EzMonTime1[t] = final_Ez[isource,jsource]
        PulseMonTime[t] = pulse
            
        # save plots
        if (t in timeList and save_snapshots == 1):            
            if (rank == 0):
                save(t, final_Ez, EzMonTime1, PulseMonTime)
                
"Main FDTD loop - FAST"
def FDTD_loop_speed(nsteps, Dz, Hx, Hy, Ez, ga):  
    # loop over all time steps
    for t in range (0,nsteps):
        boundary_communication(Ez)
        boundary_communication(Dz)
        boundary_communication(Hx)
        boundary_communication(Hy)
        
        pulse = pol_pulse(t)

        # calculate Dz, Ez 
        Ez, Dz = update_Ez(Dz, Hx, Hy, Ez, ga, pulse)
        boundary_communication(Ez)
        
        # update H 
        Hx, Hy = update_H(Hx, Hy, Ez)

def save(t, Ez, EzMonTime1, PulseMonTime):
    fig = plt.figure(figsize=(8,6))
    
    # main graph is E(z,y, time snapshops), and a small graph of E(t) as center
    ax = fig.add_axes([.2, .15, .7, .7])   
    ax2 = fig.add_axes([.035, .79, .15, .15])    

    # 2d plot - several options, two examples below
    img = ax.contourf(Ez)
    cbar=plt.colorbar(img, ax=ax)
    cbar.set_label('$Ez$ (arb. units)')

    # add labels to axes
    ax.set_xlabel('Grid Cells ($x$)')
    ax.set_ylabel('Grid Cells ($y$)')
     
    # dielectric box - comment if not using of course (if eps2=1)
    ax.vlines(X1,Y1,Y2,colors='r')
    ax.vlines(X2,Y1,Y2,colors='r')
    ax.hlines(Y1,X1,X2,colors='r')
    ax.hlines(Y2,X1,X2,colors='r')

    # add title with current simulation time step
    ax.set_title("frame time {}".format(t))
    
    ax.set_xlim(500, 2500)
    ax.set_ylim(500, 2500)

    # small graph to see time development as a single point
    PulseNorm = np.asarray(PulseMonTime)*0.2;
    ax2.plot(PulseNorm,'r',linewidth=1.6)
    ax2.plot(EzMonTime1,'b',linewidth=1.6)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_title('$E_{src}(t)$, pulse')
    plt.savefig("./iter_{}.pdf".format(t), dpi=300)
    
    fig = plt.figure(figsize=(8,6))

    
## PARALLELIZATION 
"2d Arrays"
full_Ez = np.zeros((Ymax+2*size,Xmax), dtype=np.float64)
full_Dz = np.zeros((Ymax+2*size,Xmax), dtype=np.float64)
full_Hx = np.zeros((Ymax+2*size,Xmax), dtype=np.float64)
full_Hy = np.zeros((Ymax+2*size,Xmax), dtype=np.float64)
full_ga=np.ones([Xmax,Ymax],float) # for spatially varying dielectric constant

"Time Dependent Field Monitors"
EzMonTime1 = np.zeros((nsteps),float)
PulseMonTime = np.zeros((nsteps),float)

"Local Arrays for Each Process"
local_Ez = np.empty((num_rows+2,Xmax), dtype=np.float64) # array for Ez in each process
local_Dz = np.empty((num_rows+2,Xmax), dtype=np.float64) # array for Dz in each process
local_Hx = np.empty((num_rows+2,Xmax), dtype=np.float64) # array for Hx in each process
local_Hy = np.empty((num_rows+2,Xmax), dtype=np.float64) # array for Hy in each process
local_ga = np.ones([num_rows+2,Xmax], dtype=np.float64) # array for ga in each process

comm.Scatter(full_Ez, local_Ez, root=0)
comm.Scatter(full_Dz, local_Dz, root=0)
comm.Scatter(full_Hx, local_Hx, root=0)
comm.Scatter(full_Hy, local_Hy, root=0)
comm.Scatter(full_ga, local_ga, root=0)


# local dielectric box coordinates
local_Y1 = max(Y1, start_y) - start_y  # adjust Y1 for local domain
local_Y2 = min(Y2, end_y) - start_y    # adjust Y2 for local domain

"Dielectric box"
for local_x in range(local_Y1, local_Y2+1):
    for y in range(X1, X2+1):
        if 0 <= local_x < num_rows:  # check if within local domain bounds
            local_ga[local_x, y] = 1./eps2  # update local `ga` for the dielectric box
            
# local source coordinates
if start_y <= isource < end_y:
    # the source is within this process's local domain
    isource_local = isource - start_y
    # jsource remains the same since the domain division is along the y-axis
    jsource_local = jsource
else:
    isource_local = isource
    jsource_local = jsource


"Main FDTD loop iterated over nsteps"
if (speed==1):
    if rank == 0:
        start = timeit.default_timer()
    FDTD_loop_speed(nsteps, local_Dz, local_Hx, local_Hy, local_Ez, local_ga)
    if rank == 0:
        stop = timeit.default_timer()
        print ("Time for FDTD simulation", stop - start)
else:
    FDTD_loop(nsteps, local_Dz, local_Hx, local_Hy, local_Ez, local_ga, EzMonTime1, PulseMonTime)