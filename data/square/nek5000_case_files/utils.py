import numpy as np
import subprocess
import numba as nb
import os
import pymech.neksuite as nek

def write_hist_points(n_hist):

    with open("square.his", "w") as file:

        file.write(str(n_hist) + '\n')

        # Top
        for j in range(int((n_hist-9)/4)):
            dx = j / int((n_hist-9)/4)            
            file.write(str(-0.5+dx) + ' ' + str(0.5) + ' 0.0 \n')

        # Back
        for j in range(int((n_hist-9)/4)):
            dx = j / int((n_hist-9)/4)            
            file.write(str(0.5) + ' ' + str(0.5-dx) + ' 0.0 \n')

        # Bottom
        for j in range(int((n_hist-9)/4)):
            dx = j / int((n_hist-9)/4)            
            file.write(str(0.5-dx) + ' ' + str(-0.5) + ' 0.0 \n')

        # Front
        for j in range(int((n_hist-9)/4)):
            dx = j / int((n_hist-9)/4)            
            file.write(str(-0.5) + ' ' + str(-0.5+dx) + ' 0.0 \n')

        for j in range(3):
            
            xj = 0.5+3*(j+1)
            yj = j+1.0

            file.write(str(xj) + ' 0.0  0.0 \n')
            file.write(str(xj) + ' ' + str(yj) + ' 0.0 \n')
            file.write(str(xj) + ' ' + str(-yj) + ' 0.0 \n')

@nb.jit
def read_hist_vals(hist, m, n_hist, col):

    vals = np.zeros((m, n_hist))

    for i in range(m):
        for j in range(n_hist):
            vals[i,j] = np.round(hist[i*n_hist+j, col],5)

    return vals

def parse_history_points(n_hist):
    """
    Takes history file written by nek and converts to individual files;
    pres_hist.npy --- (m * n_hist) matrix
    time_hist.npy --- (m * 1)
    hist_points.npy --- (n_hist * 2)
    Currently loads everything into memory at one which takes a while.
    """

    full_hist = np.loadtxt("square.his", skiprows=n_hist+1)
    m = int(full_hist.shape[0]/n_hist)

    u_hist = read_hist_vals(full_hist, m, n_hist, 1)
    v_hist = read_hist_vals(full_hist, m, n_hist, 2)
    p_hist = read_hist_vals(full_hist, m, n_hist, 3)
    if full_hist.shape[1] == 5: w_hist = read_hist_vals(full_hist, m, n_hist, 4)
    time_hist = np.array([full_hist[j*n_hist, 0] for j in range(m)])

    np.save('time_hist', time_hist)
    np.save('u_hist', u_hist)
    np.save('v_hist', v_hist)
    np.save('pres_hist', p_hist)
    if full_hist.shape[1] == 5: np.save('vort_hist', w_hist)

def write_par_files(T,dt,dT,Re,hist_freq):
    """
    Write parameter file for nek5000 simulation
    """

    # Make sure write interval coincides with hist_freq
    write_interval = int((dT/dt)/hist_freq)*hist_freq

    par_lines = ['#',\
                 '# nek parameter file',\
                 '#',\
                 '',\
                 '[GENERAL]',\
                 'stopAt = endTime',\
                 'endTime = '+str(T),\
                 'dt = '+str(dt),\
                 'variableDt = no',\
                 '',\
                 'targetCFL = 3.0',\
                 'timeStepper = BDF2',\
                 'extrapolation = OIFS',\
                 'writeControl = timeStep',\
                 'writeInterval = '+str(write_interval),\
                 'dealiasing = yes',\
                 '',\
                 'filtering = explicit',\
                 'filterWeight = 0.02',\
                 'filterCutoffRatio = 0.8',\
                 '',\
                 'userParam01 = '+str(int(hist_freq)),\
                 '',\
                 '[PROBLEMTYPE]',\
                 'equation = incompNS',\
                 'stressFormulation = no',\
                 '',\
                 '[PRESSURE]',\
                 'residualTol = 1e-8',\
                 'residualProj = yes',\
                 '',\
                 '[VELOCITY]',\
                 'residualTol = 1e-8',\
                 'residualProj = no',\
                 'density = 1.0',\
                 'viscosity = '+str(1./Re),\
                 'advection = yes']

    # Write square.par
    with open('square.par', 'w') as file:
        file.seek(0)
        for line in par_lines: file.write(line+'\n')
        file.truncate()

def get_data(base_dir, skip = 100, n_files = 1000):

    files = np.sort([f for f in os.listdir(base_dir) if f[:4]=='square0'])
    U = []
    V = []
    P = []
    Vort = []
    time = []
    
    _,Cx,Cy,_,_,_,mass = load_file(base_dir+files[0])

    for j in range(n_files):
                        
        t,u,v,p,T = load_file(base_dir+files[j+skip], return_xy=False)
        time.append(t)
        U.append(u)
        V.append(v)
        P.append(p)
        Vort.append(T)

    return time, mass, Cx, Cy, np.stack(U).T, np.stack(V).T, np.stack(P).T, np.stack(Vort).T

def load_file(file, return_xy=True, return_T = False):
    """
    Load velocity, pressure, and coorinates field from the file
    """

    field = nek.readnek(file)
    
    t = field.time
    nel = len(field.elem) # Number of spectral elements
    nGLL = field.elem[0].vel.shape[3] # Order of the spectral mesh
    n = nel*nGLL**2
    
    Cx = np.array([field.elem[i].pos[0, 0, j, k]
                   for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    Cy = np.array([field.elem[i].pos[1, 0, j, k]
                   for i in range(nel) for j in range(nGLL) for k in range(nGLL)])

    u = np.array([field.elem[i].vel[0, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    v = np.array([field.elem[i].vel[1, 0, j, k]
               for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    p = np.array([field.elem[i].pres[0, 0, j, k]
            for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    T = np.array([field.elem[i].temp[0, 0, j, k]
            for i in range(nel) for j in range(nGLL) for k in range(nGLL)])
    
    if return_xy: return t,Cx,Cy,u,v,p,T
    else: return t,u,v,p,T
