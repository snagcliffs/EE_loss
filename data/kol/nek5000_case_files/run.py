import numpy as np
import subprocess
import os
import argparse
from utils import *

def main(args):
    """
    
    """

    T = args.T
    dt = args.dt
    dT = args.dT
    hist_freq = args.hist_freq
    n_hist = args.n_hist
    Re = args.Re
    nproc = args.nproc

    # Convert .box files to .re2 and ma2
    subprocess.run("genbox << EOF \nKolmogorov.box\n EOF", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("mv box.re2 Kolmogorov.re2", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("genmap << EOF \nKolmogorov\n0.01\n EOF", shell=True, stdout=subprocess.DEVNULL)

    # Write and adjust files as needed
    write_par_files(T,dt,dT,Re,hist_freq)
    write_hist_points(n_hist)

    # Build and run
    subprocess.run("makenek Kolmogorov", shell=True, stdout=subprocess.DEVNULL)
    run_cmd = "nekmpi Kolmogorov "+str(nproc)
    with open('logfile.txt', "w") as outfile:
       subprocess.run(run_cmd, shell=True, executable="/bin/bash", stdout=outfile)

    parse_history_points(n_hist)

    # Move data files and restart files to their own directory
    subprocess.run("mkdir Re_"+str(int(Re)), shell=True)
    subprocess.run("mkdir Re_"+str(int(Re))+"/outfiles", shell=True)
    subprocess.run("mv *.dat Re_"+str(int(Re)), shell=True)
    subprocess.run("mv logfile.txt Re_"+str(int(Re)), shell=True)
    subprocess.run("mv *.his Re_"+str(int(Re)), shell=True)
    subprocess.run("mv *.npy Re_"+str(int(Re)), shell=True)
    subprocess.run("mv *0.f* Re_"+str(int(Re))+"/outfiles", shell=True)

    # Remove junk
    subprocess.run("./clean.sh", shell=True)

    """
    Add in any post-processing here
    """

if __name__ == "__main__":
    """
    Run time ~ 11 hours on 16 cores for T=4e4 and dt=0.0025.
    Could probably get away with much courser grid.
    """

    parser = argparse.ArgumentParser()

    # Run and step lengths
    parser.add_argument('--T', default=40000, type=float, help='Length of simulation')
    parser.add_argument('--dt', default=0.0025, type=float, help='Simulation timestep')
    parser.add_argument('--dT', default=1000, type=float, help='IO timestep')

    # Number of history points to track
    parser.add_argument('--hist_freq', default=int(40), type=int, help='Number of steps between writing history points')
    parser.add_argument('--n_hist', default=8, type=int, help='Number of history points along each direction')

    # Reynolds
    parser.add_argument('--Re', default=40, type=int, help='Reynolds number')

    # Number of processors
    parser.add_argument('--nproc', default=16, type=int, help='Number of proceesors')

    args = parser.parse_args()

    main(args)
