import numpy as np
import subprocess
import argparse
from utils import *

def main(args):
    """
    
    """

    T = args.T
    dt = args.dt
    dT = args.dT

    hist_freq = args.hist_freq
    n_hist= args.n_hist
    Re = args.Re

    # Convert .geo files to .msh, .re2, and .ma2
    subprocess.run("gmsh -format msh2 -order 2 -2 square.geo", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("gmsh2nek << EOF \n2\nsquare\n0\n EOF", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("genmap << EOF \nsquare\n0.01\n EOF", shell=True, stdout=subprocess.DEVNULL)
    
    # Write par file and hist points file
    write_par_files(T,dt,dT,Re,hist_freq)
    write_hist_points(n_hist)

    # Make files to dump square data
    subprocess.run('touch forceCoeffs.dat', shell=True)

    # Files for large storage 
    subprocess.run("mkdir store", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("mv square.his store", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("mv square.re2 store", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("mv square.ma2 store", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("mv square.par store", shell=True, stdout=subprocess.DEVNULL)

    # files for home dir
    subprocess.run("mkdir run", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("cp build_case.sh run", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("cp run_case.sh run", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("cp SIZE run", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("cp square.usr run", shell=True, stdout=subprocess.DEVNULL)
    subprocess.run("mv forceCoeffs.dat run", shell=True, stdout=subprocess.DEVNULL)

    # clean unused files
    subprocess.run("./clean.sh", shell=True, stdout=subprocess.DEVNULL)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Run and step lengths
    parser.add_argument('--T', default=5000, type=float, help='Length of simulation')
    parser.add_argument('--dt', default=0.001, type=float, help='Simulation timestep')
    parser.add_argument('--dT', default=100, type=float, help='IO timestep')

    # Number of history points to track
    parser.add_argument('--hist_freq', default=int(10), type=int, help='Number of steps between writing history points')
    parser.add_argument('--n_hist', default=49, type=int, help='Number of history points along exterior of square')

    # Reynolds number
    parser.add_argument('--Re', default=5000, type=int, help='Re')

    args = parser.parse_args()
    main(args)
