This files set up a simulation to run on a computer using the Slurm Workload Manager.

To run simulation on an XSEDE computer:
    1) Run python gen_case_files.py
       This creates two folders called run and store
    2) Move files to server       
       Files from run will be moved to a working directory
       Files from store will be moved to a directory with storage for nek5000 output
    3) In directory with run files run: ./build_case.sh
    4) Change line 8 of run_case.sh to reflect correct account
    5) Change line 21 to point to directory where store files are located
    6) run: sbatch run_case.sh

The function parse_history_points from utils.py may then be applied to square.his to obtain pressure data.
Force coefficients are stored in forceCoeffs.dat
Simulation for 20k time units took just under 43.5 hours on 128 cores.