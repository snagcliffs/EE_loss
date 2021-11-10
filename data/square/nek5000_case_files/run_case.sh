#!/bin/bash
#SBATCH -p compute                              # what type of node to run on
#SBATCH --nodes=1                               # number of nodes
#SBATCH --ntasks-per-node=128                   # number of processors / node (<=256)
#SBATCH --mem=128G                              # maximum memory allocation
#SBATCH --time=48:00:00                         # time requested
#SBATCH -J square                               # name of job
#SBATCH -A (account)                            # account to charge
#SBATCH -o output_file.o                        # output file
#SBATCH -e error_file.e                         # error file
#SBATCH --export=ALL                            # 

module purge                                    # remove loaded modules
module load cpu                                 # will run on cpu node
module load slurm                               # load slurm
module load gcc                                 # load gnu compilers
module load openmpi                             # load openmpi

casename='square'               # .re2 filename
echo $casename > SESSION.NAME   # first line of SESSION.NAME
echo '/path/to/store/' >> SESSION.NAME   # second line of SESSION.NAME
srun -n 128 ./nek5000 > $casename.log 2>&1       # run
