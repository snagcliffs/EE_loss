#!/bin/bash
#
# Trains neural network on square dataset with pressure measurements as input
#

args="--data_path ../../data/square/ \
      --input_file pres_hist.npy \
      --output_file Cd.npy \
      --save_file square \
      --save_path ../saved_results/square/ \
      --m_hist 5 \
      --stride 10 \
      --patience 5 \
      --lstm_size 16 \
      --lr 0.001 \
      --contiguous_sets all \
      --min_epochs 20 \
      --n_restarts 20 \
      --sample_rate 5 \
      --train_frac 0.5 \
      --val_frac 0.1"

for tau in 0.5 1 1.5 2
do  
   for loss_type in MSE RE OW AOW
   do
      python ../core/train.py --loss_type $loss_type --tau $tau $args
   done
done
