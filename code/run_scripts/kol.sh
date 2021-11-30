#!/bin/bash
#
# Trains neural network on kolmorogorov dataset with fourier mode coefficients as input
#

args="--data_path ../../data/kol/ \
      --input_file fourier.npy \
      --output_file D.npy \
      --save_file kol \
      --save_path ../saved_results/kol/ \
      --n_restarts 20 \
      --m_hist 20 \
      --stride 10 \
      --activation swish \
      --epochs 2500 \
      --min_epochs 10 \
      --contiguous_sets all \
      --sample_rate 1 \
      --train_frac 0.5 \
      --val_frac 0.1"

for tau in 2 4 6 8 10
   do
   for loss_type in MSE RE OW AOW
   do
      python ../core/train.py --loss_type $loss_type --tau $tau $args
   done
done

