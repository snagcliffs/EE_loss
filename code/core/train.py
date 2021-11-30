import numpy as np
import argparse
import subprocess
np.random.seed(0)

import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from generator import lstm_generator
from predictor import predictor
from density_predictors import get_density_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # uncomment to hide GPU

def main(args):
    """
    
    """

    subprocess.run('mkdir '+args.save_path, shell=True)
    subprocess.run('mkdir '+args.save_path+'temp_files', shell=True)

    ##
    ## Set up data generator
    ##
    gen = lstm_generator(args.data_path, 
                         args.input_file, 
                         args.output_file, 
                         args.tau, 
                         m_hist = args.m_hist, 
                         stride = args.stride,  
                         batch_size=args.batch_size,
                         contiguous_sets = args.contiguous_sets,
                         train_frac = args.train_frac,
                         val_frac = args.val_frac,
                         sample_rate = args.sample_rate,
                         min_time = args.min_time,
                         noise_level = args.noise_level)
    
    ##
    ## Set up NN parameters
    ##
    data_params = [args.m_hist,gen.r]

    layer_sizes = [np.asarray(args.input_dense_layer_sizes), 
                   np.asarray(args.lstm_size), 
                   np.asarray(args.output_dense_layer_sizes)]
    net_params = [layer_sizes,
                  args.activation]
    
    if args.decay_steps is None: decay_steps = gen.train_batches
    else: decay_steps = args.decay_steps

    learning_params = [args.l1_reg,
                       args.l2_reg,
                       args.lr,
                       decay_steps,
                       args.decay_rate,
                       args.loss_type]

    ##
    ## Get approximation of p(y) using training/validation data
    ##
    py_hat = get_density_function(gen.y[gen.train_val_inds], method=args.density_method)
    gen.compute_output_density(py_hat)

    ##
    ## Loop over n_restarts
    ##
    results_dict = {'true' : gen.y, \
                    'tau' : args.tau, \
                    'val_loss' : [], \
                    'test_loss' : [], \
                    'train_loss' : []}

    for trial in range(args.n_restarts):

        print("#\n#\n#\n#\n#")
        print("Loss="+args.loss_type+", tau="+str(args.tau)+", trial "+str(trial+1)+" of "+str(args.n_restarts))
        print("#\n#\n#\n#\n#")

        model = predictor(data_params,net_params,learning_params)
        model.set_py_hat(py_hat)
        model.train_model([args.epochs,args.min_epochs], gen, args.patience, args.save_path+'temp_files/'+args.save_file)
        model.save_network(args.save_path+args.save_file+'_'+args.loss_type+'_tau'+str(args.tau)+'_trial'+str(trial+1))

        results_dict['NN_'+str(trial+1)] = model.predict_full_data(gen)
        results_dict['val_loss'].append(np.min(model.val_loss))
        results_dict['test_loss'].append(model.test_loss)
        results_dict['train_loss'].append(model.train_loss[np.argmin(model.val_loss)])
    
    np.save(args.save_path + 'results_loss_'+args.loss_type+'_tau'+str(args.tau), results_dict)

if __name__ == "__main__":
    """

    """

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_path', default='../../data/square/', type=str, help='Path to datasets')
    parser.add_argument('--input_file', default='pres_hist.npy', type=str, help='Name of input data file')
    parser.add_argument('--output_file', default='Cd.npy', type=str, help='Name of output data file')
    parser.add_argument('--min_time', default=200, type=float, help='Length of transient to ignore')
    parser.add_argument('--sample_rate', default=10, type=int, help='Sample rate')

    # Save location
    parser.add_argument('--save_path', default='../saved_results/', type=str, help='Path to save results')
    parser.add_argument('--save_file', default='kol_fourier', type=str, help='File name for saved results')

    # Network structure
    parser.add_argument('--input_dense_layer_sizes', type=int, nargs='+', default=[4,8,16], help='Pre-LSTM dense layers')
    parser.add_argument('--lstm_size', type=int, nargs='+', default=[32], help='Units in LSTM layer')
    parser.add_argument('--output_dense_layer_sizes', type=int, nargs='+', default=[16,8,4], help='Post-LSTM dense layers')
    parser.add_argument('--activation', type=str, default='swish', help='Activation')
    parser.add_argument('--loss_type', type=str, default='MSE', help='Loss type')

    # NN input parameters
    parser.add_argument('--m_hist', default=50, type=int, help='Number of history points to use')
    parser.add_argument('--stride', default=1, type=int, help='Spacing of history points')

    # Lead time
    parser.add_argument('--tau', default=1.0, type=float, help='Lead time.')

    # Approximation of output density
    parser.add_argument('--density_method', default='GP', type=str, help='Method for approximating output density')

    # NN training details
    parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--decay_rate', default=1, type=float, help='Decay rate')
    parser.add_argument('--decay_steps', default=None, help='Decay stesps (int)')
    parser.add_argument('--train_frac', default=0.7, type=float, help='Fraction of dataset used for training')
    parser.add_argument('--val_frac', default=0.15, type=float, help='Fraction of dataset used for validation')
    parser.add_argument('--l1_reg', default=0, type=float, help='L1 penalty on NN weights')
    parser.add_argument('--l2_reg', default=0, type=float, help='L2 penalty on NN weights')
    parser.add_argument('--noise_level', default=0.1, type=float, help='Std dev of Gaussian noise.')
    parser.add_argument('--n_restarts', default=5, type=int, help='Number of restarts.')
    parser.add_argument('--min_epochs', default=10, type=int, help='Minimum number of epochs.')
    parser.add_argument('--epochs', default=1000, type=int, help='Maximum number of epochs.')
    parser.add_argument('--batch_size', default=1000, type=int, help='Batch size')
    parser.add_argument('--patience', default=5, type=int, help='Optimization patience.')
    parser.add_argument('--contiguous_sets', default='all', type=str, help='Data partition.')

    args = parser.parse_args()

    main(args)

