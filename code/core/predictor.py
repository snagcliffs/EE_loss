import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

class predictor(tf.keras.Model):

    def __init__(self, data_params=None, net_params=None, learning_params=None, restart_file=None, restart_dict=None):
        """
        May be initialized using the inputs below or via restart file (for a saved dictionary) or by the dictionary itself.

        Inputs:
            data_params
                m_hist  : number of history points in input time series
                r       : dimension of input time series

            net_params
                layer_sizes : list of length three
                    layer_sizes[0] : list of pre-LSTM layer sizes
                    layer_sizes[1] : list LSTM layer sizes
                    layer_sizes[2] : list of post-LSTM layer sizes
                activation  : netowrk activation e.g. 'relu', 'swish', 'sin', etc.

            learning_params
                l1_reg       : l1 regularization applied to non-LSTM layers
                l2_reg       : l2 regularization applied to non-LSTM layers
                lr           : learning rate
                decay_steps  : steps before reducing lr
                decay_rate   : geometric decay parameter
                loss_type    : which loss function to use

        Parameters for minimum density, self.min_py, and RE_lam may be set through predictor.set_min_py and predictor.set_RE_lam.
        If using OW or AOW loss, a density predictor will need to be specified through predictor.set_py_hat.
        """

        if restart_dict is None and restart_file is not None:
            restart_dict = np.load(restart_file,allow_pickle=True).item()

        if restart_dict is not None:
            data_params = restart_dict['data_params']
            net_params = restart_dict['net_params']
            learning_params = restart_dict['learning_params']
        
        super(predictor, self).__init__()
        
        m_hist,r = data_params
        self.m_hist = m_hist  
        self.r = r            
        
        layer_sizes, activation = net_params
        self.layer_sizes = layer_sizes
        self.activation = activation

        l1_reg, l2_reg, lr, decay_steps, decay_rate, loss_type = learning_params
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.reg = tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.loss_type = loss_type
        self.RE_lam = [1,0.1]

        # Build networks
        self.build_network()

        # For now set deault optimizer as Adam
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=lr,
                            decay_steps=decay_steps,
                            decay_rate=decay_rate)
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule)
        self.train_loss = []
        self.val_loss = []

        if restart_dict is not None:
            self.set_weights(restart_dict['weights'])
            self.train_loss = restart_dict['train_loss']
            self.val_loss = restart_dict['val_loss']
            self.test_loss = restart_dict['test_loss']


        # Minimal likelihood to use for y
        self.min_py = 1e-5

    def build_network(self):
        """
        Builds self.network
        """

        if self.m_hist == 1: layers = [tf.keras.layers.Input(shape=(self.r))]    
        else: layers = [tf.keras.layers.Input(shape=(self.m_hist,self.r))]
        
        # Pre-LSTM dense layers
        for l in self.layer_sizes[0]:
            layers.append(tf.keras.layers.Dense(l,activation=self.activation,
                                                kernel_regularizer=self.reg)(layers[-1]))
        
        # LSTM
        if self.m_hist > 1:

            if len(self.layer_sizes[1])==0 or self.layer_sizes[1][0]==0:
                layers.append(tf.keras.layers.Flatten()(layers[-1]))
            
            else:
                for j in range(len(self.layer_sizes[1])):

                    l = self.layer_sizes[1][j]
                
                    # Intermediate LSTM layers should return sequences
                    if j == len(self.layer_sizes[1]) - 1:
                        rs = False
                    else: rs =True 

                    layers.append(tf.keras.layers.LSTM(l,return_sequences=rs)(layers[-1]))
       
        # Post-LSTM dense layers
        for l in self.layer_sizes[2]:
            layers.append(tf.keras.layers.Dense(l,activation=self.activation,
                                                        kernel_regularizer=self.reg)(layers[-1]))
            
        layers.append(tf.keras.layers.Dense(1)(layers[-1]))
        self.network = tf.keras.Model(inputs=layers[0], outputs=layers[-1])

    @tf.function
    def compute_loss(self, X_hist, y, p_y):

        y_hat = self.network(X_hist)
        p_y = p_y + self.min_py

        try: 
            # Allows for non-integer powers on error.
            # Higher powers are less robust to outliers, hence may highlight extreme events.
            # Worked poorly in several test cases.
            self.loss_type=int(self.loss_type)
        except: 
            pass
        
        if type(self.loss_type)==int:
            """
            Mean n^th power of error w/ n=self.loss_type
            """
            loss = tf.reduce_mean(tf.abs(y-y_hat)**self.loss_type)

        elif self.loss_type == 'MSE':
            """
            Mean square error loss
            """

            loss = tf.reduce_mean((y-y_hat)**2)

        elif self.loss_type == 'MAE':
            """
            Mean absolute error loss
            """

            loss = tf.reduce_mean(tf.abs(y-y_hat))

        elif self.loss_type == 'RE':
            """
            Relative entropy loss
            """

            pos_loss = tf.reduce_mean(tf.math.exp(y_hat) - y_hat*tf.math.exp(y))
            neg_loss = tf.reduce_mean(tf.math.exp(-y_hat) + y_hat*tf.math.exp(-y))

            loss = self.RE_lam[0]*pos_loss + self.RE_lam[1]*neg_loss

        elif self.loss_type == 'OW':
            """
            OW loss.  MSE weighted by inverse pdf of output
            """
            loss = tf.reduce_mean((y-y_hat)**2 / p_y)

        elif self.loss_type == 'AOW':
            """
            AOW loss.  MSE weighted by inverse pdf of output and predicted output
            """

            # Likelihood of predicted y, given approximate p_y
            p_y_hat = self.py_hat.predict(y_hat)+self.min_py

            # Adjusted output weight
            aow = 1.0/p_y + p_y/p_y_hat

            # Weighted MSE
            loss = tf.reduce_mean((y-y_hat)**2 * aow)

        else:
            raise Exception('Loss type not implemented')

        return loss

    @tf.function
    def train_step(self, X_hist, y, p_y):
        """
        Single step of optimization
        """
        
        with tf.GradientTape() as tape:
            loss = self.compute_loss(X_hist, y, p_y)
            reg_loss = loss + tf.reduce_sum(self.network.losses)

        gradients = tape.gradient(reg_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

    def train_epoch(self, gen):
        
        train_batches, val_batches = gen.batches_per_epoch()[:2]
        
        train_loss = tf.keras.metrics.Mean()
        for j in range(train_batches):
            X_hist,y,p_y = gen.next_train()
            train_loss(self.train_step(X_hist,y,p_y))
            
                    
        val_loss = tf.keras.metrics.Mean()
        for j in range(val_batches):
            X_hist, y, p_y = gen.next_val()
            val_loss(self.compute_loss(X_hist,y,p_y))
        
        return train_loss.result(), val_loss.result()

    def compute_test_loss(self, gen):
        
        test_batches = gen.batches_per_epoch()[2]
                    
        test_loss = tf.keras.metrics.Mean()
        for j in range(test_batches):
            X_hist, y, p_y = gen.next_test()
            test_loss(self.compute_loss(X_hist,y,p_y))
        
        return test_loss.result()

    def train_model(self, training_params, gen, patience, save_file, verbose = 1):
        """
        Training loop

        Inputs:
            training_params  : epochs, and min_epochs
            gen              : an lstm_generator from generator.py
            patience         : number of epochs to wait for early stopping
            save_file        : path and filename for temporary weight files
            verbose          : if nonzero, will print train/val loss at each epoch

        """

        epochs, min_epochs = training_params

        for epoch in range(epochs):

            losses = self.train_epoch(gen)
       
            self.train_loss.append(losses[0].numpy())
            self.val_loss.append(losses[1].numpy())
            
            if epoch==0 or (epoch+1) % verbose == 0: 
                print('Epoch: {}, Train Loss: {}, Val Loss: {}'.format(epoch+1, 
                        np.round(self.train_loss[-1],6), 
                        np.round(self.val_loss[-1],6)))

            # Save weights if val loss has improved
            if self.val_loss[-1] == np.min(self.val_loss):
                self.save_weights(save_file)
            
            if epoch > min_epochs:
                if np.argmin(self.val_loss) <= epoch - patience: break

        self.load_weights(save_file)
        self.test_loss = self.compute_test_loss(gen).numpy()
        print('Final test loss:', self.test_loss)

    def save_weights(self, save_file):
        np.save(save_file, [w.numpy() for w in self.trainable_weights])

    def load_weights(self, save_file):
        self.set_weights(np.load(save_file+'.npy',allow_pickle=True))  

    def save_network(self, filename):
        np.save(filename, self.get_network_dict())

    def predict_full_data(self, gen):
        
        batch_size = gen.batch_size
        m = gen.m
        n_batches = int(np.ceil(m/batch_size))

        y_hat = np.zeros(m)
        
        for j in range(n_batches):

            batch_inds = np.arange(j*batch_size, np.min([gen.m,(j+1)*batch_size]))
            X_hist_batch = gen.get_X_hist(gen.X, batch_inds, gen.m_hist, gen.stride)
            if gen.m_hist == 1: X_hist_batch = X_hist_batch.reshape(X_hist_batch.shape[0], X_hist_batch.shape[2])

            y_hat[j*batch_size:(j+1)*batch_size] = self.network(X_hist_batch).numpy().flatten()

        return y_hat

    def get_network_dict(self):

        data_params = [self.m_hist,
                       self.r]

        net_params = [self.layer_sizes,
                      self.activation]

        learning_params = [self.l1_reg,
                           self.l2_reg,
                           self.lr,
                           self.decay_steps, 
                           self.decay_rate,
                           self.loss_type]

        network_dict = {'data_params' : data_params,
                        'net_params' : net_params, 
                        'learning_params' : learning_params,
                        'weights' : [w.numpy() for w in self.trainable_weights],
                        'train_loss' : self.train_loss,
                        'val_loss' : self.val_loss,
                        'test_loss' : self.test_loss}

        return network_dict

    def set_py_hat(self, py_hat):
        self.py_hat = py_hat

    def set_min_py(self, min_py):
        self.min_py = min_py

    def set_RE_lam(self, RE_lam):
        self.RE_lam = RE_lam

    def set_loss_type(self, loss_type):
        """
        Once compute_loss is called it is compiled by tf. 
        Changing self.loss_type afterwards does not work.
        """
        self.loss_type = loss_type


