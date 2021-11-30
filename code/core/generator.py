import numpy as np
from numba import njit

class lstm_generator():
    """
    Generator for training neural networks to predict drag coefficient.

    Inputs:
        data_path       : path to data files (should contain two files below and t.npy)
        input_file      : input time series
        output_file     : target time series
        tau             : lead time for prediction
        batch_size      : number of examples per batch
        min_time        : legnth of transient to ignore
        m_hist          : length of input time series
        stride          : spacing of input time series data
        train_frac      : fraction of data used for training
        val_frac        : fraction of data used for validation
        sample_rate     : subsample data
        noise_level     : std. dev. of noise added to X_train as fraction of std. dev of X
        contiguous_sets : how to split data.  See self.split_dataset.
    """
    
    def __init__(self, 
                 data_path, 
                 input_file, 
                 output_file, 
                 tau=0, 
                 batch_size=1000, 
                 min_time=200, 
                 m_hist=200, 
                 stride=10,
                 train_frac=0.5,
                 val_frac=0.1,
                 sample_rate=1,
                 noise_level=0.1,
                 contiguous_sets='test'):

        self.data_path = data_path
        self.input_file = input_file
        self.output_file = output_file

        self.tau = tau
        self.batch_size = batch_size
        self.min_time = min_time
        self.m_hist = m_hist
        self.stride = stride

        self.train_frac = train_frac
        self.val_frac = val_frac
        assert self.train_frac + self.val_frac < 1
        self.contiguous_sets = contiguous_sets
        self.sample_rate = sample_rate

        # Load data
        self.load_data()
        self.compute_output_histogram()

        # Size of data (subtract (m_hist-1)*stride for NN inputs)
        self.m, self.r = self.X.shape
        self.m = self.m - self.rnn_input_len

        # Split dataset into train/val/test
        self.split_dataset()
        self.train_queue = np.random.permutation(self.train_inds)
        self.val_queue = np.random.permutation(self.val_inds)
        self.test_queue = np.random.permutation(self.test_inds)

        # Add noise to training data
        self.noise_level = noise_level
        if self.noise_level !=0: self.add_noise = True
        else: self.add_noise = False

    def load_data(self):

        # Load data from files
        X = np.load(self.data_path + self.input_file)
        y = np.load(self.data_path + self.output_file)
        t = np.load(self.data_path + 't.npy')
        m = len(t)
        min_ind = np.max([np.min(np.where(t > self.min_time)),\
                          (self.m_hist-1)*self.stride])

        self.dt = t[1] - t[0]
        self.tau_steps = int(self.tau / self.dt)
        self.rnn_input_len = (self.m_hist-1)*self.stride
        self.sim_time = t[min_ind:m-self.tau_steps]

        self.X = X[min_ind-self.rnn_input_len:m-self.tau_steps,:]
        self.y = y[min_ind+self.tau_steps:]
        
        self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)
        self.y = (self.y - np.mean(self.y)) / np.std(self.y)
        
    def compute_output_histogram(self, n_bins = 100):

        # Approximation of y-density with histogram
        y_hist, y_bins = np.histogram(self.y, density = True, 
                                      bins=np.linspace(np.min(self.y),np.max(self.y),n_bins))

        y_order = np.argsort(self.y)
        y_sorted = np.sort(self.y)
        inds = [np.min(np.where(y_sorted >= e)) for e in y_bins[:-1]] + [len(self.y)]
        self.y_density = np.zeros_like(self.y)
        for j in range(len(y_bins)-1):
            self.y_density[y_order[inds[j]:inds[j+1]]] = y_hist[j]

        self.mean_density = np.mean(self.y_density)
        self.bin_centers = (y_bins[1:]+y_bins[:-1])/2
        self.y_hist = y_hist
        self.y_bins = y_bins

    def compute_output_density(self, predictor):
        self.y_density = predictor.predict(self.y.reshape(self.y.size,1)).numpy()

    def split_dataset(self):

        self.m_train = int(self.m*self.train_frac)
        self.m_val = int(self.m*self.val_frac)
        self.m_test = self.m - self.m_train - self.m_val
        
        if self.contiguous_sets == 'all':
            """
            train, val, and test all contiguous
            test will be separated from train by val.
            """
            self.train_inds = np.arange(self.m_train)
            self.val_inds = self.m_train + np.arange(self.m_val)
            self.test_inds = self.m_train + self.m_val + np.arange(self.m_test)

        elif self.contiguous_sets == 'test':
            """
            Train and val mixed up from first m_train+m_val indices, test is one contiguous set
            """
            self.train_inds = np.random.choice(self.m_train+self.m_val,self.m_train,replace=False)
            self.val_inds = np.array(list(set(np.arange(self.m_train + self.m_val)) - set(self.train_inds)))
            self.test_inds = self.m_train + self.m_val + np.arange(self.m_test)

        elif self.contiguous_sets == 'none':
            """
            All datasets randomly mixed
            """
            self.train_inds = np.random.choice(self.m,self.m_train,replace=False)
            self.val_inds = np.random.choice(list(set(np.arange(self.m)) - set(self.train_inds)), self.m_val, replace=False)
            self.test_inds = np.array(list(set(np.arange(self.m)) - set(self.train_inds) - set(self.val_inds)))

        else:
            raise Exception('contiguous_sets option not recognized')

        self.train_inds = self.train_inds[::self.sample_rate]
        self.val_inds = self.val_inds[::self.sample_rate]
        self.test_inds = self.test_inds[::self.sample_rate]
        
        self.m_train = len(self.train_inds)
        self.m_val = len(self.val_inds)
        self.m_test = len(self.test_inds)

        self.train_val_inds = np.concatenate([self.train_inds, self.val_inds])
        self.train_batches = int(np.ceil(self.m_train/self.batch_size))
        self.val_batches = int(np.ceil(self.m_val/self.batch_size))
        self.test_batches = int(np.ceil(self.m_test/self.batch_size))

    def batches_per_epoch(self):

        return self.train_batches, self.val_batches, self.test_batches

    def next_train(self):

        batch_inds = self.train_queue[:self.batch_size]
        self.train_queue = self.train_queue[self.batch_size:]
        if len(self.train_queue) == 0: 
            self.train_queue = np.random.permutation(self.train_inds)

        return self.get_batch(batch_inds, add_noise = self.add_noise)

    def next_val(self):

        batch_inds = self.val_queue[:self.batch_size]
        self.val_queue = self.val_queue[self.batch_size:]
        if len(self.val_queue) == 0: self.val_queue = np.random.permutation(self.val_inds)

        return self.get_batch(batch_inds)

    def next_test(self):

        batch_inds = self.test_queue[:self.batch_size]
        self.test_queue = self.test_queue[self.batch_size:]
        if len(self.test_queue) == 0: self.test_queue = np.random.permutation(self.test_inds)

        return self.get_batch(batch_inds)

    @staticmethod
    @njit
    def get_X_hist(X, batch_inds, m_hist, stride):

        X_hist_batch = np.zeros((len(batch_inds), m_hist, X.shape[1]))

        for i in range(len(batch_inds)):
            for j in range(m_hist):
                for k in range(X.shape[1]):
                    X_hist_batch[i,m_hist-j-1,k] = X[(m_hist-1)*stride+batch_inds[i]-j*stride, k]

        return X_hist_batch

    def get_batch(self, batch_inds, add_noise=False):
        """

        """

        if self.m_hist > 1:
            X_batch = self.get_X_hist(self.X, batch_inds, self.m_hist, self.stride)
        else:
            X_batch = self.X[batch_inds,...]

        y_batch = self.y[batch_inds].reshape(len(batch_inds),1)
        py_batch = self.y_density[batch_inds].reshape(len(batch_inds),1)

        if add_noise:
            X_batch = X_batch + self.noise_level*np.random.randn(*X_batch.shape)
            #y_batch = y_batch + self.noise_level*np.random.randn(*y_batch.shape)

        return X_batch, y_batch, py_batch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.train_batches = int(np.ceil(self.m_train/self.batch_size))
        self.val_batches = int(np.ceil(self.m_val/self.batch_size))
        self.test_batches = int(np.ceil(self.m_test/self.batch_size))


