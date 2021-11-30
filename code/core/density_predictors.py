import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import GPy
from sklearn.mixture import GaussianMixture

tf.keras.backend.set_floatx('float64')

class tf_GP():
    """
    Estimator of probability density function of output.
    Takes as input a GPy GP with Matern-5/2 covariance function trained to predict log(p_y)
    """

    def __init__(self, GP, kernel_type = 'M52'):
        
        self.X = tf.constant(GP.X)
        self.y = tf.constant(GP.Y)
        
        self.mean = GP.mean_function.f(0)
        self.obs_variance = tf.constant(GP.kern.white_hetero.variance)

        if kernel_type == 'M32':
            self.amplitude    = tf.constant(np.sqrt(GP.kern.Mat32.variance))
            self.lengthscale  = tf.constant(GP.kern.Mat32.lengthscale)
            self.K = tfp.math.psd_kernels.MaternThreeHalves(amplitude=self.amplitude, 
                                                           length_scale=self.lengthscale)

        elif kernel_type == 'M52':
            self.amplitude    = tf.constant(np.sqrt(GP.kern.Mat52.variance))
            self.lengthscale  = tf.constant(GP.kern.Mat52.lengthscale)
            self.K = tfp.math.psd_kernels.MaternFiveHalves(amplitude=self.amplitude, 
                                                           length_scale=self.lengthscale)

        elif kernel_type == 'RBF':
            self.amplitude    = tf.constant(np.sqrt(GP.kern.RBF.variance))
            self.lengthscale  = tf.constant(GP.kern.RBF.lengthscale)
            self.K = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=self.amplitude, 
                                                                 length_scale=self.lengthscale)

        else:
            raise Exception("kernel type not recognized.  please use on of ['M32','M52','RBF']")

                
        K_XX = self.K.matrix(self.X, self.X) + self.obs_variance*tf.eye(self.X.shape[0],dtype=tf.float64)
        chol_K_XX = tf.linalg.cholesky(K_XX[0,:,:])
        
        self.alpha = tf.linalg.cholesky_solve(chol_K_XX, self.y - self.mean)
    
    @tf.function
    def predict(self,x):
        
        KxX = self.K.matrix(x,self.X)[0,...]
        return tf.exp(KxX @ self.alpha + self.mean)
    
    @tf.function
    def predict_grad(self,x):
                
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = self.predict(x)
            
        return g.gradient(y, x)   

class tf_GMM():
    """
    Estimator of probability density function of output.

    Inputs:
        locs    : means of GMM
        scales  : standard deviation of GMM
        weights : micture weights
    """

    def __init__(self, locs, scales, weights):
        
        self.locs = [tf.constant(l, dtype=tf.float64) for l in locs]
        self.scales = [tf.constant(s, dtype=tf.float64) for s in scales]
        self.weights = [tf.constant(w, dtype=tf.float64) for w in weights]
        
        self.dists = [tfp.distributions.Normal(loc, scale) for (loc,scale) in zip(self.locs,self.scales)]
    
    @tf.function
    def predict(self,x):

        probs = [dist.prob(x) for dist in self.dists]
        return tf.reduce_sum([w*p for (w,p) in zip(self.weights,probs)], axis=0)
    
    @tf.function
    def predict_grad(self,x):
                
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = self.predict(x)
            
        return g.gradient(y, x)   

def get_hist(y, n_bins):
    """
    Return a histogram estimate of density and uncertainty in log-density.
    """

    density, bins = np.histogram(y, density = True, bins=n_bins)
    centers = (bins[1:]+bins[:-1])/2
    inds = np.where(density != 0)
    L = np.max(bins) - np.min(bins)
    log_sample_var = centers.size/(L**2*y.size)*(1 + 1/(density+1e-16))

    return centers[inds], density[inds], log_sample_var[inds]

def get_kde(y, n_bw=10):
    """
    Uses CV to select bandwidth and returns kde.
    Not currently used.
    """
    
    bandwidths = np.logspace(-3, -1, n_bw)*np.ptp(y)
    cv = ShuffleSplit(n_splits=3, test_size=100/y.size, random_state=0)
    
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': bandwidths},
                        cv=cv)
    
    grid.fit(y[:, np.newaxis])
    bw = grid.best_params_['bandwidth']
    kde = KernelDensity(bandwidth=bw).fit(y[:,np.newaxis])
        
    return kde

def get_density_function(y, method='GP', kernel_type='M52', num_restarts=1, n_bins=100, n_mix = 10):
    """
    Return either a GP or GMM implmented in tensorflow.
    """

    if method == 'GP':


        centers, density, log_sample_var = get_hist(y, n_bins = n_bins)
        inds = np.where(density != 0)
        m = len(inds[0])
    
        if kernel_type == 'M32': 
            kern = GPy.kern.Matern32(input_dim=1)
        elif kernel_type == 'M52': 
            kern = GPy.kern.Matern52(input_dim=1)
        elif kernel_type == 'RBF': 
            kern = GPy.kern.RBF(input_dim=1)
        else:
            raise Exception('GP kernel type not recognized')
        
        # Initialize heterscedatic noise using variance of log-histogram-estimate.
        kern = kern + GPy.kern.WhiteHeteroscedastic(input_dim=1, num_data=m, variance=log_sample_var[inds])
        
        # Set mean to -16, so that mean predicted density far away from data is 10^-16
        mf = GPy.core.Mapping(1,1)
        mf.f = lambda x: np.log(10**-16)
        mf.update_gradients = lambda a,b: None

        # GP object with specified kernel and training data
        GP = GPy.models.GPRegression(centers[:,np.newaxis][inds], 
                                     np.log(density[inds][:,np.newaxis]), 
                                     kernel=kern,
                                     mean_function=mf)

        opts = GP.optimize_restarts(num_restarts = num_restarts)
    
        return tf_GP(GP, kernel_type)

    else:

        gmm = GaussianMixture(n_components=n_mix, n_init=num_restarts).fit(y.reshape(y.size,1))

        weights = gmm.weights_.flatten()
        locs = gmm.means_.flatten()
        scales = np.sqrt(gmm.covariances_.flatten())

        return tf_GMM(locs, scales, weights)











