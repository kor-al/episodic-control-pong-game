import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

image_h = 84 #'Original image height')
image_w =  84 # 'Original image width')
sg = tf.contrib.bayesflow.stochastic_graph
st = tf.contrib.bayesflow.stochastic_tensor
distributions = tf.contrib.distributions

def inference_network(latent_dim):
  """Construct an inference network parametrizing a Gaussian.
  Args:
    x: A batch of images
    latent_dim: The latent dimensionality.
  Returns:
    mu: Mean parameters for the variational family Normal
    sigma: Standard deviation parameters for the variational family Normal
  """
  x = tf.placeholder(tf.float32, [None, image_h, image_w, 1])
  tf.summary.image('data', x)

  kernels = [32, 32, 64, 64]
  kernel_sizes = [4, 5, 5, 4]
  kernel_strides = [2, 2, 2, 2]

  with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding = "VALID"):
    net = slim.conv2d(x, kernels[0], kernel_size = kernel_sizes[0],stride = kernel_strides[0])
    for i in range(1,4):
        net = slim.conv2d(net, kernels[i], kernel_size = kernel_sizes[i],stride = kernel_strides[i])
        print(i, net.shape)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
  gaussian_params = slim.fully_connected(net, latent_dim * 2,  activation_fn=None)


  # The mean parameter is unconstrained
  mu = gaussian_params[:, :latent_dim]
  # The standard deviation must be positive. Parametrize with a softplus and
  # add a small epsilon for numerical stability
  sigma = 1e-6 + tf.nn.softplus(gaussian_params[:, latent_dim:])
  sigma = tf.sqrt(tf.exp(sigma))
   # encoder output log(sigma**2) (its domain is -inf, inf)
  return x, mu, sigma


def generative_network(z):
  """Build a generative network parametrizing the likelihood of the data
  Args:
    z: Samples of latent variables
  Returns:
    mu: Mean parameters for the Normal likelihood of the data
    sigma: Standard deviation parameters for the Normal likelihood of the data
  """
  net = slim.fully_connected(z, 576, activation_fn=tf.nn.relu)
  net = tf.reshape(net, [-1,3, 3,64])

  kernels = [64, 64, 32, 32 ]
  kernel_sizes = [4, 5, 5, 4]
  kernel_strides = [2, 2, 2, 2]

  with slim.arg_scope([slim.convolution2d_transpose], activation_fn=tf.nn.relu, padding = "VALID"):
    for i in range(0,4):
        net = slim.conv2d_transpose(net, kernels[i], kernel_size = kernel_sizes[i],stride = kernel_strides[i])
    gaussian_params = slim.conv2d_transpose(net, 2, 1, scope='output')

  # The mean parameter is unconstrained
  mu = gaussian_params[:, :,:, 0]
  mu = tf.reshape(mu,[-1, image_h, image_w, 1])
  # The standard deviation must be positive. Parametrize with a softplus and
  # add a small epsilon for numerical stability
  sigma = 1e-6 + tf.nn.softplus(gaussian_params[:,:,:,1])
  sigma = tf.reshape(sigma,[-1, image_h, image_w, 1])
  sigma = tf.sqrt(tf.exp(sigma))
  return mu, sigma


class VAE(object):
    def __init__(self, latent_dim):
        # x is an observsation batch
        self.n_samples = 1
        with tf.variable_scope('model'):
            self.x, self.q_mu, self.q_sigma = inference_network(latent_dim)
            self.q_z = st.StochasticTensor(distributions.Normal(loc=self.q_mu, scale =self.q_sigma))
            self.p_x_given_z_mu, self.p_x_given_z_sigma  = generative_network(z=self.q_z)
            self.z = self.q_z.distribution.sample(self.n_samples)#prior z

        with tf.variable_scope('model', reuse=True):
            self.p_x_given_z = distributions.Normal(loc=self.p_x_given_z_mu, scale = self.p_x_given_z_sigma)
            self.p_z = distributions.Normal(loc=np.zeros(latent_dim, dtype=np.float32), scale =np.ones(latent_dim, dtype=np.float32))

        # Define the loss function.

        self.kl = tf.reduce_sum(distributions.kl(self.q_z.distribution, self.p_z), 1)
        self.expected_log_likelihood = tf.reduce_sum(self.p_x_given_z.log_prob(self.x), [1, 2, 3])
        self.elbo = tf.reduce_sum(self.expected_log_likelihood - self.kl, 0)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5)
        self.train_step = optimizer.minimize(-self.elbo)

        tf.summary.scalar("elbo", self.elbo)
        self.summaries = tf.summary.merge_all()

    def get_state(self, session, obs):
        z = session.run([self.z], feed_dict={self.x: obs.reshape(1,image_h, image_w, 1)})
        z = np.ravel(z)
        return z


    def train(self, session, obs_batch):
        # perform gradient step
        summaries, _, elbo, z = session.run([self.summaries, self.train_step, self.elbo, self.z], feed_dict={self.x: obs_batch})
        return summaries, elbo, z