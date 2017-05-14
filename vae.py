import itertools
import matplotlib as mpl
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import seaborn as sns
import glob


from matplotlib import pyplot as plt
from scipy.misc import imsave
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

sns.set_style('whitegrid')

sg = tf.contrib.bayesflow.stochastic_graph
st = tf.contrib.bayesflow.stochastic_tensor
distributions = tf.contrib.distributions


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/dat/', 'Directory for data')
flags.DEFINE_string('logdir', '/tmp/log/', 'Directory for logs')



image_h = 84 #'Original image height')
image_w =  84 # 'Original image width')
N = 50000 #observations

flags.DEFINE_integer('latent_dim', 32, 'Latent dimensionality of model')
flags.DEFINE_integer('batch_size', 100, 'Minibatch size')
flags.DEFINE_integer('n_samples', 1, 'Number of samples to save')
flags.DEFINE_integer('print_every', 100, 'Print every n iterations')
flags.DEFINE_integer('hidden_size', 24, 'Hidden size for neural networks')
flags.DEFINE_integer('n_iterations', 400000, 'number of iterations')



FLAGS = flags.FLAGS

#-------------------------------------------------------------------------------
def next_batch(batch_size, image):
    """
    Return a total of `num` samples from the array `data`.
    """
    num_preprocess_threads = 10
    min_queue_examples = 256

    images = tf.train.shuffle_batch([image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    return images


def read_data(path, num):
     imglist = glob.glob(path+"*.png")
     filename_queue = tf.train.string_input_producer(imglist)
     reader = tf.WholeFileReader()
     key, value = reader.read(filename_queue)
     image = tf.image.decode_png(value) # use png or jpg decoder based on your files.
     image.set_shape((image_h, image_w, 1))
     image = tf.to_float(image, name='ToFloat')
     return image



path = r"D:/Alice/Documents/HSE/masters/observations/"

#----------------------------------------------------------------


def inference_network(x, latent_dim):
  print('inference')
  """Construct an inference network parametrizing a Gaussian.
  Args:
    x: A batch of images
    latent_dim: The latent dimensionality.
    hidden_size: The size of the neural net hidden layers.
  Returns:
    mu: Mean parameters for the variational family Normal
    sigma: Standard deviation parameters for the variational family Normal
  """
  # with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
  #   net = slim.flatten(x)
  #   net = slim.fully_connected(net, hidden_size)
  #   net = slim.fully_connected(net, hidden_size)
  #   gaussian_params = slim.fully_connected(
  #       net, latent_dim * 2, activation_fn=None)

  kernels = [32, 32, 64, 64]
  kernel_sizes = [4, 5, 5, 4]
  kernel_strides = [2, 2, 2, 2]

  with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding = "VALID"):
    net = slim.conv2d(x, kernels[0], kernel_size = kernel_sizes[0],stride = kernel_strides[0])
    print(0, net.shape)
    for i in range(1,4):
        net = slim.conv2d(net, kernels[i], kernel_size = kernel_sizes[i],stride = kernel_strides[i])
        print(i, net.shape)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
    print('relu',net.shape)
  gaussian_params = slim.fully_connected(net, latent_dim * 2,  activation_fn=None)
  print('result',gaussian_params.shape)
  print(gaussian_params.shape , ' params')


  # The mean parameter is unconstrained
  mu = gaussian_params[:, :latent_dim]
  # The standard deviation must be positive. Parametrize with a softplus and
  # add a small epsilon for numerical stability
  sigma = 1e-6 + tf.nn.softplus(gaussian_params[:, latent_dim:])
  sigma = tf.sqrt(tf.exp(sigma))
   # encoder output log(sigma**2) (its domain is -inf, inf)
  return mu, sigma


def generative_network(z, latent_dim):
  """Build a generative network parametrizing the likelihood of the data
  Args:
    z: Samples of latent variables
    hidden_size: Size of the hidden state of the neural net
  Returns:
    bernoulli_logits: logits for the Bernoulli likelihood of the data
  """
  # with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
  #   net = slim.fully_connected(z, hidden_size)
  #   net = slim.fully_connected(net, hidden_size)
  #   bernoulli_logits = slim.fully_connected(net, image_h*image_w, activation_fn=None)
  #   bernoulli_logits = tf.reshape(bernoulli_logits, [-1, image_h, image_w, 1])
  # return bernoulli_logits
  net = slim.fully_connected(z, 576, activation_fn=tf.nn.relu)
  net = tf.reshape(net, [-1,3, 3,64])

  kernels = [64, 64, 32, 32 ]
  kernel_sizes = [4, 5, 5, 4]
  kernel_strides = [2, 2, 2, 2]

  with slim.arg_scope([slim.convolution2d_transpose], activation_fn=tf.nn.relu, padding = "VALID"): # , kernel_size = kernel_sizes, stride = kernel_strides
    for i in range(0,4):
        net = slim.conv2d_transpose(net, kernels[i], kernel_size = kernel_sizes[i],stride = kernel_strides[i])
        print(i, net.shape)
    gaussian_params = slim.conv2d_transpose(net, 2, 1, scope='output')
    print(gaussian_params.shape)

  # The mean parameter is unconstrained
  mu = gaussian_params[:, :,:, 0]
  mu = tf.reshape(mu,[-1, image_h, image_w, 1])
  # The standard deviation must be positive. Parametrize with a softplus and
  # add a small epsilon for numerical stability
  sigma = 1e-6 + tf.nn.softplus(gaussian_params[:,:,:,1])
  sigma = tf.reshape(sigma,[-1, image_h, image_w, 1])
  sigma = tf.sqrt(tf.exp(sigma))
  return mu, sigma


def train():
  # Train a Variational Autoencoder on MNIST
  # Input placeholders
  with tf.name_scope('data'):
    x = tf.placeholder(tf.float32, [None, image_h, image_w, 1])
    tf.summary.image('data', x)

  with tf.variable_scope('variational'):
    q_mu, q_sigma = inference_network(x=x, latent_dim=FLAGS.latent_dim)
    with st.value_type(st.SampleValue()):
      # The variational distribution is a Normal with mean and standard
      # deviation given by the inference network
      q_z = st.StochasticTensor(distributions.Normal(loc=q_mu, scale =q_sigma))
      print (tf.shape(q_z))

  with tf.variable_scope('model'):
    # The likelihood is Bernoulli-distributed with logits given by the
    # generative network
    p_x_given_z_mu, p_x_given_z_sigma  = generative_network(z=q_z, latent_dim=FLAGS.latent_dim)
    #p_x_given_z = distributions.Bernoulli(logits=p_x_given_z_logits)
    p_x_given_z = distributions.Normal(loc=p_x_given_z_mu, scale = p_x_given_z_sigma)
    posterior_predictive_samples = p_x_given_z.sample()
    tf.summary.image('posterior_predictive',
                     tf.cast(posterior_predictive_samples, tf.float32))

  # Take samples from the prior
  with tf.variable_scope('model', reuse=True):
    p_z = distributions.Normal(loc=np.zeros(FLAGS.latent_dim, dtype=np.float32),
                               scale=np.ones(FLAGS.latent_dim, dtype=np.float32))
    p_z_sample = p_z.sample(FLAGS.n_samples)
    p_x_given_z_mu, p_x_given_z_sigma = generative_network(z=p_z_sample,
                                            latent_dim=FLAGS.latent_dim)
    prior_predictive = distributions.Normal(loc=p_x_given_z_mu, scale = p_x_given_z_sigma)
    prior_predictive_samples = prior_predictive.sample()
    tf.summary.image('prior_predictive',
                     tf.cast(prior_predictive_samples, tf.float32))

  # Take samples from the prior with a placeholder
  with tf.variable_scope('model', reuse=True):
    z_input = tf.placeholder(tf.float32, [None, FLAGS.latent_dim])
    p_x_given_z_mu, p_x_given_z_sigma = generative_network(z=z_input,
                                            latent_dim=FLAGS.latent_dim)
    prior_predictive_inp = distributions.Normal(loc=p_x_given_z_mu, scale =p_x_given_z_sigma)
    prior_predictive_inp_sample = prior_predictive_inp.sample()


  #read data
  image = read_data(path, N)
  images_batch = next_batch(FLAGS.batch_size, image)
  #
  # Build the evidence lower bound (ELBO) or the negative loss
  kl = tf.reduce_sum(distributions.kl(q_z.distribution, p_z), 1)

  expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x), [1, 2, 3])


  elbo = tf.reduce_sum(expected_log_likelihood - kl, 0)

  optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5)

  train_op = optimizer.minimize(-elbo)

  # Merge all the summaries
  summary_op = tf.summary.merge_all()
  #
  init_op = tf.global_variables_initializer()
  #
  # # Run training
  sess = tf.InteractiveSession()
  #
  sess.run(init_op)
  #
  #
  print('Saving TensorBoard summaries and images to: %s' % FLAGS.logdir)
  train_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  for i in range(FLAGS.n_iterations):
    # Re-binarize the data at every batch; this improves results
    np_x = images_batch.eval()
    np_x = (np_x > 0.5).astype(np.float32)
    sess.run(train_op, {x: np_x})

    # Print progress and save samples every so often
    t0 = time.time()
    if i % FLAGS.print_every == 0:
      np_elbo, summary_str = sess.run([elbo, summary_op], {x: np_x})
      train_writer.add_summary(summary_str, i)
      print('Iteration: {0:d} ELBO: {1:.3f} Examples/s: {2:.3e}'.format(
          i,
          np_elbo / FLAGS.batch_size,
          FLAGS.batch_size * FLAGS.print_every / (time.time() - t0)))
      t0 = time.time()

      # Save samples
      np_posterior_samples, np_prior_samples = sess.run(
          [posterior_predictive_samples, prior_predictive_samples], {x: np_x})
      for k in range(FLAGS.n_samples):
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_posterior_predictive_%d_data.jpg' % (i, k))
        imsave(f_name, np_x[k, :, :, 0])
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_posterior_predictive_%d_sample.jpg' % (i, k))
        imsave(f_name, np_posterior_samples[k, :, :, 0])
        f_name = os.path.join(
            FLAGS.logdir, 'iter_%d_prior_predictive_%d.jpg' % (i, k))
        imsave(f_name, np_prior_samples[k, :, :, 0])

  coord.request_stop()
  coord.join(threads)



def main(_):
  if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
  tf.gfile.MakeDirs(FLAGS.logdir)
  train()


if __name__ == '__main__':
   tf.app.run()
