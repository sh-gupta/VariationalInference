import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from gm_generate import *

class IntermediateExample2(object):
	"""
		Class implementing variational inference for 1D gaussian mixture by using monte-carlo approximation to ELBO
	"""
	
	def __init__(self, M, N, K=1, sigma_mu=1.0, sigma_sigma=1.0, learning_rate=1e-4):
		"""
			__init__(IntermediateExample2, float, float, int, int) -> None
			sigma_mu: Prior for mu -> P(mu) = Normal(0, sigma_mu*I)
			sigma_sigma: Prior for sigma -> P(sigma) = logNormal(0, sigma_sigma*I)
			M: Number of gaussian mixtures
			N: Number of examples
			K: Number of samples to use for estimating expectation 
			Assumptions:
				1. Uniform prior for P(z)
				2. IID data points
				3. Mean and variance of different gaussian components are independent
		"""
		self.M = M
		self.K = K
		self.N = N
		self.x = tf.placeholder(tf.float32, (self.N,))
		self.sigma_mu = sigma_mu
		self.sigma_sigma = sigma_sigma
		self.learning_rate = learning_rate
		
		# Initialize the variational parameters for mu
		self.m_mu = tf.Variable(tf.random_normal(shape=(M,)))
		self.s_mu = tf.Variable(tf.random_uniform(shape=(M,)))
		
		# Initialize the variational parameters for sigma
		self.m_sigma = tf.Variable(tf.random_normal(shape=(M,)))
		self.s_sigma = tf.Variable(tf.random_uniform(shape=(M,)))
		
		# Initialize the variational parameters for cluster assignment
		self.pi = tf.Variable(tf.random_uniform(shape=(N, M)))
		
		# Make pi a valid probability distribution
		self.pi_prob = self.pi**2 / tf.stack([tf.reduce_sum(self.pi**2, axis=1)]*M, axis=1)
		
		# Generate K samples of mu
		eps_samples = tf.random_normal(shape=(K, M))	# Reparameterisation trick for normal
		mu_samples = tf.stack([self.m_mu]*K, axis=0) + eps_samples*tf.stack([self.s_mu**2]*K, axis=0)
		
		# Generate K samples of sigma
		eps_samples2 = tf.random_normal(shape=(K, M))	# Reparameterisation trick for lognormal
		sigma_samples = tf.stack([self.m_sigma]*K, axis=0) + eps_samples2*tf.stack([self.s_sigma**2]*K, axis=0)
		sigma_samples = tf.exp(sigma_samples)	# exp(normal) = log normal
		
		# Approximate ELBO (without constant term)
		x_stacked = tf.stack([tf.stack([self.x]*M, axis=1)]*K, axis=0)
		mu_stacked = tf.transpose(tf.stack([mu_samples]*N, axis=0), [1, 0, 2])
		sigma_stacked = tf.transpose(tf.stack([sigma_samples]*N, axis=0), [1, 0, 2])
		pi_stacked = tf.stack([self.pi_prob]*K, axis=0)
		temp = (x_stacked - mu_stacked)**2 / sigma_stacked
		eps = 1e-10	# A small number to be added in to log to avoid NaN
		
		self.ELBO = tf.reduce_mean(-0.5 * (1.0/self.sigma_mu) * tf.reduce_sum(mu_samples**2, axis=1) - \
				0.5 * (1.0/self.sigma_sigma) * tf.reduce_sum(tf.log(eps + sigma_samples)**2, axis=1) - \
				0.5 * tf.reduce_sum(tf.matmul(self.pi_prob, tf.transpose(tf.log(eps + sigma_samples))), axis=0) - \
				0.5 * tf.reduce_sum(tf.multiply(pi_stacked, temp), axis=[1, 2]) + \
				0.5 * tf.reduce_sum(((mu_samples-tf.stack([self.m_mu]*K, axis=0))**2)/tf.stack([self.s_mu**2]*K, axis=0), axis=1) + \
				0.5 * tf.reduce_sum(((tf.log(eps + sigma_samples)-tf.stack([self.m_sigma]*K, axis=0))**2)/tf.stack([self.s_sigma**2]*K, axis=0), axis=1)				
				) + \
				0.5 * tf.reduce_sum(tf.log(eps + self.s_sigma**2)) + \
				0.5 * tf.reduce_sum(tf.log(eps + self.s_mu**2)) - \
				tf.reduce_sum(tf.multiply(self.pi_prob, tf.log(eps + self.pi_prob)))
		
		self.ELBO = -self.ELBO
		
		# Maximize ELBO
		self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.ELBO)
		
		# Set up the session
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())
		
	
	def train_step(self, x):
		"""
			train_step(IntermediateExample2, ndarray) -> (float, ndarray, ndarray, ndarray, ndarray, ndarray)
			x: Input data, vector of real numbers
			
			Returns:
				cost: -ELBO after a step of optimization
				m_mu, m_sigma, s_mu, s_sigma, pi
		"""
		_, cost = self.sess.run([self.train, self.ELBO], feed_dict={self.x: x})		
		m_mu, m_sigma, s_mu, s_sigma, pi = self.sess.run([self.m_mu, self.m_sigma, self.s_mu, self.s_sigma, self.pi_prob])
		return (-cost, m_mu, m_sigma, s_mu**2, s_sigma**2, pi)
		


def run_intermediate_example_2(data_mu, data_sigma, sigma_mu=1.0, sigma_sigma=1.0, learning_rate=1e-4, num_iter=1000, K=1, N=100, M=2):
	"""
		run_intermediate_example_2(list, list, float, float, float, int, int, int, int) -> None
		Runs the demo shown in the notebook for different settings of hyperparameters.
	"""
	# Generate the data
	pz = [1.0/M]*M
	mu = [np.asarray([x]) for x in data_mu]
	sigma = [np.asarray([x]).reshape(1, 1) for x in data_sigma]
	gmm = GMM(pz=pz, muks=mu, sigmaks=sigma, k=M, d=1)
	x, _ = gmm.generate_points(n=N)
	x = x.reshape((-1,))

	# Compute variational inference estimate for the parameters
	costs = []
	example = IntermediateExample2(M, N, K, sigma_mu, sigma_sigma, learning_rate)
	for i in range(num_iter):
		cost, m_mu, m_sigma, s_mu, s_sigma, pi = example.train_step(x)
		costs.append(cost)

	# Compute mu_expected, sigma_expected
	mu_expected = m_mu # Expected value of mu using q_1
	sigma_expected = np.exp(m_sigma + 0.5*s_sigma) # Expected value of sigma using q_2
	
	# Print results
	print 'Mean for mu:', m_mu
	print 'Mean for sigma:', m_sigma
	print 'Variance for mu:', s_mu
	print 'Variance for sigma:', s_sigma
	print 'Optimal mu:', mu_expected
	print 'Optimal s^2:', sigma_expected
	
	# Plot cost vs iterations
	plt.plot(costs)
	plt.title('Iteration vs ELBO')
	plt.xlabel('Iterations')
	plt.ylabel('ELBO')
	plt.show()
	
	# Show the histogram, true distribution and estimated distribution
	plt.hist(x, normed=True, color='#cccccc')
	def estimated_dist(x):
		N = np.shape(x)[0]
		x_stacked = np.stack([x]*M, axis=1)
		mu_stacked = np.stack([mu_expected]*N, axis=0)
		sigma_stacked = np.stack([sigma_expected]*N, axis=0)
		x_given_z = np.exp(-0.5 * np.power((x_stacked - mu_stacked) / sigma_stacked, 2))
		x_given_z = x_given_z * np.stack([1.0 / np.sqrt(2 * np.pi * sigma_expected)]*N, axis=0)
		pz = np.mean(pi / np.stack([np.sum(pi, axis=1)]*M, axis=1), axis=0)
		px = np.multiply(x_given_z, np.stack([pz]*N, axis=0))
		return np.sum(px, axis=1)
	def true_dist(x):
		N = np.shape(x)[0]
		x_stacked = np.stack([x]*M, axis=1)
		mu_stacked = np.stack([np.asarray(data_mu)]*N, axis=0)
		sigma_stacked = np.stack([np.asarray(data_sigma)]*N, axis=0)
		x_given_z = np.exp(-0.5 * np.power((x_stacked - mu_stacked) / sigma_stacked, 2))
		x_given_z = x_given_z * np.stack([1.0 / np.sqrt(2 * np.pi * np.asarray(data_sigma))]*N, axis=0)
		prz = np.asarray(pz)
		px = np.multiply(x_given_z, np.stack([prz]*N, axis=0))
		return np.sum(px, axis=1)
		
	x_axis = np.arange(np.min(x)-0.5, np.max(x)+0.5, 0.01)
	plt.plot(x_axis, true_dist(x_axis), 'ro', label='True Distribution')
	plt.plot(x_axis, estimated_dist(x_axis), 'co', label='Estimated Distribution')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.show()
		

"""		
if __name__ == '__main__':
	ex = IntermediateExample1(learning_rate=0.05)
	for i in range(1000):
		cost, mu, sigma, alpha, beta = ex.train_step(np.asarray([1.0, 2.0, 3.0]))
		print (cost, mu, sigma, alpha, beta)
"""
