import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from gm_generate import *

class IntermediateExample1(object):
	"""
		Class implementing variational inference for 1D gaussian data by using monte-carlo approximation to ELBO
	"""
	
	def __init__(self, mu_p=0.0, sigma_p=1.0, mu_var_p=0.0, sigma_var_p=1.0, learning_rate=1e-4, K=1, N=100):
		"""
			__init__(IntermediateExample1, float, float, float, float, float, int, int) -> None
			mu_p: Hyperparamer for p(mu)
			sigma_p: Hyperparameter for p(mu)
			mu_var_p: Hyperparameter for p(sigma)
			sigma_var_p: Hyperparameter for p(sigma)
			learning_rate: Learning rate for optimizer
			K: Number of samples used for Monte-Carlo approximation of ELBO
			N: Number of examples per batch
		"""
		self.N = N
		self.x = tf.placeholder(tf.float32, (self.N,))
		self.mu_p = mu_p
		self.sigma_p = sigma_p
		self.mu_var_p = mu_var_p
		self.sigma_var_p = sigma_var_p
		self.learning_rate = learning_rate
		self.K = K
		
		# Initialize the variables
		self.sigma = tf.Variable(random.random())
		self.mu = tf.Variable(random.random())
		self.sigma_var = tf.Variable(random.random())
		self.mu_var = tf.Variable(random.random())
		
		# Generate K samples of z
		eps_samples = tf.random_normal(shape=(K,))	# Reparameterisation trick for normal
		mu_samples = self.mu + eps_samples*self.sigma**2
		eps_samples2 = tf.random_uniform(shape=(K,))	# Reparameterisation trick for exponential
		sigma_samples = (self.mu_var + eps_samples2*self.sigma_var**2) ** 2
		
		
		# Approximate ELBO (without constant term)
		self.ELBO = tf.reduce_mean(-(N/2.0) * tf.log(sigma_samples) - \
				0.5 * tf.reduce_sum((self.x - tf.stack([mu_samples]*N, axis=1))**2, axis=1) / sigma_samples - \
				0.5 * (mu_samples - self.mu_p)**2 / self.sigma_p**2 - \
				0.5 * (sigma_samples - self.mu_var_p)**2 / self.sigma_var_p**2 - \
				0.5 * tf.log(self.sigma**2) + \
				0.5 * (mu_samples - self.mu)**2 / self.sigma**2 + \
				0.5 * tf.log(self.sigma_var**2) + \
				0.5 * (sigma_samples - self.mu_var)**2 / self.sigma_var**2) 
		
		self.ELBO = -self.ELBO
		
		# Maximize ELBO
		self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.ELBO)
		
		# Set up the session
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())
		
	
	def train_step(self, x):
		"""
			train_step(IntermediateExample1, ndarray) -> (float, float, float, float, float)
			x: Input data, vector of real numbers
			
			Returns:
				cost: -ELBO after a step of optimization
				mu,	sigma, mu_var, sigma_var
		"""
		_, cost = self.sess.run([self.train, self.ELBO], feed_dict={self.x: x})
		mu, sigma, mu_var, sigma_var = self.sess.run([self.mu, self.sigma, self.mu_var, self.sigma_var])
		return (-cost, mu, sigma**2, mu_var, sigma_var**2)
		


def run_intermediate_example_1(num_examples=100, data_mu=0.0, data_sigma=1.0, mu_p=0.0, sigma_p=1.0, mu_var_p=0.0, sigma_var_p=1.0, learning_rate=1e-4, num_iter=1000, K=1, N=100):
	"""
		run_intermediate_example_1(int, float, float, float, float, float, float, float, float, int, int) -> None
		Runs the demo shown in the notebook for different settings of hyperparameters.
	"""
	# Generate the data
	pz = [1]
	mu = [np.asarray([data_mu])]
	sigma = [np.asarray([data_sigma]).reshape((1, 1))]
	gmm = GMM(pz=pz, muks=mu, sigmaks=sigma, k=1, d=1)
	x, _ = gmm.generate_points(n=num_examples)
	x = x.reshape((-1,))

	# Compute variational inference estimate for the parameters
	costs = []
	example = IntermediateExample1(mu_p, sigma_p, mu_var_p, sigma_var_p, learning_rate, K, N)
	for i in range(num_iter):
		cost, mu_ex, sigma_ex, mu_var, sigma_var = example.train_step(x)
		costs.append(cost)

	# Compute mu_expected, sigma_expected
	mu_expected = mu_ex # Expected value of mu using q_1
	sigma_expected = mu_var**2 # Expected value of sigma using q_2

	# Print results
	print 'Optimal m:', mu_ex
	print 'Optimal s^2:', sigma_ex
	print 'Optimal m var:', mu_var
	print 'Optimal s^2 var:', sigma_var
	print 'Expected Value for mu:', mu_expected
	print 'Optimal Value for sigma^2:', sigma_expected
		
	# Plot cost vs iterations
	plt.plot(costs)
	plt.title('Iteration vs ELBO')
	plt.xlabel('Iterations')
	plt.ylabel('ELBO')
	plt.show()

	# Show the histogram, true distribution and estimated distribution
	plt.hist(x, normed=True, color='#cccccc')
	def true_dist(x):
	   return (1.0/np.sqrt(2*np.pi*sigma[0])*np.exp(-0.5*((x-mu[0])/sigma[0])**2))
	def estimated_dist(x):
		return (1.0/np.sqrt(2*np.pi*sigma_expected)*np.exp(-0.5*((x-mu_expected)/sigma_expected)**2))
	x_axis = np.arange(np.min(x)-0.5, np.max(x)+0.5, 0.01)
	plt.plot(x_axis, true_dist(x_axis).reshape((-1,)), 'ro', label='True Distribution')
	plt.plot(x_axis, estimated_dist(x_axis).reshape((-1,)), 'co', label='Estimated Distribution')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.show()
	
		

"""		
if __name__ == '__main__':
	ex = IntermediateExample1(learning_rate=0.05)
	for i in range(1000):
		cost, mu, sigma, alpha, beta = ex.train_step(np.asarray([1.0, 2.0, 3.0]))
		print (cost, mu, sigma, alpha, beta)
"""
