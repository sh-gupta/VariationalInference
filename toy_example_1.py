import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from gm_generate import *

class ToyExample1(object):
	"""
		Class implementing variational inference for 1D gaussian data
	"""
	
	def __init__(self, mu_p=0.0, sigma_p=1.0, alpha_p=3.5, beta_p=0.5, learning_rate=1e-4):
		"""
			__init__(ToyExample1, float, float, float, float, float) -> None
			mu_p: Hyperparamer for p(mu)
			sigma_p: Hyperparameter for p(mu)
			alpha_p: Hyperparameter for p(sigma)
			beta_p: Hyperparameter for p(sigma)
			learning_rate: Learning rate for optimizer
		"""
		self.x = tf.placeholder(tf.float32, (None,))
		self.mu_p = mu_p
		self.sigma_p = sigma_p
		self.alpha_p = alpha_p
		self.beta_p = beta_p
		self.learning_rate = learning_rate
		
		# Initialize the variables
		self.sigma = tf.Variable(random.random())
		self.mu = tf.Variable(random.random())
		self.alpha = tf.Variable(1.0+random.random())
		self.beta = tf.Variable(1.0+random.random())
		N = tf.cast(tf.shape(self.x), tf.float32)
		
		# Calculate ELBO (without constant term)
		gamma_ratio = tf.exp(tf.lgamma(self.alpha**2 - 1))
		gamma_ratio /= (tf.exp(tf.lgamma(self.alpha**2))*self.beta**2)
		pi = tf.constant(np.pi)
		e = tf.constant(np.e)				
		self.ELBO = -(N/2.0)*(tf.digamma(self.alpha**2) + tf.log(self.beta**2)) + \
				gamma_ratio*(self.mu*tf.reduce_sum(self.x) - \
				(N/2.0)*(self.mu**2 + self.sigma**2) - \
				0.5*tf.reduce_sum(self.x ** 2)) + \
				(self.alpha_p - 1) * (tf.digamma(self.alpha**2) + tf.log(self.beta**2)) - \
				(self.alpha**2*self.beta**2)/self.beta_p + \
				self.alpha**2 + tf.log(self.beta**2) + tf.lgamma(self.alpha**2) + \
				(1-self.alpha**2)*tf.digamma(self.alpha**2) + \
				0.5 * tf.log(2.0 * self.sigma**2 * pi * e)
		self.ELBO = -self.ELBO
		
		# Maximize ELBO
		self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.ELBO)
		
		# Set up the session
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())
		
	
	def train_step(self, x):
		"""
			train_step(ToyExample1, ndarray) -> (float, float, float, float, float)
			x: Input data, vector of real numbers
			
			Returns:
				cost: -ELBO after a step of optimization
				mu,	sigma, alpha, beta
		"""
		_, cost = self.sess.run([self.train, self.ELBO], feed_dict={self.x: x})
		mu, sigma, alpha, beta = self.sess.run([self.mu, self.sigma, self.alpha, self.beta])
		return (-cost, mu, sigma**2, alpha**2, beta**2)
		


def run_toy_example_1(num_examples=100, data_mu=0.0, data_sigma=1.0, mu_p=0.0, sigma_p=1.0, alpha_p=3.5, beta_p=0.5, learning_rate=1e-4, num_iter=1000):
	"""
		run_toy_example_1(int, float, float, float, float, float, float, float, float) -> None
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
	toy_example = ToyExample1(mu_p, sigma_p, alpha_p, beta_p, learning_rate)
	for i in range(num_iter):
		cost, mu_ex, sigma_ex, alpha, beta = toy_example.train_step(x)
		costs.append(cost)

	# Compute mu_expected, sigma_expected
	mu_expected = mu_ex # Expected value of mu using q_1 (normal)
	sigma_expected = beta*alpha # Expected value of sigma using q_2 (gamma)

	# Print results
	print 'Optimal m:', mu_ex
	print 'Optimal s^2:', sigma_ex
	print 'Optimal alpha:', alpha
	print 'Optimal beta:', beta
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
	ex = ToyExample1(learning_rate=0.05)
	for i in range(1000):
		cost, mu, sigma, alpha, beta = ex.train_step(np.asarray([1.0, 2.0, 3.0]))
		print (cost, mu, sigma, alpha, beta)
"""
