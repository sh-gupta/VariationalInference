import numpy as np
from gm_generate import *
import matplotlib.pyplot as plt

class EM(object):
	"""
		Implements the Expectation-Maximization algorithm for 1D Mixture of Gaussians
	"""
	
	def __init__(self, x, M, N):
		"""
			__init__(EM, ndarray, int, int) -> None
			x: ndarray containing data points x_i's
			M: Number of mixture components
			N: Number of examples
		"""
		self.x = x
		self.M = M
		self.N = N
		
		# Initialise the mu and sigma
		self.mu = np.asarray([np.random.normal() for i in range(M)])
		self.sigma = np.asarray([np.random.uniform() for i in range(M)])
		
		# Initialize p(z_i|x_i)
		self.pi = np.random.uniform(size=(self.N, self.M))
		self.pi = self.pi / np.reshape(np.sum(self.pi, axis=1), (self.N, 1))
	

	
	def train(self):
		"""
			train(EM) -> ndarray, ndarray, ndarray
			Runs one iteration of EM
			Returns:
				z_i: The cluster corresponding to x_i
				mu: Cluster centers
				sigma: Cluster variances
		"""	
		# Calculate mu and sigma
		x_stacked = np.stack([self.x]*self.M, axis=1)
		self.mu = np.sum(np.multiply(x_stacked, self.pi), axis=0) / np.sum(self.pi, axis=0)
		self.sigma = np.sum(np.multiply((x_stacked - self.mu)**2, self.pi), axis=0) / np.sum(self.pi, axis=0)
	
		# Calculate p(x_i|z_i)
		px_given_z = self.calculate_px_given_z()
		
		# Recalculate pi
		self.pi = np.multiply(px_given_z, self.pi)
		self.pi = self.pi / np.reshape(np.sum(self.pi, axis=1), (self.N, 1))
		
		# Calculate cost
		cost = np.sum(np.log(np.sum(np.multiply(px_given_z, self.pi), axis=0)))
	
		return (self.pi, self.mu, self.sigma, cost)
	
	
	
	def calculate_px_given_z(self):
		"""
			calculate_px_given_z(EM) -> ndarray
			Returns:
				px_given_z: px_given_z[i, j] = p(x_i|z_i = j)
		"""
		x_stacked = np.stack([self.x]*self.M, axis=1)
		mu_stacked = np.stack([self.mu]*self.N, axis=0)
		sigma_stacked = np.stack([self.sigma]*self.N, axis=0)
		denom = 1.0/np.sqrt(2*np.pi*self.sigma)
		denom_stacked =  np.stack([denom]*self.N, axis=0)
		px_given_z = (x_stacked - mu_stacked)**2 / (2*sigma_stacked)
		px_given_z = np.exp(-px_given_z) / denom_stacked
		
		return px_given_z
		
		


def run_em(data_mu, data_sigma, num_iter=100, N=1000, M=3):
	"""
		run_em(list, list, int, int, int) -> None
		Runs the demo shown in the notebook for different settings of hyperparameters.
	"""
	# Generate the data
	pz = [1.0/M]*M
	mu = [np.asarray([x]) for x in data_mu]
	sigma = [np.asarray([x]).reshape(1, 1) for x in data_sigma]
	gmm = GMM(pz=pz, muks=mu, sigmaks=sigma, k=M, d=1)
	x, _ = gmm.generate_points(n=N)
	x = x.reshape((-1,))

	# Run the EM algorithm
	em = EM(x, M, N)
	costs = []
	for i in range(num_iter):
		pi, mu, sigma, cost = em.train()
		costs.append(cost)
	
	# Print results
	print 'Mu:', mu
	print 'Sigma:', sigma
	
	# Plot cost vs iterations
	plt.plot(costs)
	plt.title('Iteration vs Cost')
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.show()
	
	# Show the histogram, true distribution and estimated distribution
	plt.hist(x, normed=True, color='#cccccc')
	def estimated_dist(x):
		N = np.shape(x)[0]
		x_stacked = np.stack([x]*M, axis=1)
		mu_stacked = np.stack([mu]*N, axis=0)
		sigma_stacked = np.stack([sigma]*N, axis=0)
		x_given_z = np.exp(-0.5 * np.power((x_stacked - mu_stacked) / sigma_stacked, 2))
		x_given_z = x_given_z * np.stack([1.0 / np.sqrt(2 * np.pi * sigma)]*N, axis=0)
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
