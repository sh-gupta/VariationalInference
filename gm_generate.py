import random
import numpy as np

class GMM(object):
	"""
		Generates data points from gaussian mixture model
	"""
	
	def __init__(self, pz, muks, sigmaks, k, d, seed=None):
		"""
			__init__(GMM, list, list, list, int, int, float) -> None
			k: Number of mixture components
			d: Dimensionality of data
			pz: List containing P(z = k) for all k's
			uks: List containing (mean) u_k for all k's
			sigmaks: List containing covariance matrix sigma_k for all k's
			seed: Random seed to use
		"""
		self.k = k
		self.d = d
		self.pz = pz
		
		# Initialize the random number generator
		state = random.getstate()
		random.seed(seed)
		self.random_state = random.getstate()
		random.setstate(state)	# Set original state
		
		# Sanity Checks
		assert len(muks) == k, "Need to specify mean value for all the clusters"
		assert len(sigmaks) == k, "Need to specify covariance value for all the clusters"
		for u in muks:
			assert np.shape(u) == (d,), "Incorrect dimension for mean"
		for sigma in sigmaks:
			assert np.shape(sigma) == (d, d), "Incorrect dimension for sigma"
		
		
		# Initialize the mean and variance
		self.uks = muks
		self.sigmaks = sigmaks
		
	
	def generate_points(self, n):
		"""
			generate_points(GMM, int) -> (ndarray, ndarray)
			n: Number of points to generate
			Returns (n, d) ndarray of generated points
		"""
		# Set up the random number generator
		state = random.getstate()
		random.setstate(self.random_state)
		
		# Initialize the data
		data = np.zeros((n, self.d + 1))
		
		# Generate data points
		classes = np.random.multinomial(n, self.pz)
		class_idx = 0
		example_idx = 0
		for k in classes:
			data[example_idx:example_idx+k, 0:self.d] = np.random.multivariate_normal(self.uks[class_idx], \
														self.sigmaks[class_idx], (k))
			data[example_idx:example_idx+k, self.d] = class_idx * np.ones((k,))
			example_idx += k
			class_idx += 1
		
		# Shuffle the data randomly
		np.random.shuffle(data)
		x = data[:, 0:self.d]
		z = data[:, self.d]
		
		# Restore state of the random number generator
		random.setstate(state)
		
		return (x, z)
