import tensorflow as tf

class VAE(object):
	"""
		Class implementing a simple Variational Autoencoder.
	"""
	
	def __init__(self, network_architecture, learning_rate=1e-4, batch_size=128):
		"""
			__init__(VAE, dict, float, int) -> None
			network_architecture: Should have the following structure
				network_architecture['input_shape'] = [Shape of the input]
				network_architecture['encoder'] = 
					[
						(layer_type, activation_fn, layer_params)	# For input layer
						(layer_type, activation_fn, layer_params)	# For hidden layer 1
						.
						.
						.
						(layer_type, activation_fn, layer_params)	# For output layer
						# The final two layers will be used as output layers for mean and log sigma
					]
				network_architecture['decoder'] = list of same type as encoder
				layer_type: Can be on of the following
					'conv': Convolutional layer, same size, single stride convolution, SAME padding
					'tconv': Transposed convolution layer, single stride, SAME padding
					'pool': pooling layer, non-overlapping mean pooling
					'fc': Fully connected layer
				activation_fn: Must be one of the valid activation functions. Eg. tf.nn.sigmoid
				layer_params: Can be one of the following
					num_units, prev_num_units: If layer_type is 'fc'
					filter_dim, channel_in, channel_out: If layer type is 'conv'
					filter_dim, channel_in, channel_out, output_shape, stride: If layer type is 'tconv'
						output_shape: Parameter for conv2d_transpose function of tensorflow
						stride: Parameter for conv2d_transpose function of tensorflow
					pool_dim: If layer type is 'pool'					
				network_architecture['first_roll_shape'] = (height, width, num_channels)	# Roll shape in decoder for first deconv layer
			
			learning_rate: The learning rate for the network
			batch_size: Batch size for input
		"""
		
		# Initialize the weights of the network
		self.weights = self.setup_network(network_architecture)
		
		# Prepare the input
		self.input = tf.placeholder(tf.float32, [batch_size] + network_architecture['input_shape'])
		
		# Run the encoder
		self.mus, self.log_sigmas = self.encode(self.input, network_architecture)
		
		# Reparameterization, draw samples
		num_hidden = network_architecture['encoder'][-1][2]
		eps = tf.random_normal((batch_size, num_hidden), 0.0, 1.0)
		self.z = tf.add(self.mus, tf.multiply(tf.sqrt(tf.exp(self.log_sigmas)), eps))
		
		# Run the decoder
		self.output = self.decode(self.z, network_architecture)
		
		# Calculate the cost
		unrolled_input = tf.reshape(self.input, (batch_size, -1))
		unrolled_output = tf.reshape(self.output, (batch_size, -1))
		reconstruction_cost = -tf.reduce_sum(unrolled_input * tf.log(1e-10 + unrolled_output) + \
											(1-unrolled_input)*tf.log(1e-10 + 1 - unrolled_output), 1)
		variational_cost = -0.5 * tf.reduce_sum(1.0 + self.log_sigmas - \
												tf.square(self.mus) - \
												tf.exp(self.log_sigmas), 1)
		self.cost =  tf.reduce_mean(reconstruction_cost + variational_cost)
					
		# Create the optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
		
		# Prepare the session
		self.sess = tf.InteractiveSession()
		self.sess.run(tf.global_variables_initializer())
		
		
	
	
	def _get_weight(self, shape):
		return tf.Variable(tf.truncated_normal(shape, stddev=0.01))
	
	def _get_bias(self, shape):
		return tf.Variable(tf.zeros(shape))
		
	
	def setup_network(self, arch):
		"""
			setup_network(VAE, dict) -> dict
			arch: Same definition as network_architecture in __init__
			Returns the weights corresponding to the provided architecture
		"""
		
		weights = dict()
		weights['encoder'] = dict()
		weights['decoder'] = dict()
		
		# Intialize encoder weights
		count = 0
		for layer in arch['encoder']:
			layer_type = layer[0]
			
			if count == 0:
				count += 1
				continue	# Skip the first layer
			
			if layer_type == 'conv':
				filter_dim = layer[2]
				channel_in = layer[3]
				channel_out = layer[4]
				weights['encoder']['w' + str(count)] = self._get_weight((filter_dim, filter_dim, channel_in, channel_out))
				weights['encoder']['b' + str(count)] = self._get_bias((channel_out))
			elif layer_type == 'fc':
				num_units = layer[2]
				prev_num_units = layer[3]
				weights['encoder']['w' + str(count)] = self._get_weight((prev_num_units, num_units))
				weights['encoder']['b' + str(count)] = self._get_bias((num_units))
			else:
				weights['encoder']['w' + str(count)]  = []
				weights['encoder']['b' + str(count)]  = []
			count += 1
			
		# Intialize decoder weights
		count = 0
		for layer in arch['decoder']:
			layer_type = layer[0]
			
			if layer_type == 'tconv':
				filter_dim = layer[2]
				channel_in = layer[3]
				channel_out = layer[4]
				weights['decoder']['w' + str(count)] = self._get_weight((filter_dim, filter_dim, channel_out, channel_in))
				weights['decoder']['b' + str(count)] = self._get_bias((channel_out))
			elif layer_type == 'fc':
				num_units = layer[2]
				prev_num_units = layer[3]
				weights['decoder']['w' + str(count)] = self._get_weight((prev_num_units, num_units))
				weights['decoder']['b' + str(count)] = self._get_bias((num_units))
			count += 1
		
		return weights
		
		
	
		
	def encode(self, x, arch):
		"""
			encode(VAE, tensor, dict) -> tensor for mean, tensor for variance
			Runs the encoder for VAE
			x: Input tensor (batch_size, input_size)
			arch: Same as network_architecture in __init__
			Returns mean and log sigma
		"""
		n = len(arch['encoder'])
		layers = arch['encoder']	# All layers of encoder
		layer_input = x
		
		unroll = True	# Unroll inputs at first fully connected layer
		for i in range(1, n-2):
			
			# Get the weights for the current layer		
			w = self.weights['encoder']['w' + str(i)]
			b = self.weights['encoder']['b' + str(i)]
			
			# Get the activation function to be used
			activation = layers[i][1]
			
			if layers[i][0] == 'fc':
				if unroll:
					unroll = False
					layer_input = tf.reshape(layer_input, (-1, layers[i][3]))
				layer_input = activation(tf.add(tf.matmul(layer_input, w), b))
			elif layers[i][0] == 'conv':
				layer_input = activation(tf.nn.bias_add(tf.nn.conv2d(layer_input, w, [1, 1, 1, 1], padding='SAME'), b))
			elif layers[i][0] == 'pool':
				pool_dim = layers[i][2]
				layer_input = tf.nn.avg_pool(layer_input, ksize=[1, pool_dim, pool_dim, 1], strides=[1, pool_dim, pool_dim, 1], padding='SAME')
				
		# Calculate the final mean and log sigma (assumed to be fully connected)
		if unroll:
			unroll = False
			layer_input = tf.reshape(layer_input, (-1, layers[n-2][3]))
		
		# Calculate mean
		w = self.weights['encoder']['w' + str(n-2)]
		b = self.weights['encoder']['b' + str(n-2)]
		activation = layers[n-2][1]
		mu = activation(tf.add(tf.matmul(layer_input, w), b))
	
		# Calculate log sigma
		w = self.weights['encoder']['w' + str(n-1)]
		b = self.weights['encoder']['b' + str(n-1)]
		activation = layers[n-1][1]
		log_sigma = activation(tf.add(tf.matmul(layer_input, w), b))
		
		return (mu, log_sigma)
		
		
		
		
	def decode(self, z, arch):
		"""
			decode(VAE, tensor, dict) -> tensor for reconstructed images
			Runs the decoder for VAE
			z: Hidden representation tensor
			arch: Same as network_architecture in __init__
			Returns the reconstructed images
		"""
		n = len(arch['decoder'])
		layers = arch['decoder']	# All layers of encoder
		layer_input = z
		
		roll = True	# Roll inputs at first transposed convolution layer
		for i in range(n):
			# Get the weights for the current layer		
			w = self.weights['decoder']['w' + str(i)]
			b = self.weights['decoder']['b' + str(i)]
			
			# Get the activation function to be used
			activation = layers[i][1]
			
			if layers[i][0] == 'fc':
				layer_input = activation(tf.add(tf.matmul(layer_input, w), b))
			elif layers[i][0] == 'tconv':
				if roll:
					roll = False
					shape = arch['first_roll_shape']
					shape = (-1,) + shape
					layer_input = tf.reshape(layer_input, shape)
				output_shape = layers[i][5]
				stride = layers[i][6]
				layer_input = activation(tf.nn.bias_add(tf.nn.conv2d_transpose(layer_input, w, output_shape, stride, padding='SAME'), b))
								
		return layer_input
		
		
		
	
	def train(self, inputs):
		"""
			train(VAE, ndarray) -> float
			Run the training on the provided batch
			inputs: The batch of input (batch_size, input_size)
			Returns the cost after training
		"""
		_, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.input: inputs})
		return cost
		
		
	
	def get_hidden_representation(self, inputs):
		"""
			get_hidden_representation(VAE, ndarray) -> ndarray
			Returns the hidden representations for the specified inputs
		"""
		return self.sess.run(self.mus, feed_dict={self.input: inputs})
		
	
	
	def generate_image(self, z):
		"""
			generate_image(VAE, ndarray) -> ndarray
			Generates a batch of images using VAE for the given hidden representations
		"""
		return self.sess.run(self.output, feed_dict = {self.z: z})
