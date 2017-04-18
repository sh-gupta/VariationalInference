import numpy as np
from matplotlib import pyplot as plt
	
class CIFAR10:
	"""
		Provides data for CIFAR10 (binary version)
	"""
	
	def __init__(self, data_dir):
		"""
			__init__(CIFAR10, str) -> None
		"""
		
		# Read the Training data
		data = np.array([np.fromfile(data_dir + '/data_batch_' + str(i+1) + '.bin', dtype=np.uint8) \
										for i in range(5)]).flatten()	# Read training data
		data = np.reshape(data, (-1, 3073))
		self.training_labels = data[:, 0]
		self.training_images = np.transpose(np.reshape(data[:, 1:], (-1, 3, 32, 32)), axes=(0, 2, 3, 1))
		
		# Read the test data		
		data = np.fromfile(data_dir + '/test_batch.bin', dtype=np.uint8)	# Read test data
		data = np.reshape(data, (-1, 3073))
		self.test_labels = data[:, 0]
		self.test_images = np.transpose(np.reshape(data[:, 1:], (-1, 3, 32, 32)), axes=(0, 2, 3, 1))
		
		# Read label names
		self.label_texts = [line.strip() for line in open(data_dir + '/batches.meta.txt')]
		
		# Initialize starting index of new batch
		self.training_batch_start = 0
		self.test_batch_start = 0
		
		# Initialize other variables
		self.num_training_examples = np.shape(self.training_images)[0]
		self.num_test_examples = np.shape(self.test_images)[0]
	
	
	def get_data(self, batch_size, is_training=True):
		"""
			get_data(int, bool) -> (numpy.ndarray of batch of images, numpy.ndarray of batch of labels)
		"""
		
		if is_training:
			assert self.num_training_examples >= batch_size, "Batch size can not be larger than data set"
		else:
			assert self.num_test_examples >= batch_size, "Batch size can not be larger than data set"
		
		if is_training:
			images = []
			labels = []
			if (self.training_batch_start + batch_size <= self.num_training_examples):
				images = self.training_images[self.training_batch_start:self.training_batch_start + \
															batch_size, :, :, :]
				labels = self.training_labels[self.training_batch_start:self.training_batch_start + batch_size]
			else:
				examples = range(self.training_batch_start, self.num_training_examples) + \
							range(0, batch_size - self.num_training_examples + self.training_batch_start)
				images = self.training_images[examples, :, :, :]
				labels = self.training_labels[examples]
			
			self.training_batch_start = (self.training_batch_start + batch_size) % self.num_training_examples
			
			# Convert labels to one hot encoding
			temp = np.zeros((batch_size, 10))
			temp[np.arange(batch_size), labels] = 1
			labels = temp
			
			return images, labels
		else:
			images = []
			labels = []
			if (self.test_batch_start + batch_size <= self.num_test_examples):
				images = self.test_images[self.test_batch_start:self.test_batch_start + \
															batch_size, :, :, :]
				labels = self.test_labels[self.test_batch_start:self.test_batch_start + batch_size]
			else:
				examples = range(self.test_batch_start, self.num_test_examples) + \
							range(0, batch_size - self.num_test_examples + self.test_batch_start)
				images = self.test_images[examples, :, :, :]
				labels = self.test_labels[examples]
			self.test_batch_start = (self.test_batch_start + batch_size) % self.num_test_examples
			
			# Convert labels to one hot encoding
			temp = np.zeros((batch_size, 10))
			temp[np.arange(batch_size), labels] = 1
			labels = temp
			
			return images, labels
			
			
			
	def get_num_training_examples(self):
		return self.num_training_examples
		
	def get_num_test_examples(self):
		return self.num_test_examples
		
	def get_label(self, class_index):
		return self.label_texts[class_index]
		
	def show_image(self, image, label):
		plt.imshow(image)
		print self.get_label(label)
		plt.show()
