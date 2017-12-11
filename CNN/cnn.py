import tensorflow as tf
import numpy as np

class TextCNN(object):
	"""
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    #sequence_length = length of sentences
    # num_classes - number of classes in output layer - 5
    #vocabulary_size - size of vocabulary_sizeulary
    # embedding_size - dimensinality of our embeddings
    #filter_sizes - number of words convolutional filter covers at a time
    # num_filters - number of filters per filter size
	def __init__(self,sequence_length,num_classes, vocab_size, embedding_size,filter_sizes,num_filters,l2_reg_lambda=0.0):
		self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name="input_x")
		self.input_y = tf.placeholder(tf.float32,[None,num_classes],name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

		# tf.placeholder creates variables in the network
    	#second paramter is dimension.None is passed as for batch_size dimension, so that any dimension could be used
    	#probability of keeping neuron in dropout layer is also given.dropout is enabled only during training not testing.

    	# keep track of l2 regularization loss
		l2_loss = tf.constant(0.0)

    	# the first layer is the embeddig layer which maps vocabulary_sizeulary words into low dimensional vectors.It is essentially a 
    	# lookup table learnt from data
		with tf.device('/cpu:0'),tf.name_scope("embedding"):
			W=tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name="W")
			self.embedded_chars = tf.nn.embedding_lookup(W,self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,-1)
    	# by default tensorflow try to put operation on the gpu if available,but embedding implementation currently 
    	# does not have GPU support.so we are forcing it on CPU

    	# w is embedding matrix we learn during training.tf.nn.embedding_lookup creates actual embedding operation.
    	# result of embedding operation is [None, sequence_length, embedding_size]

    	# tensorflow convolution con2d operator expects a 4 dimensional tensor dimension with dimensions as
    	# [batch, width, height, channel].so we manually added channel dimension.


    	# we use filters of different sizes.Because each convoltuion produces tensors of different shapes, we need to
    	# iterate theough them , create a layer for each of them and then merge the results into one big feature vector.

		pooled_outputs = []
		for i,filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# convolution layer
				filter_shape = [filter_size,embedding_size,1,num_filters]
				# w is our filter matrix
				W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
				b = tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b")
				conv = tf.nn.conv2d(self.embedded_chars_expanded,W,strides=[1,1,1,1],padding="VALID",name="conv")

				# apply non linearity to convolution output
				h = tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")

				# each filter strides over the whole embedding, but varies in how many words it covers
				# "VALID" padding means that we slide the filter over our sentence without padding the edges,performing
				# a narrow convolution  that gives us shape [1,sequence_length-filter_size+1,1,1

				# max-pooling over the outputs
				pooled = tf.nn.max_pool(h,ksize=[1,sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding="VALID",name="pool")
				
				# performing a max-pooling over the output of a specific filter size leaves us with a tensor of shape
				# [batch_size, 1,1,num_filters].This is a feature vector where the last dimension corresponds to our
				# features

				pooled_outputs.append(pooled)

		# combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(3,pooled_outputs)
		# once we have all the pooled output tensors from each filter size we combine them into one long feature vector
		# of shape [batch_size, num_filters_total]
		# using reshape , we can flatten the dimension
		self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

		# dropout layer stochastically disables a fraction of its neurons.this prevents neurons from  co-adapting and
		# forces them to learn individual useful features.fraction of neurons we keep enabled is defined by the 
		# dropout_keep_prob input to our network.it is generally 0.5 for training and 1(disable droupout) when testing

		# Adding Dropout
		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)

		# scores and predictions
		# using fature vector from the max-pooling with dropout applied we can generate predictions by matrix multiplication
		# and picking class with highest score.we can also apply softmax function

		with tf.name_scope("output"):
			W = tf.get_variable("W",shape=[num_filters_total, num_classes],initializer=tf.contrib.layers.xavier_initializer())
			b=tf.Variable(tf.constant(0.1,shape=[num_classes]),name="b")

			l2_loss+=tf.nn.l2_loss(W)
			l2_loss+=tf.nn.l2_loss(b)

			# xw_plus_b perfroms Wx+b
			self.scores = tf.nn.xw_plus_b(self.h_drop,W,b,name="scores")
			self.predictions = tf.argmax(self.scores,1,name="predictions")

		# using our scores,we can define loss function
		# cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(self.scores,self.input_y)
			self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss

		# accuracy
		with tf.name_scope("accuracy"):
			correct_predictions=tf.equal(self.predictions,tf.argmax(self.input_y,1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"),name="accuracy")


