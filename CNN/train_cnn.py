import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from cnn import *
from tensorflow.contrib import learn

# parameters

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



# Data Preparatopn
# ==================================================

root = "/home/phaneendra/Documents/TDBMS/project"
text_path = os.path.join(root,"sentence_labels.txt")
train_path=os.path.join(root,"train.txt")
valid_path=os.path.join(root,"valid.txt")
test_path=os.path.join(root,"test.txt")



# Load data
print("Loading data...")
x_text, y_text = data_helpers.load_data_and_labels(text_path)
x_train, y_train = data_helpers.load_data_and_labels(train_path)
x_dev, y_dev = data_helpers.load_data_and_labels(valid_path)
x_test,y_test=data_helpers.load_data_and_labels(test_path)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
x_train = np.array(list(vocab_processor.transform(x_train)))
x_dev = np.array(list(vocab_processor.transform(x_dev)))
x_test = np.array(list(vocab_processor.transform(x_test)))

y_train=np.array(y_train)
y_dev=np.array(y_dev)
y_test=np.array(y_test)

print x_train.shape


# x = np.array(list(vocab_processor.fit_transform(x_text)))

# # Randomly shuffle data
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices]

# # Split train/test set
# # TODO: This is very crude, should use cross-validation
# x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
# y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# in tensorflow ,Session is the environment graph operations are executed and it contains state about variables and Queues
# each session operates on a single graph
# if no explicit session is created, tensorflow uses default session

# a graph contains operators and tensors.we can use same graph in multiple session but not vice versa

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)
	sess=tf.Session(config=session_conf)
	# allow_soft_placement setting allows allows tensorflow to fall back on a device with a certain operation implemented
	# when the preferred device does not exist
	# log_device_placement is tensorflow log  on which CPU places operations
	with sess.as_default():
		cnn=TextCNN(sequence_length=x_train.shape[1],num_classes=y_train.shape[1],vocab_size=len(vocab_processor.vocabulary_),
			embedding_size=FLAGS.embedding_dim,filter_sizes=list(map(int,FLAGS.filter_sizes.split(","))),	num_filters=FLAGS.num_filters)

		# use the adam optimizer 
		# by defining global step anfd passing it to optimizer,we allow tensorflow handle the counting of training 
		# steps for us
		global_step=tf.Variable(0,name="global_step",trainable=False)
		optimizer=tf.train.AdamOptimizer(1e-3)
		grads_and_vars=optimizer.compute_gradients(cnn.loss)
		# train_op is a newly created operation that we can run to perform a gradient update on our parameters.
		# each execution of train_op is a training step.global step will be automatically incremented by one every time
		# train_op is executed
		train_op=optimizer.apply_gradients(grads_and_vars,global_step=global_step)

		# tensrflow summaries helps us to keep track and visualize various quantities during training and evaluation

		# output directory for models and summaries
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp))
		print "Writing to {}\n".format(out_dir) 

		# summaries of loss and accuracy
		loss_summary = tf.scalar_summary("loss",cnn.loss)
		acc_summary = tf.scalar_summary("accuracy",cnn.accuracy)

		# train summaries
		train_summary_op = tf.merge_summary([loss_summary,acc_summary])
		train_summary_dir=os.path.join(out_dir,"summaries","train")
		train_summary_writer=tf.train.SummaryWriter(train_summary_dir,sess.graph)

		# Dev Summaries
		dev_summary_op=tf.merge_summary([loss_summary,acc_summary])
		dev_summary_dir=os.path.join(out_dir,"summaries","dev")
		dev_summary_writer=tf.train.SummaryWriter(dev_summary_dir,sess.graph)

		# Dev Summaries
		test_summary_op=tf.merge_summary([loss_summary,acc_summary])
		test_summary_dir=os.path.join(out_dir,"summaries","test")
		test_summary_writer=tf.train.SummaryWriter(test_summary_dir,sess.graph)


		# checkpointing is saving the parameters to restore them later
		# checkpoints can be used to continue training at a later point	or to pick the best parameters setting using early stopping

		# checkpointing
		checkpoint_dir=os.path.abspath(os.path.join(out_dir,"checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir,"model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver=tf.train.Saver(tf.all_variables())

		# initialize all variables
		# we can also call initializer of variables manually.that is useful if we want to initialize embeddings
		# sith pre trained values
		sess.run(tf.initialize_all_variables())


		# define a single training step,evaluating model and updating model parameters
		def train_step(x_batch,y_batch):
			""" A single Training step"""
			feed_dict={cnn.input_x:x_batch,cnn.input_y:y_batch,cnn.dropout_keep_prob:FLAGS.dropout_keep_prob}
			# we execute train_op using session.run which returns values of all operations we ask it to evaluate
			# train_op returns nothing.it just updates the parameters of network
			_, step, summaries, loss, accuracy = sess.run([train_op,global_step,train_summary_op,cnn.loss,cnn.accuracy],feed_dict)
			time_str=datetime.datetime.now().isoformat()
			# loss and accuracy for training batch may vary significantly across batches if our batvh size is small
			print "{} : step {},loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)
			train_summary_writer.add_summary(summaries,step)	

		# evaluate loss and accuracy on validation set
		# there is no training operation and no dropout
		def dev_step(x_batch,y_batch,writer=None):
			""" Evaluate model on a dev set"""
			feed_dict={cnn.input_x:x_batch,cnn.input_y:y_batch,cnn.dropout_keep_prob:1.0}
			step,summaries,loss,accuracy = sess.run([global_step,dev_summary_op,cnn.loss,cnn.accuracy],feed_dict)
			time_str = datetime.datetime.now().isoformat()
			s = "{}: step {}, loss {:g}, acc {:g}".format(time_str,step,loss,accuracy)
			print "{}: step {}, loss {:g}, acc {:g}".format(time_str,step,loss,accuracy)
			if writer:
				writer.add_summary(summaries,step)
			return s

		# generate batches
		batches = data_helpers.batch_iter(list(zip(x_train,y_train)),FLAGS.batch_size, FLAGS.num_epochs)

		# Training Loop.For each batch...
		for batch in batches:
			x_batch, y_batch = zip(*batch)
			train_step(x_batch,y_batch)
			current_step=tf.train.global_step(sess,global_step)
			if current_step%FLAGS.evaluate_every==0:
				print "\nEvalutaion:"
				dev_step(x_dev,y_dev,writer=dev_summary_writer)
				print ""
			if current_step%FLAGS.checkpoint_every==0:
				path=saver.save(sess,checkpoint_prefix,global_step=current_step)
				print "Saved model checkpoint to {}\n".format(path)

		f = open("results.txt",'a')
		f.write("Epochs : "+str(FLAGS.num_epochs)+"\n")
		print "\nTraining Accuracy..."
		f.write(dev_step(x_train,y_train,writer=dev_summary_writer)+"\n")

		print "\nValidationm Accuracy..."
		f.write(dev_step(x_dev,y_dev,writer=dev_summary_writer)+"\n")

		print "\nTesting Accuracy..."
		f.write(dev_step(x_test,y_test,writer=dev_summary_writer)+"\n")
		f.close()