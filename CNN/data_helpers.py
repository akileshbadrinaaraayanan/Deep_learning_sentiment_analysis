import numpy as np
import re
import itertools
from collections import Counter
import os

root = "/home/phaneendra/Documents/TDBMS/project"
train_path=os.path.join(root,"train.txt")
valid_path=os.path.join(root,"valid.txt")

num_classes=5

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(path):
	sentences=[]
	labels=[]
	f = open(path,'r')
	for line in f:
		line=line.strip().split("\t")
		# print line
		sentences.append(clean_str(line[0]))
		label_vector = [0 for i in range(num_classes)]
		label=int(line[2])
		label_vector[label]=1
		labels.append(label_vector)

	# print sentences
	# print labels

    # # Load data from files
    # train_data = list(open(train_path, "r").readlines())
    # train_data = [s.strip().split("\t")[0] for s in train_data]

    # negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    # negative_examples = [s.strip() for s in negative_examples]

    # # Split by words
    # # x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    # y = np.concatenate([positive_labels, negative_labels], 0)


	return [sentences, labels]

# load_data_and_labels(train_path)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]