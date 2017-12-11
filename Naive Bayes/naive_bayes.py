# Naive Bayes Classifier for Sentiment Analysis
# Group 1
# Topics in DBMS

# Reference for Naive Bayes for Sentiment Analysis : http://ataspinar.com/2016/02/15/sentiment-analysis-with-the-naive-bayes-classifier/

from collections import defaultdict, OrderedDict
from math import log

# sudo pip install nltk
# after installation, run nltk.download() in shell/script
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Types for classes
'''
0 - Very Negative
1 - Negative
2 - Neutral
3 - Positive
4 - Very Positive
'''

remove_stop_words = True

classes = ['0','1','2','3','4']

# store the train samples and different statistics
train_file = open("train.txt","r")

# stores all sentences in training set
training_list=[]

# dictionary storing classes as key and list of all sentences of this class as value
training_set=defaultdict(list)

# get the stop words in th english from the corpus
stop_words = set(stopwords.words("english"))

# stroe the training samples in list
for line in train_file:
	line=line.split("\t")
	training_list.append((line[0],line[2].strip("\n")))
	training_set[line[2].strip("\n")].append(line[0])

# sorting the training set by key
training_set = OrderedDict(sorted(training_set.items()))

# to store the number of training samples per class 
no_of_samples_per_class = [len(training_set[i]) for i in classes]

# total no of samples in training
total_no_samples=sum(no_of_samples_per_class)

# stores frequency map of words per class
frequency_map_class = {}
for clas, samples in training_set.iteritems():
	# stores word frequencies in this class
	frequency = {}
	# initialize values of all keys to zero
	frequency = defaultdict(lambda:0, frequency)
	for sample in samples:
		# returns the list of tokens in sentence
		sample = word_tokenize(sample)
		# consider the word if it is not in stop_words
		if remove_stop_words==True:
			sample = [w for w in sample if w not in stop_words]
		for word in sample:
			frequency[word]+=1
	# after calulating frequency map of class, append it to main frequency map per class
	frequency_map_class[clas] = frequency

# total number of words per class
total_words_class = {c:sum(frequency_map_class[c].itervalues()) for c in frequency_map_class}


# given a sample, finds the probability that it belongs to this class
def clas_prob(clas):
	# one could define number of training samples of this class / total no of samples
	# here for generalisation, new sample could belong to any of the classes
	return 1.0/len(classes)

# find the probablity that given sample belongs to a given clas
def cond_prob_log(sample, clas):
	# this is product of probabilities of each word belonging to this class
	# which is equal to product of (frequency of each word in this class / sum of frequencies of all terms in this sample)
	# but in involves many multiplications.so apply log for normalization
	sample = word_tokenize(sample)
	# consider the word if it is not in stop_words
	if remove_stop_words==True:
			sample = [w for w in sample if w not in stop_words]
	p=0.0
	for word in sample:
		# 1 is added to prevent math domain error in log function
		p+=log(frequency_map_class[clas][word]+1)
	p-=len(sample)*log(total_words_class[clas])
	return p

# read the validation data to calculate validation accuracy
valid_file = open("valid.txt","r")
valid_list=[]
for line in valid_file:
	line=line.split("\t")
	valid_list.append((line[0],line[2].strip("\n")))

# read the test data to calculate 
test_file = open("test.txt","r")
test_list = []
for line in test_file:
	line=line.split("\t")
	test_list.append((line[0],line[2].strip("\n")))

data_to_test = [training_list, valid_list, test_list]

# build confusion matrices for training, validation and testing
confusion_matrices = []

for data in data_to_test:
	confusion_matrix = [[0 for i in range(len(classes))] for i in range(len(classes))]
	for item in data:
		sample=item[0]
		max_prob=-1000000
		pred_class=-1
		for clas in classes:
			# calcualte log values for easy computation
			log_prob=cond_prob_log(sample,clas)+log(clas_prob(clas))
			if log_prob>max_prob:
				max_prob=log_prob
				pred_class=int(clas)
		actual_class=int(item[1])
		# print actual_class, pred_class
		confusion_matrix[actual_class][pred_class]+=1
	confusion_matrices.append(confusion_matrix)
	

accuracy_list=["Training Accuracy","Validation Accuracy","Test Accuracy"]

# to calculate accuracy in each test, calculate TP+TN/(TP+TN+FP+FN)
# TP+TN = diagonal sum
# FP = column of that class except correct prediction
# FN = row of that class except correct prediction

# Ref to calculate accuracy in confusion matrix
# https://www.researchgate.net/post/Can_someone_help_me_to_calculate_accuracy_sensitivity_of_a_66_confusion_matrix

print "Naive Bayes Classifier\n"
for i in range(len(confusion_matrices)):
	diagonal_sum = sum([confusion_matrices[i][j][j] for j in range(len(confusion_matrices[i]))])
	total=0.0
	for j in range(len(confusion_matrices[i])):
		b = sum([confusion_matrices[i][k][j] for k in range(len(confusion_matrices[i])) if k!=j])
		a = sum([confusion_matrices[i][j][k] for k in range(len(confusion_matrices[i])) if k!=j])
		total+=float(diagonal_sum)/(diagonal_sum+a+b)
	accuracy=float(total)/len(confusion_matrices[i])
	print accuracy_list[i],':',accuracy














