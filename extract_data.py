import numpy as np
import unicodedata

root = "stanfordSentimentTreebank/"

sentences_file = open(root+"datasetSentences.txt")
sentences=[]
sentences_id = {}
for line in sentences_file:
	sentence = line.split("\t")[1].strip("\n")
	# sentence = unicode(line.split("\t")[1].strip("\n"),"ISO-8859-1")
	# sentence = sentence.encode("utf-8")
	# unicodedata.normalize('NFKD', sentence).encode('ascii','ignore')
	# print sentence
	sentences.append(sentence)
	sentences_id[sentence]=line.split("\t")[0]
sentences_file.close()

phrase_id_file = open(root+"dictionary.txt")
phrases={}
for line in phrase_id_file:
	line=line.split("|")
	phrases[line[0]]=line[1].strip("\n")
phrase_id_file.close()

phraseid_sentiment_file = open(root+"sentiment_labels.txt")
phrase_sentiments = {}
for line in phraseid_sentiment_file:
	line = line.split("|")
	phrase_sentiments[line[0]]=line[1].strip("\n")
phraseid_sentiment_file.close()

sentenceid_splitset_file = open(root+"datasetSplit.txt")
sentenceid_splitset = {}
for line in sentenceid_splitset_file:
	line = line.split(",")
	# print line
	sentenceid_splitset[line[0]]=line[1].strip("\n")
sentenceid_splitset_file.close()


def get_class(label):
	k=float(label)*5.0
	if k>=0 and k<=1:
		return "0"
	elif k>=1 and k<=2:
		return "1"
	elif k>=2 and k<=3:
		return "2"
	elif k>=3 and k<=4:
		return "3"
	elif k>=4 and k<=5:
		return "4"

sentence_labels_file = open("sentence_labels.txt","w")
train_file = open("train.txt","w")
valid_file = open("valid.txt","w")
test_file = open("test.txt","w")
for sentence in sentences:
	if sentence in phrases and sentence in sentences_id:
		sentence_id = sentences_id[sentence]
		split_set = sentenceid_splitset[sentence_id]
		phrase_id = phrases[sentence]
		label = phrase_sentiments[phrase_id]
		sentence_labels_file.write(sentence+"\t"+label+"\n")
		if split_set=="1":
			train_file.write(sentence+"\t"+label+"\t"+get_class(label)+"\n")
		elif split_set=="2":
			test_file.write(sentence+"\t"+label+"\t"+get_class(label)+"\n")
		elif split_set=="3":
			valid_file.write(sentence+"\t"+label+"\t"+get_class(label)+"\n")

		# print sentence, label
sentence_labels_file.close()
train_file.close()
valid_file.close()
test_file.close()