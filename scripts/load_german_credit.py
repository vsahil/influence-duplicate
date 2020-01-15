import numpy as np
from numpy import genfromtxt
from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet

def load_german_credit(validation_size=50):
	total_dataset = genfromtxt("../german-credit-dataset/normalised-features-german.csv", delimiter=",")      # this is the standarised/normalised data, so no need to renormalize
	total_labels = genfromtxt("../german-credit-dataset/labels.csv", delimiter=",")
	
	train_examples = 750
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]

	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)



def load_german_credit_partial(index, validation_size=50):
	total_dataset = genfromtxt("../german-credit-dataset/normalised-features-german.csv", delimiter=",")      # this is the standarised/normalised data, so no need to renormalize
	total_labels = genfromtxt("../german-credit-dataset/labels.csv", delimiter=",")

	train_examples = 750
	X_train = total_dataset[:train_examples]
	X_train = X_train[index]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_train = Y_train[index]
	
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]

	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)

