import numpy as np
import pandas as pd
from numpy import genfromtxt
from tensorflow.contrib.learn.python.learn.datasets import base
import sys
# sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

from influence.dataset import DataSet

def exclude_some_examples(exclude, validation_size=0, remove_biased_test=True):
	assert False
	total_dataset = genfromtxt("../../adult-income-dataset/normalized_adult_features.csv", delimiter=",")
	total_labels = genfromtxt("../../adult-income-dataset/adult_labels.csv", delimiter=",")
	real_biased_df = pd.read_csv("intersections.csv")		# it is sorted in ascending order of biased points

	descending_order_biased_points = real_biased_df.Points.tolist()[::-1]		# reversed the ascending list

	assert(exclude < len(descending_order_biased_points))
	removal_points = []
	train_examples = 8000
	if exclude > 0:
		for d in descending_order_biased_points:
			if d < train_examples:
				removal_points.append(d)
			if len(removal_points) == exclude:
				break

	assert(len(removal_points) == exclude)
	removal_points = sorted(removal_points, reverse=True)	# descending order
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]

	for i in removal_points:
		# if i < train_examples:		# all must be less than train_examples
		X_train = np.delete(X_train, i, axis = 0)
		Y_train = np.delete(Y_train, i, axis = 0)
			
	biased_test_points = sorted([i for i in descending_order_biased_points if i >= train_examples], reverse=True)
	# print(len(biased_test_points), "see")
	# exit(0)
	if remove_biased_test:
		for i in biased_test_points:
			X_test = np.delete(X_test, i - train_examples, axis = 0)
			Y_test = np.delete(Y_test, i - train_examples, axis = 0)

	if remove_biased_test: 
		assert(len(Y_test) < 45222 - train_examples)
	else:
		assert(len(Y_test) == 45222 - train_examples)

	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def load_student(perm=-1, validation_size=0):
	total_dataset = pd.read_csv("../../student-dataset/normalized_student_features.csv").to_numpy()
	total_labels = pd.read_csv("../../student-dataset/student_labels.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 500		# testing set is 149		# weird size (about 20% - similar to german credit dataset and adult income dataset)
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 149)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def disparate_removed_load_student(validation_size=0):
	assert False
	total_dataset = genfromtxt("disparate_impact_removed/normalized_disparateremoved_features-german.csv", delimiter=",")      # this is the standarised/normalised data, so no need to renormalize
	total_labels = genfromtxt("disparate_impact_removed/normalized_disparateremoved_labels-german.csv", delimiter=",")

	train_examples = 800		# size changed from 750 to 800, testing set is 200
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


def load_student_partial(index, perm=-1, validation_size=0):
	total_dataset = pd.read_csv("../../student-dataset/normalized_student_features.csv").to_numpy()
	total_labels = pd.read_csv("../../student-dataset/student_labels.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 500
	X_train = total_dataset[:train_examples]
	X_train = X_train[index]		# removes the biased train examples here
	assert(len(X_train) == len(index))
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_train = Y_train[index]
	
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 149)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


# These are 20 permutations of the full student dataset. 
def permutations(perm):
	x = np.load(f"data-permutations-two-year/split{perm}.npy")
	return list(x)


np.random.seed(2)
def produce_permutations():
	for split in range(20):
		idx = np.random.permutation(649)	# size of student dataset
		np.save(f"data-permutations/split{split}.npy", idx)
		print(split, "done")


if __name__ == "__main__":
    raise NotImplementedError
	# for k in range(5):
	# produce_permutations()
	# print(permutations(15)[:10])