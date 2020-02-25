import numpy as np
from numpy import genfromtxt
from tensorflow.contrib.learn.python.learn.datasets import base
import sys
# sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

from influence.dataset import DataSet

def exclude_some_examples(exclude, validation_size=0, remove_biased_test=False):
	assert False
	total_dataset = genfromtxt("../german-credit-dataset/normalised-features-german.csv", delimiter=",")      # this is the standarised/normalised data, so no need to renormalize
	total_labels = genfromtxt("../german-credit-dataset/labels.csv", delimiter=",")

	ascending_order_biased_points = [11, 29, 50, 56, 58, 68, 87, 93, 98, 101, 112, 134, 142, 145, 154, 170, 
	174, 180, 182, 194, 197, 199, 212, 213, 220, 226, 227, 228, 229, 237, 247, 257, 282, 285, 289, 302, 
	307, 313, 314, 335, 336, 347, 349, 360, 364, 368, 398, 402, 404, 431, 442, 450, 476, 485, 491, 501, 
	503, 507, 524, 530, 540, 556, 557, 562, 579, 587, 593, 594, 604, 612, 639, 648, 649, 655, 658, 663, 
	668, 669, 674, 699, 704, 720, 721, 727, 736, 755, 763, 765, 774, 784, 804, 811, 813, 815, 822, 827, 
	846, 853, 864, 866, 878, 900, 907, 923, 931, 934, 951, 963, 970, 973, 975, 978, 993, 999, 13, 34, 
	59, 78, 79, 105, 130, 184, 190, 216, 233, 255, 268, 293, 328, 333, 344, 383, 387, 449, 465, 467, 
	477, 509, 525, 531, 534, 561, 570, 580, 597, 598, 607, 621, 632, 653, 656, 665, 672, 731, 754, 780, 
	793, 796, 818, 849, 862, 886, 889, 917, 949, 965, 982, 5, 7, 106, 141, 166, 286, 375, 395, 421, 443, 
	481, 517, 616, 692, 719, 821, 844, 152, 287, 505, 564, 596, 646, 744, 911, 945, 124, 294, 374, 657, 
	876, 895, 921, 972, 172, 560, 205, 373, 504, 568, 666, 783, 890, 988, 225, 359, 414, 475, 618, 650, 
	1, 10, 65, 392, 419, 548, 850, 163, 337, 885, 198, 615, 637, 471, 808, 809, 99, 500, 761, 775, 155, 
	685, 272, 310, 936, 15, 240, 751, 922, 585, 633, 724, 829, 842, 439, 700, 308, 610, 722, 834, 986, 
	249, 278, 118, 640, 595, 601, 603, 444, 18, 771, 17, 747, 711, 858, 521, 647, 120, 44, 429, 80, 
	424, 195, 826, 186, 611, 92, 790, 113, 331, 129, 351, 614]

	assert(exclude < len(ascending_order_biased_points))
	removal_points = []
	if exclude > 0:
		removal_points = sorted(ascending_order_biased_points[-exclude:], reverse=True)
	assert(len(removal_points) == exclude)
		
	# import ipdb; ipdb.set_trace()

	train_examples = 800		# size changed from 750 to 800, testing set is 200
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]

	for i in removal_points:
		if i < 800:
			X_train = np.delete(X_train, i, axis = 0)
			Y_train = np.delete(Y_train, i, axis = 0)
			
	biased_test_points = sorted([i for i in ascending_order_biased_points if i >= 800], reverse=True)
	# print(len(biased_test_points))
	# exit(0)
	if remove_biased_test:
		for i in biased_test_points:
			X_test = np.delete(X_test, i - train_examples, axis = 0)
			Y_test = np.delete(Y_test, i - train_examples, axis = 0)

	if remove_biased_test: 
		assert(len(Y_test) < 1000 - train_examples)
	else:
		assert(len(Y_test) == 1000 - train_examples)

	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def load_adult_income(perm=-1, validation_size=0):
	total_dataset = genfromtxt("../../adult-income-dataset/normalized_adult_features.csv", delimiter=",")      # this is the standarised/normalised data, so no need to renormalize
	total_labels = genfromtxt("../../adult-income-dataset/adult_labels.csv", delimiter=",")
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	# no. of 1's in adult dataset is 11208, and 8947 in training set.
	train_examples = 36000		# testing set is 9222		# weird size (about 20% - similar to german credit dataset)
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


def reweighted_load_adult_income(validation_size=0):
	assert False
	total_dataset = genfromtxt("reweighted_german/normalized_reweighted_features-german.csv", delimiter=",")      # this is the standarised/normalised data, so no need to renormalize
	total_labels = genfromtxt("reweighted_german/normalized_reweighted_labels-german.csv", delimiter=",")

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


def disparate_removed_load_adult_income(validation_size=0):
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


def load_adult_income_partial(index, perm=-1, validation_size=0):
	total_dataset = genfromtxt("../../adult-income-dataset/normalized_adult_features.csv", delimiter=",")      # this is the standarised/normalised data, so no need to renormalize
	total_labels = genfromtxt("../../adult-income-dataset/adult_labels.csv", delimiter=",")
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 36000
	X_train = total_dataset[:train_examples]
	X_train = X_train[index]
	assert(len(X_train) == len(index))
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


# These are 20 permutations of the full adult income dataset. 
def permutations(perm):
	x = np.load(f"data-permutations/split{perm}.npy")
	return list(x)


np.random.seed(2)
def produce_permutations():
	assert False
	# total_dataset = genfromtxt("../german-credit-dataset/normalised-features-german.csv", delimiter=",")      # this is the standarised/normalised data, so no need to renormalize
	# total_labels = genfromtxt("../german-credit-dataset/labels.csv", delimiter=",")
	# with open("permuted_data.txt", "w") as f:
	for split in range(20):
		idx = np.random.permutation(45222)
			# f.write(str(idx.tolist()) + "\n")
		np.save(f"data-permutations/split{split}.npy", idx)
		print(split, "done")
	# x,y = data[idx], classes[idx]


if __name__ == "__main__":
    raise NotImplementedError
	# for k in range(5):
	# produce_permutations()
	# print(permutations(15)[:10])