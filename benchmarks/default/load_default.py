import numpy as np
import pandas as pd
from numpy import genfromtxt
from tensorflow.contrib.learn.python.learn.datasets import base
import sys, os
sys.path.append(".")
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


def load_default(perm=-1, validation_size=0):
	total_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/normalized_default_features.csv").to_numpy()
	total_labels = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/default_labels.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 24000		# testing set is 6000		# exactly 20%
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 6000)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def disparate_removed_load_default(perm, validation_size=0):
	total_dataset = genfromtxt(f"{os.path.dirname(os.path.realpath(__file__))}/disparate_impact_removed/normalized_disparateremoved_features-default.csv", delimiter=",")      # this is the standarised/normalised data, so no need to renormalize
	total_labels = genfromtxt(f"{os.path.dirname(os.path.realpath(__file__))}/disparate_impact_removed/normalized_disparateremoved_labels-default.csv", delimiter=",")
	
	# total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 24000		# testing set is 6000		# exactly 20%
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 6000)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def load_default_partial(index, perm=-1, validation_size=0):
	total_dataset = pd.read_csv("../../default-dataset/normalized_default_features.csv").to_numpy()
	total_labels = pd.read_csv("../../default-dataset/default_labels.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 24000
	X_train = total_dataset[:train_examples]
	X_train = X_train[index]		# removes the biased train examples here
	assert(len(X_train) == len(index))
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_train = Y_train[index]
	
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 6000)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def before_preferential_sampling(perm, validation_size=0):
	original_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/raw_default.csv")
	total_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/normalized_default_features.csv").to_numpy()
	total_labels  = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/default_labels.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]
	# import ipdb; ipdb.set_trace()
	train_examples = 24000		# testing set is 6000		# exactly 20%
	original_dataset = original_dataset.reindex(ordering[:train_examples])
	original_dataset = original_dataset.reset_index(drop=True)		# helps reset the index
	x_both = original_dataset.groupby(['sex', 'target']).indices
	# import ipdb; ipdb.set_trace()
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 6000)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test), x_both


def resampled_dataset(perm, dep_neg_candidates, dep_pos_candidates, fav_neg_candidates, fav_pos_candidates, validation_size=0):
	original_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/raw_default.csv")
	total_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/normalized_default_features.csv").to_numpy()
	total_labels  = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/default_labels.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 24000		# testing set is 6000		# exactly 20%
	original_dataset = original_dataset.reindex(ordering[:train_examples])
	original_dataset = original_dataset.reset_index(drop=True)		# helps reset the index
	x_gender = original_dataset.groupby(['sex']).indices
	x_target = original_dataset.groupby(['target']).indices

	deprived_negative_size = int(round(x_gender[0].shape[0] * x_target[0].shape[0] / train_examples)) 	# * female_bad_credit)
	deprived_positive_size = int(round(x_gender[0].shape[0] * x_target[1].shape[0] / train_examples))	# * female_good_credit)

	favoured_negative_size = int(round(x_gender[1].shape[0] * x_target[0].shape[0] / train_examples))	# * male_bad_credit)
	favoured_positive_size = int(round(x_gender[1].shape[0] * x_target[1].shape[0] / train_examples))	# * male_good_credit)

	assert deprived_negative_size + deprived_positive_size + favoured_negative_size + favoured_positive_size == train_examples
	
	# choose deprived negative candidates - no sampling - decrease
	dep_neg_finalists = dep_neg_candidates[:deprived_negative_size].tolist()
	# choose favoured positive candidates - no sampling - decrease
	fav_pos_finalists = fav_pos_candidates[:favoured_positive_size].tolist()
	
	# add extra deprived positive candidates - increase
	extra_pos = deprived_positive_size - dep_pos_candidates.shape[0]
	assert(extra_pos >= 0)
	dep_pos_finalists = dep_pos_candidates.tolist()
	while len(dep_pos_finalists) < deprived_positive_size:
		dep_pos_duplicates = dep_pos_candidates[:extra_pos].tolist()
		dep_pos_finalists.extend(dep_pos_duplicates)
		extra_pos -= len(dep_pos_duplicates)
	assert (len(dep_pos_finalists) == deprived_positive_size)

	# add extra favoured negative candidates - increase
	extra_neg = favoured_negative_size - fav_neg_candidates.shape[0]
	assert(extra_neg >= 0)
	fav_neg_finalists = fav_neg_candidates.tolist()
	while len(fav_neg_finalists) < favoured_negative_size:
		fav_neg_duplicates = fav_neg_candidates[:extra_neg].tolist()
		fav_neg_finalists.extend(fav_neg_duplicates)
		extra_neg -= len(fav_neg_duplicates)
		
	assert (len(fav_neg_finalists) == favoured_negative_size)
	# import ipdb; ipdb.set_trace()
	final_order = dep_neg_finalists + dep_pos_finalists + fav_neg_finalists + fav_pos_finalists
	final_order= sorted(final_order) 
	assert len(final_order) == train_examples

	X_train = total_dataset[final_order]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[final_order]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 6000)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def kamiran_discrimination_pairs(df):
	# Remember 0 - female and male - 1
	# for target 1 - good, 0 - bad
	x = df.groupby(['sex','target']).indices		# this gives the indices in the df, no the index
	male_good_credit = x[(1, 1)]
	male_bad_credit = x[(1, 0)]
	female_good_credit = x[(0, 1)]
	female_bad_credit = x[(0, 0)]
	d_male = male_good_credit.shape[0] + male_bad_credit.shape[0]
	male_half = male_good_credit.shape[0] / d_male
	d_female = female_good_credit.shape[0] + female_bad_credit.shape[0]
	female_half = female_good_credit.shape[0] / d_female
	discm = male_half - female_half
	pairs = int(discm * d_male * d_female / df.shape[0]) + 1
	return discm, pairs, male_good_credit, male_bad_credit, female_good_credit, female_bad_credit
	

def before_massaging_dataset(perm, validation_size=0):
	original_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/raw_default.csv")
	total_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/normalized_default_features.csv").to_numpy()
	total_labels  = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/default_labels.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]
	
	train_examples = 24000		# testing set is 6000		# exactly 20%
	original_dataset = original_dataset.reindex(ordering[:train_examples])
	original_dataset = original_dataset.reset_index(drop=True)		# helps reset the index
	# import ipdb; ipdb.set_trace()
	discm, pairs_to_flip, male_good_credit, male_bad_credit, female_good_credit, female_bad_credit = kamiran_discrimination_pairs(original_dataset)
	# print(perm, discm, pairs_to_flip)
	# return
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 6000)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test), male_good_credit, male_bad_credit, female_good_credit, female_bad_credit, pairs_to_flip
	

def massaged_dataset(perm, promotion_candidates, demotion_candidates, validation_size=0):
	original_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/raw_default.csv")
	total_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/normalized_default_features.csv").to_numpy()
	total_labels  = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../default-dataset/default_labels.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 24000		# testing set is 6000		# exactly 20%
	original_dataset = original_dataset.reindex(ordering[:train_examples])
	original_dataset = original_dataset.reset_index(drop=True)		# helps reset the index
	for p, d in zip(promotion_candidates, demotion_candidates):
		assert original_dataset.loc[p, 'sex'] == original_dataset.loc[p, 'target'] == int(total_labels[p]) == 0
		assert original_dataset.loc[d, 'sex'] == original_dataset.loc[d, 'target'] == int(total_labels[d]) == 1
		original_dataset.loc[p, 'target'] = 1		# promote the female of bad credit
		total_labels[p] = 1.0
		original_dataset.loc[d, 'target'] = 0		# demote the male of good credit
		total_labels[d] = 0.0
		assert p < train_examples	# the index of both promotion and demotion candidates should be within training set
		assert d < train_examples

	discm, pairs_to_flip, _, _, _, _ = kamiran_discrimination_pairs(original_dataset)
	assert discm <= 0 	# negative or zero discrimination
	assert pairs_to_flip == 0 or pairs_to_flip == 1
	# you can't check df feature as it is normalized values
	# df_target = pd.DataFrame(total_labels, columns=['target'])
	# df_feature = pd.DataFrame(total_dataset, columns=original_dataset.columns.tolist()[:-1])		# column names remove target
	# df_feature['target'] = df_target
	# import ipdb; ipdb.set_trace()
	# print(perm, discm, pairs_to_flip)
	# return
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 6000)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


# These are 20 permutations of the full default dataset. 
def permutations(perm):
	x = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/data-permutations/split{perm}.npy")
	return list(x)


np.random.seed(2)
def produce_permutations():
	for split in range(20):
		idx = np.random.permutation(30000)	# size of default dataset
		np.save(f"data-permutations/split{split}.npy", idx)
		print(split, "done")


if __name__ == "__main__":
    raise NotImplementedError
	# for k in range(5):
	# produce_permutations()
	# print(permutations(15)[:10])