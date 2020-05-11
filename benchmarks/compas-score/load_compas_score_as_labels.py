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
	raise NotImplementedError
	total_dataset = pd.read_csv("../../compas-dataset/normalized_scores_as_labels_features.csv").to_numpy()
	total_labels = pd.read_csv("../../compas-dataset/target_compas_score_as_label.csv").to_numpy()
	total_labels = total_labels.flatten()
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


def load_compas_two_year_nosensitive(perm=-1, validation_size=0):
	total_dataset = pd.read_csv("../../compas-dataset/normalized_scores_as_labels_nosensitive_features.csv").to_numpy()
	total_labels = pd.read_csv("../../compas-dataset/target_compas_score_as_label.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 5000		# testing set is 1150		# weird size (about 20% - similar to german credit dataset and adult income dataset)
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 1150)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def load_compas_two_year(perm=-1, validation_size=0):
	total_dataset = pd.read_csv("../../compas-dataset/normalized_scores_as_labels_features.csv").to_numpy()
	total_labels = pd.read_csv("../../compas-dataset/target_compas_score_as_label.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 5000		# testing set is 1150		# weird size (about 20% - similar to german credit dataset and adult income dataset)
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 1150)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def load_compas_partial_method1(perm, model_count, train_pts_removed, name, validation_size=0):
	total_dataset = pd.read_csv("../../compas-dataset/normalized_scores_as_labels_features.csv").to_numpy()
	total_labels = pd.read_csv("../../compas-dataset/target_compas_score_as_label.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 5000		# testing set is 1150
	X_train = total_dataset[:train_examples]
	Y_train = total_labels[:train_examples]

	ranked_influential_training_points = f"ranking_points_ordered_method1/{name}.npy"
	sorted_training_points = list(np.load(ranked_influential_training_points))
	remaining_train_indexes = np.array(sorted_training_points[train_pts_removed:])
	assert len(remaining_train_indexes) <= len(X_train)
	X_train = X_train[remaining_train_indexes]
	Y_train = Y_train[remaining_train_indexes]

	X_validation = total_dataset[train_examples:train_examples + validation_size]
	Y_validation = total_labels[train_examples:train_examples + validation_size]

	X_test  = total_dataset[train_examples + validation_size:]
	Y_test  = total_labels[train_examples + validation_size:]
	test_points = np.array(ordering[train_examples + validation_size:])
	biased_test_points = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/compas-score_biased_points.npy")
	# intersection = np.intersect1d(test_points, biased_test_points)
	mask = np.in1d(test_points, biased_test_points)		# True if the point is biased
	mask_new = ~mask			# invert it		# this is a boolean vector
	X_test = X_test[mask_new]
	Y_test = Y_test[mask_new]
	assert(X_test.shape == (len(test_points)-sum(mask), X_train.shape[1]))
	assert(len(Y_test) == len(test_points)-sum(mask))
	print(len(Y_test), len(Y_train), "see the length of test and train")

	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def load_fair_representations(perm, training_dataset, training_labels, validation_size=0):
	total_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/normalized_scores_as_labels_features.csv").to_numpy()
	total_labels = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/target_compas_score_as_label.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 5000		# testing set is 1150
	X_train = training_dataset		# this is already permuted in the right order
	X_validation = training_dataset[train_examples:train_examples + validation_size]
	Y_train = training_labels
	Y_validation = training_dataset[train_examples:train_examples + validation_size]

	X_test  = total_dataset[train_examples + validation_size:]
	Y_test  = total_labels[train_examples + validation_size:]
	test_points = np.array(ordering[train_examples + validation_size:])
	biased_test_points = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/compas-score_biased_points.npy")
	# intersection = np.intersect1d(test_points, biased_test_points)
	mask = np.in1d(test_points, biased_test_points)		# True if the point is biased
	mask_new = ~mask			# invert it		# this is a boolean vector
	X_test = X_test[mask_new]
	Y_test = Y_test[mask_new]
	assert(X_test.shape == (len(test_points)-sum(mask), X_train.shape[1]))
	assert(len(Y_test) == len(test_points)-sum(mask))

	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def disparate_removed_load_compas(perm, validation_size=0):
	sys.path.insert(1, "../")
	sys.path.append("../../../")
	sys.path.append("../../../competitors/AIF360/")
	from aif360.datasets import MyCompasScoreDataset
	from aif360.metrics import BinaryLabelDatasetMetric
	from aif360.algorithms.preprocessing import DisparateImpactRemover
	from sklearn.preprocessing import MinMaxScaler

	total_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/disparate_impact_removed/normalized_disparateremoved_features-compas_score.csv").to_numpy()
	total_labels = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/disparate_impact_removed/normalized_disparateremoved_labels-compas_score.csv").to_numpy()
	total_labels = total_labels.flatten()

	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	dataset_orig = MyCompasScoreDataset(
		protected_attribute_names=['race'],                   
		privileged_classes=[[1]], 
		normalized = False,
		permute = perm
	)
	
	train_examples = 5000
	dataset_orig_train, dataset_orig_test = dataset_orig.split([train_examples], shuffle=False)
	assert(len(dataset_orig_train.convert_to_dataframe()[0]) == train_examples)
	di = DisparateImpactRemover(repair_level=1.0)
	train_repd = di.fit_transform(dataset_orig_train)
	new_df = train_repd.convert_to_dataframe()[0]		# this also has labels
	target = new_df['compas_score']
	new_df = new_df.drop(columns=['compas_score'])
	mins_and_ranges = []
	for j in list(new_df):
		i = new_df[j]
		mins_and_ranges.append((np.min(i), np.max(i) - np.min(i)))
	assert(len(mins_and_ranges) == 10)
	
	df_ = new_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
	X_train = df_.to_numpy()
	Y_train = target.to_numpy()

	X_validation = total_dataset[train_examples:train_examples + validation_size]
	Y_validation = total_labels[train_examples:train_examples + validation_size]

	X_test  = total_dataset[train_examples + validation_size:]
	Y_test  = total_labels[train_examples + validation_size:]
	test_points = np.array(ordering[train_examples + validation_size:])
	biased_test_points = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/compas-score_biased_points.npy")
	# intersection = np.intersect1d(test_points, biased_test_points)
	mask = np.in1d(test_points, biased_test_points)		# True if the point is biased
	mask_new = ~mask			# invert it		# this is a boolean vector
	X_test = X_test[mask_new]
	Y_test = Y_test[mask_new]
	assert(X_test.shape == (len(test_points)-sum(mask), X_train.shape[1]))
	assert(len(Y_test) == len(test_points)-sum(mask))

	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test), mins_and_ranges


def load_compas_two_year_partial(index, perm=-1, validation_size=0):
	total_dataset = pd.read_csv("../../compas-dataset/normalized_scores_as_labels_features.csv").to_numpy()
	total_labels = pd.read_csv("../../compas-dataset/target_compas_score_as_label.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 5000
	X_train = total_dataset[:train_examples]
	X_train = X_train[index]
	assert(len(X_train) == len(index))
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_train = Y_train[index]
	
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 1150)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def before_preferential_sampling(perm, validation_size=0):
	original_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/compas_score_as_label.csv")
	original_dataset['sex'] = original_dataset['sex'].replace({"Male":1, "Female":0})
	original_dataset['race'] = original_dataset['race'].replace({"Caucasian":1, "African-American":0})
	original_dataset['c_charge_degree'] = original_dataset['c_charge_degree'].replace({"F":1, "M":0})
	total_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/normalized_scores_as_labels_features.csv").to_numpy()
	total_labels = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/target_compas_score_as_label.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]
	# import ipdb; ipdb.set_trace()
	train_examples = 5000		# testing set is 1150
	original_dataset = original_dataset.reindex(ordering[:train_examples])
	original_dataset = original_dataset.reset_index(drop=True)		# helps reset the index
	x_both = original_dataset.groupby(['race', 'compas_score']).indices
	# import ipdb; ipdb.set_trace()
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	assert(len(Y_test) == 1150)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test), x_both


def resampled_dataset(perm, dep_neg_candidates, dep_pos_candidates, fav_neg_candidates, fav_pos_candidates, validation_size=0):
	original_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/compas_score_as_label.csv")
	original_dataset['sex'] = original_dataset['sex'].replace({"Male":1, "Female":0})
	original_dataset['race'] = original_dataset['race'].replace({"Caucasian":1, "African-American":0})
	original_dataset['c_charge_degree'] = original_dataset['c_charge_degree'].replace({"F":1, "M":0})
	total_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/normalized_scores_as_labels_features.csv").to_numpy()
	total_labels = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/target_compas_score_as_label.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 5000		# testing set is 1150
	original_dataset = original_dataset.reindex(ordering[:train_examples])
	original_dataset = original_dataset.reset_index(drop=True)		# helps reset the index
	x_gender = original_dataset.groupby(['race']).indices
	x_target = original_dataset.groupby(['compas_score']).indices
	# import ipdb; ipdb.set_trace()
	# deprived_negative_size = int(round(x_gender[0].shape[0] * x_target[0].shape[0] / train_examples)) 	# * female_bad_credit)
	# deprived_positive_size = int(round(x_gender[0].shape[0] * x_target[1].shape[0] / train_examples))	# * female_good_credit)

	# favoured_negative_size = int(round(x_gender[1].shape[0] * x_target[0].shape[0] / train_examples))	# * male_bad_credit)
	# favoured_positive_size = int(round(x_gender[1].shape[0] * x_target[1].shape[0] / train_examples))	# * male_good_credit)

	deprived_negative_size = int(round(x_gender[1].shape[0] * x_target[0].shape[0] / train_examples)) 	# * male_bad_credit)
	deprived_positive_size = int(round(x_gender[1].shape[0] * x_target[1].shape[0] / train_examples))	# * male_good_credit)

	favoured_negative_size = int(round(x_gender[0].shape[0] * x_target[0].shape[0] / train_examples))	# * female_bad_credit)
	favoured_positive_size = int(round(x_gender[0].shape[0] * x_target[1].shape[0] / train_examples))	# * female_good_credit)

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
	Y_train = total_labels[final_order]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	# assert(len(Y_test) == 1150)
	
	X_test  = total_dataset[train_examples + validation_size:]
	Y_test  = total_labels[train_examples + validation_size:]
	test_points = np.array(ordering[train_examples + validation_size:])
	biased_test_points = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/compas-score_biased_points.npy")
	# intersection = np.intersect1d(test_points, biased_test_points)
	mask = np.in1d(test_points, biased_test_points)		# True if the point is biased
	mask_new = ~mask			# invert it		# this is a boolean vector
	X_test = X_test[mask_new]
	Y_test = Y_test[mask_new]
	assert(X_test.shape == (len(test_points)-sum(mask), X_train.shape[1]))
	assert(len(Y_test) == len(test_points)-sum(mask))
	
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


def kamiran_discrimination_pairs(df):
	# Remember 1 - female and male - 0		# reversed in this dataset
	# for target 1 - good, 0 - bad
	x = df.groupby(['race', 'compas_score']).indices		# this gives the indices in the df, no the index
	# male_good_credit = x[(1, 1)]
	# male_bad_credit = x[(1, 0)]
	# female_good_credit = x[(0, 1)]
	# female_bad_credit = x[(0, 0)]

	male_good_credit = x[(0, 1)]
	male_bad_credit = x[(0, 0)]
	female_good_credit = x[(1, 1)]
	female_bad_credit = x[(1, 0)]

	d_male = male_good_credit.shape[0] + male_bad_credit.shape[0]
	male_half = male_good_credit.shape[0] / d_male
	d_female = female_good_credit.shape[0] + female_bad_credit.shape[0]
	female_half = female_good_credit.shape[0] / d_female
	discm = male_half - female_half
	pairs = int(discm * d_male * d_female / df.shape[0]) + 1
	return discm, pairs, male_good_credit, male_bad_credit, female_good_credit, female_bad_credit
	

def before_massaging_dataset(perm, validation_size=0):
	original_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/compas_score_as_label.csv")
	original_dataset['sex'] = original_dataset['sex'].replace({"Male":1, "Female":0})
	original_dataset['race'] = original_dataset['race'].replace({"Caucasian":1, "African-American":0})
	original_dataset['c_charge_degree'] = original_dataset['c_charge_degree'].replace({"F":1, "M":0})
	total_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/normalized_scores_as_labels_features.csv").to_numpy()
	total_labels = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/target_compas_score_as_label.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]
	
	train_examples = 5000		# testing set is 1150
	original_dataset = original_dataset.reindex(ordering[:train_examples])
	original_dataset = original_dataset.reset_index(drop=True)		# helps reset the index
	discm, pairs_to_flip, male_good_credit, male_bad_credit, female_good_credit, female_bad_credit = kamiran_discrimination_pairs(original_dataset)
	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	X_test  = total_dataset[train_examples + validation_size:]
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	Y_test  = total_labels[train_examples + validation_size:]
	# assert(len(Y_test) == 1150)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test), male_good_credit, male_bad_credit, female_good_credit, female_bad_credit, pairs_to_flip
	

def massaged_dataset(perm, promotion_candidates, demotion_candidates, validation_size=0):
	original_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/compas_score_as_label.csv")
	original_dataset['sex'] = original_dataset['sex'].replace({"Male":1, "Female":0})
	original_dataset['race'] = original_dataset['race'].replace({"Caucasian":1, "African-American":0})
	original_dataset['c_charge_degree'] = original_dataset['c_charge_degree'].replace({"F":1, "M":0})
	total_dataset = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/normalized_scores_as_labels_features.csv").to_numpy()
	total_labels = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/../../compas-dataset/target_compas_score_as_label.csv").to_numpy()
	total_labels = total_labels.flatten()
	assert(perm < 20)		# we only have 20 permutations
	if perm >= 0:	# for negative number don't do
		ordering = permutations(perm)
		total_dataset, total_labels = total_dataset[ordering], total_labels[ordering]

	train_examples = 5000		# testing set is 1150
	original_dataset = original_dataset.reindex(ordering[:train_examples])
	original_dataset = original_dataset.reset_index(drop=True)		# helps reset the index
	for p, d in zip(promotion_candidates, demotion_candidates):
		assert original_dataset.loc[d, 'race'] == original_dataset.loc[p, 'compas_score'] == int(total_labels[p]) == 0		# notice in the first part, p and d are flipped in these two statements
		assert original_dataset.loc[p, 'race'] == original_dataset.loc[d, 'compas_score'] == int(total_labels[d]) == 1
		original_dataset.loc[p, 'compas_score'] = 1		# promote the female of bad credit
		total_labels[p] = 1.0
		original_dataset.loc[d, 'compas_score'] = 0		# demote the male of good credit
		total_labels[d] = 0.0
		assert p < train_examples	# the index of both promotion and demotion candidates should be within training set
		assert d < train_examples

	discm, pairs_to_flip, _, _, _, _ = kamiran_discrimination_pairs(original_dataset)
	assert discm <= 0 	# negative or zero discrimination
	assert pairs_to_flip == 0 or pairs_to_flip == 1

	X_train = total_dataset[:train_examples]
	X_validation = total_dataset[train_examples:train_examples + validation_size]
	Y_train = total_labels[:train_examples]
	Y_validation = total_labels[train_examples:train_examples + validation_size]
	
	X_test  = total_dataset[train_examples + validation_size:]
	Y_test  = total_labels[train_examples + validation_size:]
	test_points = np.array(ordering[train_examples + validation_size:])
	biased_test_points = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/compas-score_biased_points.npy")
	# intersection = np.intersect1d(test_points, biased_test_points)
	mask = np.in1d(test_points, biased_test_points)		# True if the point is biased
	mask_new = ~mask			# invert it		# this is a boolean vector
	X_test = X_test[mask_new]
	Y_test = Y_test[mask_new]
	assert(X_test.shape == (len(test_points)-sum(mask), X_train.shape[1]))
	assert(len(Y_test) == len(test_points)-sum(mask))

	# assert(len(Y_test) == 1150)
	train = DataSet(X_train, Y_train)
	validation = DataSet(X_validation, Y_validation)
	test = DataSet(X_test, Y_test)

	return base.Datasets(train=train, validation=validation, test=test)


# These are 20 permutations of the full compas dataset. 
def permutations(perm):
	x = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/../compas/data-permutations-two-year/split{perm}.npy")
	return list(x)


np.random.seed(2)
def produce_permutations():
	for split in range(20):
		idx = np.random.permutation(6150)	# size of compas_two_year dataset
		np.save(f"data-permutations-two-year/split{split}.npy", idx)
		print(split, "done")


if __name__ == "__main__":
    raise NotImplementedError
	# for k in range(5):
	# produce_permutations()
	# print(permutations(15)[:10])