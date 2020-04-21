from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
from pathlib import Path
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse
from scipy.optimize import fmin_ncg

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import backend as K
from tensorflow.contrib.learn.python.learn.datasets import base

from influence.hessians import hessian_vector_product
from influence.dataset import DataSet


def variable(name, shape, initializer):
    dtype = tf.float32
    var = tf.get_variable(
        name, 
        shape, 
        initializer=initializer, 
        dtype=dtype)
    return var

def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = variable(
        name, 
        shape, 
        initializer=tf.truncated_normal_initializer(
            stddev=stddev, 
            dtype=dtype))
 
    if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)

    return var

def normalize_vector(v):
    """
    Takes in a vector in list form, concatenates it to form a single vector,
    normalizes it to unit length, then returns it in list form together with its norm.
    """
    norm_val = np.linalg.norm(np.concatenate(v))
    norm_v = [a/norm_val for a in v]
    return norm_v, norm_val


class GenericNeuralNet(object):
    """
    Multi-class classification.
    """

    def __init__(self, **kwargs):
        np.random.seed(0)
        tf.set_random_seed(0)
        
        self.batch_size = kwargs.pop('batch_size')
        self.data_sets = kwargs.pop('data_sets')
        self.discm_data_set = None
        self.train_dir = kwargs.pop('train_dir', 'output')
        self.hvp_files = kwargs.pop('hvp_files')
        self.hvp_iterations = 0
        self.log_dir = kwargs.pop('log_dir', 'log')
        self.model_name = kwargs.pop('model_name')
        self.scheme_name = kwargs.pop('scheme')
        self.num_classes = kwargs.pop('num_classes')
        self.initial_learning_rate = kwargs.pop('initial_learning_rate')        
        self.decay_epochs = kwargs.pop('decay_epochs')

        if 'keep_probs' in kwargs: self.keep_probs = kwargs.pop('keep_probs')
        else: self.keep_probs = None
        
        if 'mini_batch' in kwargs: self.mini_batch = kwargs.pop('mini_batch')        
        else: self.mini_batch = True
        
        if 'damping' in kwargs: self.damping = kwargs.pop('damping')
        else: self.damping = 0.0
        
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # Initialize session
        config = tf.ConfigProto()        
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

        # Setup input
        self.input_placeholder, self.labels_placeholder = self.placeholder_inputs()
        self.num_train_examples = self.data_sets.train.labels.shape[0]
        self.num_test_examples = self.data_sets.test.labels.shape[0]
        
        # Setup inference and training
        if self.keep_probs is not None:
            self.keep_probs_placeholder = tf.placeholder(tf.float32, shape=(2))
            self.logits = self.inference(self.input_placeholder, self.keep_probs_placeholder)
        elif hasattr(self, 'inference_needs_labels'):            
            self.logits = self.inference(self.input_placeholder, self.labels_placeholder)
        else:
            self.logits = self.inference(self.input_placeholder)

        self.total_loss, self.loss_no_reg, self.indiv_loss_no_reg = self.loss(
            self.logits, 
            self.labels_placeholder)
        

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.Variable(self.initial_learning_rate, name='learning_rate', trainable=False)
        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        self.update_learning_rate_op = tf.assign(self.learning_rate, self.learning_rate_placeholder)
        
        self.train_op = self.get_train_op(self.total_loss, self.global_step, self.learning_rate)
        self.train_sgd_op = self.get_train_sgd_op(self.total_loss, self.global_step, self.learning_rate)
        self.accuracy_op = self.get_accuracy_op(self.logits, self.labels_placeholder)        
        self.preds = self.predictions(self.logits)

         # summary
        tf.summary.scalar('loss_val', self.total_loss)
        tf.summary.scalar('train_acc', self.accuracy_op)
        self.write_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # Setup misc
        self.saver = tf.train.Saver()

        # Setup gradients and Hessians
        self.params = self.get_all_params()
        self.grad_total_loss_op = tf.gradients(self.total_loss, self.params)
        self.grad_loss_no_reg_op = tf.gradients(self.loss_no_reg, self.params)
        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]
        self.u_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]

        self.hessian_vector = hessian_vector_product(self.total_loss, self.params, self.v_placeholder)

        self.grad_loss_wrt_input_op = tf.gradients(self.total_loss, self.input_placeholder)        

        # Because tf.gradients auto accumulates, we probably don't need the add_n (or even reduce_sum)        
        self.influence_op = tf.add_n(
            [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in zip(self.grad_total_loss_op, self.v_placeholder)])

        self.grad_influence_wrt_input_op = tf.gradients(self.influence_op, self.input_placeholder)
    
        self.checkpoint_file = os.path.join(self.train_dir, "%s-checkpoint" % self.model_name)

        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test)

        init = tf.global_variables_initializer()        
        self.sess.run(init)

        self.vec_to_list = self.get_vec_to_list_fn()
        self.adversarial_loss, self.indiv_adversarial_loss = self.adversarial_loss(self.logits, self.labels_placeholder)
        if self.adversarial_loss is not None:
            self.grad_adversarial_loss_op = tf.gradients(self.adversarial_loss, self.params)
        

    def get_vec_to_list_fn(self):
        params_val = self.sess.run(self.params)
        self.num_params = len(np.concatenate(params_val))        
        print('Total number of parameters: %s' % self.num_params)


        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in params_val:
                return_list.append(v[cur_pos : cur_pos+len(p)])
                cur_pos += len(p)

            assert cur_pos == len(v)
            return return_list

        return vec_to_list


    def reset_datasets(self):
        for data_set in self.data_sets:
            if data_set is not None:
                data_set.reset_batch()


    def fill_feed_dict_with_all_ex(self, data_set):
        feed_dict = {
            self.input_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels
        }
        return feed_dict


    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove):
        num_examples = data_set.x.shape[0]
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False
        feed_dict = {
            self.input_placeholder: data_set.x[idx, :],
            self.labels_placeholder: data_set.labels[idx]
        }
        return feed_dict


    def fill_feed_dict_with_batch(self, data_set, batch_size=0):
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size
    
        input_feed, labels_feed = data_set.next_batch(batch_size)                              
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,            
        }
        return feed_dict


    def fill_feed_dict_with_some_ex(self, data_set, target_indices):
        input_feed = data_set.x[target_indices, :].reshape(len(target_indices), -1)
        labels_feed = data_set.labels[target_indices].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict


    def fill_feed_dict_with_one_ex(self, data_set, target_idx):
        input_feed = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict


    def fill_feed_dict_manual(self, X, Y):
        X = np.array(X)
        Y = np.array(Y) 
        input_feed = X.reshape(len(Y), -1)
        labels_feed = Y.reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict        


    def minibatch_mean_eval(self, ops, data_set):
        
        num_examples = data_set.num_examples
        print(num_examples, self.batch_size)
        assert num_examples % self.batch_size == 0
        num_iter = int(num_examples / self.batch_size)

        self.reset_datasets()

        ret = []
        for i in xrange(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(data_set)
            ret_temp = self.sess.run(ops, feed_dict=feed_dict)
            
            if len(ret)==0:
                for b in ret_temp:
                    if isinstance(b, list):
                        ret.append([c / float(num_iter) for c in b])
                    else:
                        ret.append([b / float(num_iter)])
            else:
                for counter, b in enumerate(ret_temp):
                    if isinstance(b, list):
                        ret[counter] = [a + (c / float(num_iter)) for (a, c) in zip(ret[counter], b)]
                    else:
                        ret[counter] += (b / float(num_iter))
            
        return ret


    def print_model_eval(self, no_print=False):
        params_val = self.sess.run(self.params)
        ori = self.mini_batch
        self.mini_batch = False
        if self.mini_batch == True:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val = self.minibatch_mean_eval(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss, self.accuracy_op],
                self.data_sets.train)
            
            test_loss_val, test_acc_val = self.minibatch_mean_eval(
                [self.loss_no_reg, self.accuracy_op],
                self.data_sets.test)

        else:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val, ans = self.sess.run(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss, self.accuracy_op, self.logits], 
                feed_dict=self.all_train_feed_dict)

            test_loss_val, test_acc_val = self.sess.run(
                [self.loss_no_reg, self.accuracy_op], 
                feed_dict=self.all_test_feed_dict)
            
        pred = np.argmax(ans, axis=1)
        # import ipdb; ipdb.set_trace()
        # print(np.sum(pred), pred.shape, "hello")#, len(np.where(pred)[0].tolist()))    # print the nos of 1's in the list

        self.mini_batch = ori       # whatever the original value was

        if not no_print:
            print('Train loss (w reg) on all data: %s' % loss_val)
            print('Train loss (w/o reg) on all data: %s' % loss_no_reg_val)

            print('Test loss (w/o reg) on all data: %s' % test_loss_val)
            print('Train acc on all data:  %s' % train_acc_val)
            print('Test acc on all data:   %s' % test_acc_val)

            print('Norm of the mean of gradients: %s' % np.linalg.norm(np.concatenate(grad_loss_val)))
            print('Norm of the params: %s' % np.linalg.norm(np.concatenate(params_val)))
        return train_acc_val, test_acc_val


    def retrain(self, num_steps, feed_dict):        
        for step in xrange(num_steps):   
            self.sess.run(self.train_op, feed_dict=feed_dict)


    def update_learning_rate(self, step):
        # assert self.num_train_examples % self.batch_size == 0
        num_steps_in_epoch = self.num_train_examples / self.batch_size
        # epoch = step // num_steps_in_epoch
        # print(epoch, self.decay_epochs, "SEE")
        multiplier = 1
        if step < self.decay_epochs[0]:
            multiplier = 1
        elif step < self.decay_epochs[1]:
            # if step < self.decay_epochs[0] + 500:
                # print(step, "THERE")
            multiplier = 0.1
        else:
            # if step < self.decay_epochs[1] + 500:
                # print(step, "HERE")
            multiplier = 0.01
        
        # assert(multiplier == 1)
        self.sess.run(
            self.update_learning_rate_op, 
            feed_dict={self.learning_rate_placeholder: multiplier * self.initial_learning_rate})        


    def train(self, num_steps, 
              iter_to_switch_to_batch=20000, 
              iter_to_switch_to_sgd=40000,
              save_checkpoints=True, verbose=True, plot_loss=False):
        """
        Trains a model for a specified number of steps.
        """
        if verbose: print('Training for %s steps' % num_steps)

        sess = self.sess            

        for step in xrange(num_steps):
            self.update_learning_rate(step)

            start_time = time.time()

            if step < iter_to_switch_to_batch:                
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train)
                _, loss_val, summary = sess.run([self.train_op, self.total_loss, self.write_op], feed_dict=feed_dict)
                
            elif step < iter_to_switch_to_sgd:
                feed_dict = self.all_train_feed_dict          
                _, loss_val, summary = sess.run([self.train_op, self.total_loss, self.write_op], feed_dict=feed_dict)

            else: 
                feed_dict = self.all_train_feed_dict          
                _, loss_val, summary = sess.run([self.train_sgd_op, self.total_loss, self.write_op], feed_dict=feed_dict)          

            duration = time.time() - start_time
            
            # summary = sess.run(write_op, {log_var: random.rand()})
            if plot_loss:
                self.summary_writer.add_summary(summary, step)
                self.summary_writer.flush()

            if verbose:
                if step % 1000 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.8f (%.3f sec)' % (step, loss_val, duration))
                    print("Accuracies: ", self.print_model_eval())
            else:
                if step % 10000 == 0:
                    print(step)
                    # _, _ = self.print_model_eval()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == num_steps:
                if save_checkpoints: self.saver.save(sess, self.checkpoint_file, global_step=step)
                if verbose: self.print_model_eval()
                # print("Learning rate: ", )


    def load_checkpoint(self, iter_to_load, do_checks=True):
        checkpoint_to_load = "%s-%s" % (self.checkpoint_file, iter_to_load) 
        self.saver.restore(self.sess, checkpoint_to_load)

        if do_checks:
            # print('Model %s loaded. Sanity checks ---' % checkpoint_to_load)
            return self.print_model_eval(no_print=True)


    def find_discm_examples(self, class0_data, class1_data, print_file, scheme):
        # import ipdb; ipdb.set_trace()
        length = class0_data.shape[0]
        assert length == class1_data.shape[0]

        l_zero = np.zeros(length, dtype=np.int)
        l_one = np.ones(length, dtype=np.int)
        
        feed_dict_class0_label0 = self.fill_feed_dict_manual(class0_data, l_zero)
        feed_dict_class0_label1 = self.fill_feed_dict_manual(class0_data, l_one)
        feed_dict_class1_label0 = self.fill_feed_dict_manual(class1_data, l_zero)
        feed_dict_class1_label1 = self.fill_feed_dict_manual(class1_data, l_one)
        
        ops = [self.preds, self.indiv_loss_no_reg]
        
        predictions_class0_, loss_class0_label_0 = self.sess.run(ops, feed_dict=feed_dict_class0_label0)
        predictions_class0, loss_class0_label_1 = self.sess.run(ops, feed_dict=feed_dict_class0_label1)
        assert (predictions_class0_ == predictions_class0).all()    #"""This should hold"""
        predictions_class1_, loss_class1_label_0 = self.sess.run(ops, feed_dict=feed_dict_class1_label0)
        predictions_class1, loss_class1_label_1 = self.sess.run(ops, feed_dict=feed_dict_class1_label1)
        assert (predictions_class1_ == predictions_class1).all()    #"""This should hold"""
        
        # for german credit dataset
        if "german" in self.model_name or "student" in self.model_name:
            loss_class0_label_0 = loss_class0_label_0
            loss_class0_label_1 = loss_class0_label_1
            loss_class1_label_1 = loss_class1_label_1
            loss_class1_label_0 = loss_class1_label_0

        # for adult income
        elif "adult" in self.model_name or "compas" in self.model_name or "default" in self.model_name:
            loss_class0_label_0 = loss_class0_label_0[0]                                                            
            loss_class0_label_1 = loss_class0_label_1[0]                                                            
            loss_class1_label_1 = loss_class1_label_1[0]                                                            
            loss_class1_label_0 = loss_class1_label_0[0]

        else:
            assert False

        num_discriminating = sum(predictions_class0 != predictions_class1)    # Gives the number of discriminating examples
        print("Number of discriminating examples: ", num_discriminating)

        if not print_file:
            return num_discriminating
        
        # import ipdb; ipdb.set_trace()
        # idx1 = np.where(predictions_class0 != predictions_class1)[0]     # find points where predictions are not equal for the two demographic groups
        # idx2 = np.where(predictions_class0 == 0)[0]
        # idx = list(set(idx1).intersection(idx2))            # find only points which are responsible for assigning 0 to class0, not 0 to class1 
        idx = np.where(predictions_class0 != predictions_class1)[0]
        discm_class0 = class0_data[idx]     # so discm_class0, discm_class1 are the vectors that only
        discm_class1 = class1_data[idx]     # differ in sensitive attribute
        write = False
        if write:
            with open("discriminating_tests_german.csv", "w") as f:
                for x, y in zip(discm_class0, discm_class1):
                    x = str(tuple(x))[1:-1]
                    y = str(tuple(y))[1:-1]
                    z = (x, y)
                    f.write(str(z) + "\n")

        # zero labels if both the data-points in the discriminating pair is labelled as 0
        zero_labels_loss = loss_class0_label_0[idx] + loss_class1_label_0[idx]
        # one labels if both the data-points in the discriminating pair is labelled as 1
        ones_labels_loss = loss_class0_label_1[idx] + loss_class1_label_1[idx]
        
        # keep the label which produces lower loss for each pair of discriminating tests. 
        # selection of the labels is due to the sum of losses on the pair of discriminating tests, 
        # but the final ranking should be only based on the loss of the one data-point 
        # and the model's prediction for it. Therfore no use of lower_loss_labelling
        lower_loss = list(map(lambda x: x[1] if x[0] > x[1] else x[0], zip(zero_labels_loss, ones_labels_loss)))
        
        desired_labels = list(map(lambda x: 1 if x[0] > x[1] else 0, zip(zero_labels_loss, ones_labels_loss)))  # label the pair with the one that produces lower loss
        actual_predictions = list(map(lambda x: 0 if x == 1 else 1, desired_labels))      # """Just the inverse of desired_labels. This is by definition of causal/individual discrimination"""

        # assert(scheme == 8)
        # print(len(actual_predictions), len(zero_labels_loss), len(ones_labels_loss))
        if scheme == 1:
            X_discm, Y_discm = [], []
            with open("scheme1_labelled_generated_tests.csv", "w") as f:
                f.write("Checking-ccount,Months,Credit-history,Purpose,Credit-mount,Savings-ccount,Present-employment-since,Instllment-rte,Gender,Other-debtors,Present-residence-since,Property,age,Other-instllment-plns,Housing,Number-of-existing-credits,Job,Number-of-people-being-lible,Telephone,Foreign-worker, Final-label\n")
                for dt0, dt1, label in zip(discm_class0, discm_class1, desired_labels):
                    f.write(str(dt0.tolist())[1:-1] + ", " + str(label) + "\n")
                    f.write(str(dt1.tolist())[1:-1] + ", " + str(label) + "\n")
                    X_discm.append(dt0)
                    X_discm.append(dt1)
                    Y_discm.append(label)
                    Y_discm.append(label)
            
            X_discm = np.array(X_discm)
            Y_discm = np.array(Y_discm)
            self.discm_data_set = DataSet(X_discm, Y_discm)
        
        elif scheme == 8:
            l00 = loss_class0_label_0[idx]   # class0 refers to the senstive attribute value == 0
            l10 = loss_class1_label_0[idx]   # class1 refers to the senstive attribute value == 1
            l11 = loss_class1_label_1[idx]
            l01 = loss_class0_label_1[idx]

            # for the data point whose prediction == label, return its complement data-point (gender flipped) and its gender in a tuple
            which_data_point = list(map(lambda x: (x[3], 1) if int(x[0]) == int(x[1]) else (x[2], 0), zip(predictions_class0[idx], desired_labels, discm_class0, discm_class1)))

            gender = [i[1] for i in which_data_point]
            which_data_point = [i[0] for i in which_data_point]

            losses_at_this_point = list(map(lambda x: x[2] if x[0] == 0 and x[1] == 0 else (x[3] if x[0] == 0 and x[1] == 1 else (x[4] if x[0] == 1 and x[1] == 0 else x[5])) , zip(gender, actual_predictions, l00, l01, l10, l11)))

            # arg_ = np.argsort(loss_labelling).tolist()
            arg_ = np.argsort(losses_at_this_point).tolist()[::-1]       # decreasing order of loss as the point with highest loss is easiest to flip prediction
            actual_predictions_sorted = [actual_predictions[i] for i in arg_]
            which_data_point_sorted = [which_data_point[i] for i in arg_]
            
            # this should not be desired labels as the prediction of the trained model is not the desired label and we want the point responsible for the current prediction of the trained model
            # with open("scheme8_labelled_generated_tests.csv", "w") as f:
            #     for dt, la in zip(which_data_point_sorted, desired_labels_sorted):
            #         f.write(str(dt.astype(int).tolist())[1:-1] + ", " + str(la) + "\n")       # don't take the scaled and modified input
            
            X_discm, Y_discm = [], []
            write = False
            # if write
            with open("scheme8_labelled_generated_tests.csv", "w") as f:
                # f.write("Checking-ccount,Months,Credit-history,Purpose,Credit-mount,Savings-ccount,Present-employment-since,Instllment-rte,Gender,Other-debtors,Present-residence-since,Property,age,Other-instllment-plns,Housing,Number-of-existing-credits,Job,Number-of-people-being-lible,Telephone,Foreign-worker, Final-label\n")
                for dt, label in zip(which_data_point_sorted, actual_predictions_sorted):
                    # f.write(str(dt.tolist())[1:-1] + ", " + str(label) + "\n")
                    X_discm.append(dt)
                    Y_discm.append(label)
            
            X_discm = np.array(X_discm)
            Y_discm = np.array(Y_discm)
            self.discm_data_set = DataSet(X_discm, Y_discm)

        elif scheme == 9:
            # l00 = loss_class0_label_0[idx]   # class0 refers to the senstive attribute value == 0
            # l10 = loss_class1_label_0[idx]   # class1 refers to the senstive attribute value == 1
            # l11 = loss_class1_label_1[idx]
            # l01 = loss_class0_label_1[idx]

            # for the data point whose prediction == label, return its complement data-point (gender flipped) and its gender in a tuple
            # which_data_point = list(map(lambda x: (x[3], 1) if int(x[0]) == int(x[1]) else (x[2], 0), zip(predictions_class0[idx], desired_labels, discm_class0, discm_class1)))

            # gender = [i[1] for i in which_data_point]
            # which_data_point = [i[0] for i in which_data_point]

            # losses_at_this_point = list(map(lambda x: x[2] if x[0] == 0 and x[1] == 0 else (x[3] if x[0] == 0 and x[1] == 1 else (x[4] if x[0] == 1 and x[1] == 0 else x[5])) , zip(gender, actual_predictions, l00, l01, l10, l11)))

            # arg_ = np.argsort(loss_labelling).tolist()
            # arg_ = np.argsort(losses_at_this_point).tolist()[::-1]       # decreasing order of loss as the point with highest loss is easiest to flip prediction
            # actual_predictions_sorted = [actual_predictions[i] for i in arg_]
            # which_data_point_sorted = [which_data_point[i] for i in arg_]
            
            # These two will be used in X_discm
            # discm_class0 = class0_data[idx]     # so discm_class0, discm_class1 are the vectors that only
            # discm_class1 = class1_data[idx]
            predictions_class0 = predictions_class0[idx]
            predictions_class1 = predictions_class1[idx]
            assert(len(predictions_class0) == len(predictions_class1) == len(discm_class0) == len(discm_class1))
            X_discm, Y_discm = [], []
            for dt, label in zip(discm_class0, predictions_class0):
                X_discm.append(dt)
                Y_discm.append(label)
            
            for dt, label in zip(discm_class1, predictions_class1):
                X_discm.append(dt)
                Y_discm.append(label)
                
            X_discm = np.array(X_discm)
            Y_discm = np.array(Y_discm)
            self.discm_data_set = DataSet(X_discm, Y_discm)
 
        
        else:
            raise NotImplementedError
        
        self.mini_batch = False
        # return discm_class0, discm_class1, desired_labels


    def get_train_op(self, total_loss, global_step, learning_rate):
        """
        Return train_op
        """
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op


    def get_train_sgd_op(self, total_loss, global_step, learning_rate=0.001):
        """
        Return train_sgd_op
        """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op


    def get_accuracy_op(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """        
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32)) / tf.shape(labels)[0]


    def loss_per_instance(self):
        ops = self.indiv_loss_no_reg
        loss_each_training_points = self.sess.run(ops, feed_dict=self.all_train_feed_dict)
        if "german" in self.model_name or "student" in self.model_name:
            return loss_each_training_points

        # for adult income
        elif "adult" in self.model_name or "compas" in self.model_name or "default" in self.model_name:
            return loss_each_training_points[0]

        else:
            assert False


    def loss(self, logits, labels):
        labels = tf.one_hot(labels, depth=self.num_classes, dtype=tf.float32)
        """This returns a tensor of the shape labels (tantamount to a vector of size equal to one mini-batch)"""
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # cross_entropy = -tf.reduce_sum(tf.multiply(labels, tf.nn.log_softmax(logits)), reduction_indices=1)
        # for adult income dataset and compas dataset
        if "adult" in self.model_name:
            ratio = 11200*1.36 / 45200       # nos of one's by total dataset size, multiply by 1.2 for control over loss
            class_weight = tf.constant([[ratio, 1.0 - ratio]])
            weight_per_label = tf.transpose(tf.matmul(labels, tf.transpose(class_weight)) )
            xent = tf.multiply(weight_per_label, cross_entropy, name='xent')
            indiv_loss_no_reg = xent
            loss_no_reg = tf.reduce_mean(xent, name='xentropy_mean')
        

        elif "compas" in self.model_name:
            ratio = 3768*0.94 / 9884       # nos of one's by total dataset size, multiply by 0.94 for control over loss
            class_weight = tf.constant([[ratio, 1.0 - ratio]])
            weight_per_label = tf.transpose(tf.matmul(labels, tf.transpose(class_weight)) )
            xent = tf.multiply(weight_per_label, cross_entropy, name='xent')
            indiv_loss_no_reg = xent
            loss_no_reg = tf.reduce_mean(xent, name='xentropy_mean')

        
        elif "default" in self.model_name:
            ratio = 6636*1.09 / 30000       # nos of one's by total dataset size, multiply by 1.09 for control over loss
            class_weight = tf.constant([[ratio, 1.0 - ratio]])
            weight_per_label = tf.transpose(tf.matmul(labels, tf.transpose(class_weight)) )
            xent = tf.multiply(weight_per_label, cross_entropy, name='xent')
            indiv_loss_no_reg = xent
            loss_no_reg = tf.reduce_mean(xent, name='xentropy_mean')
        
        # for german credit dataset
        elif "german" in self.model_name or "student" in self.model_name:
            indiv_loss_no_reg = cross_entropy
            loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        
        tf.add_to_collection('losses', loss_no_reg)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return total_loss, loss_no_reg, indiv_loss_no_reg


    def adversarial_loss(self, logits, labels):
        # Computes sum of log(1 - p(y = true|x))
        # No regularization (because this is meant to be computed on the test data)

        labels = tf.one_hot(labels, depth=self.num_classes)        
        wrong_labels = (labels - 1) * -1 # Flips 0s and 1s
        wrong_labels_bool = tf.reshape(tf.cast(wrong_labels, tf.bool), [-1, self.num_classes])

        wrong_logits = tf.reshape(tf.boolean_mask(logits, wrong_labels_bool), [-1, self.num_classes - 1])
        
        indiv_adversarial_loss = tf.reduce_logsumexp(wrong_logits, reduction_indices=1) - tf.reduce_logsumexp(logits, reduction_indices=1)
        adversarial_loss = tf.reduce_mean(indiv_adversarial_loss)
        
        return adversarial_loss, indiv_adversarial_loss #, indiv_wrong_prob


    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            feed_dict[pl_block] = vec_block        
        return feed_dict


    def get_inverse_hvp(self, v, approx_type='cg', approx_params=None, verbose=True):
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            return self.get_inverse_hvp_lissa(v, **approx_params)
        elif approx_type == 'cg':
            return self.get_inverse_hvp_cg(v, verbose)


    def get_inverse_hvp_lissa(self, v, 
                              batch_size=None,
                              scale=10, damping=0.0, num_samples=1, recursion_depth=10000):
        """
        This uses mini-batching; uncomment code for the single sample case.
        """    
        inverse_hvp = None
        print_iter = recursion_depth / 10

        for i in range(num_samples):
            # samples = np.random.choice(self.num_train_examples, size=recursion_depth)
           
            cur_estimate = v

            for j in range(recursion_depth):
             
                # feed_dict = fill_feed_dict_with_one_ex(
                #   data_set, 
                #   images_placeholder, 
                #   labels_placeholder, 
                #   samples[j])   
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=batch_size)

                feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)
                hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
                cur_estimate = [a + (1-damping) * b - c/scale for (a,b,c) in zip(v, cur_estimate, hessian_vector_val)]    

                # Update: v + (I - Hessian_at_x) * cur_estimate
                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    print("Recursion at depth %s: norm is %.8lf" % (j, np.linalg.norm(np.concatenate(cur_estimate))))
                    feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)

            if inverse_hvp is None:
                inverse_hvp = [b/scale for b in cur_estimate]
            else:
                inverse_hvp = [a + b/scale for (a, b) in zip(inverse_hvp, cur_estimate)]  

        inverse_hvp = [a/num_samples for a in inverse_hvp]
        return inverse_hvp
  
    
    def minibatch_hessian_vector_val(self, v):

        num_examples = self.num_train_examples
        if self.mini_batch == True:
            batch_size = 100
            assert num_examples % batch_size == 0
        else:
            batch_size = self.num_train_examples

        num_iter = int(num_examples / batch_size)

        self.reset_datasets()
        hessian_vector_val = None
        for i in xrange(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=batch_size)
            # Can optimize this
            feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, v)
            hessian_vector_val_temp = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
            if hessian_vector_val is None:
                hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + (b / float(num_iter)) for (a,b) in zip(hessian_vector_val, hessian_vector_val_temp)]
            
        hessian_vector_val = [a + self.damping * b for (a,b) in zip(hessian_vector_val, v)]

        return hessian_vector_val


    def get_fmin_loss_fn(self, v):

        def get_fmin_loss(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)
        return get_fmin_loss


    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
            
            return np.concatenate(hessian_vector_val) - np.concatenate(v)
        return get_fmin_grad


    def get_fmin_hvp(self, x, p):
        hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(p))

        return np.concatenate(hessian_vector_val)


    def get_cg_callback(self, v, verbose):
        self.hvp_iterations = 0
        def cg_callback(x):
            # nonlocal iteration
            self.hvp_iterations +=1
            print(self.hvp_iterations, "Iteration completed")

        return cg_callback

    
    def get_inverse_hvp_cg(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        fmin_grad_fn = self.get_fmin_grad_fn(v)
        cg_callback = self.get_cg_callback(v, verbose)

        fmin_results_all = fmin_ncg(
            f=fmin_loss_fn,
            x0=np.concatenate(v),
            fprime=fmin_grad_fn,
            fhess_p=self.get_fmin_hvp,
            callback=cg_callback,
            avextol=1e-8,
            maxiter=100,
            full_output=True) 

        fmin_results = fmin_results_all[0]
        warning = fmin_results_all[-1]
        assert(isinstance(warning, int))
        # print(warning, "hello")
        if warning == 1 or warning == 3:
            with open("bad_adult_models.txt", "a") as f:
                f.write(f"{self.model_name}\n")
            assert False
        # print(self.model_name, "see")
        return self.vec_to_list(fmin_results)


    def get_test_grad_loss_no_reg_val(self, test_indices, batch_size=100, loss_type='normal_loss'):

        if loss_type == 'normal_loss':
            op = self.grad_loss_no_reg_op
        elif loss_type == 'adversarial_loss':
            op = self.grad_adversarial_loss_op
        else:
            raise(ValueError, 'Loss must be specified')

        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / batch_size))

            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i+1) * batch_size, len(test_indices)))
                test_feed_dict = self.fill_feed_dict_with_some_ex(self.discm_data_set, test_indices[start:end])
                # test_feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.test, test_indices[start:end])

                temp = self.sess.run(op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end-start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end-start) for (a, b) in zip(test_grad_loss_no_reg_val, temp)]

            test_grad_loss_no_reg_val = [a/len(test_indices) for a in test_grad_loss_no_reg_val]

        else:
            test_grad_loss_no_reg_val = self.minibatch_mean_eval([op], self.discm_data_set)[0]
        
        return test_grad_loss_no_reg_val


    def get_influence_on_test_loss(self, test_indices, train_idx, 
        approx_type='cg', approx_params=None, force_refresh=True, test_description=None,
        loss_type='normal_loss',
        X=None, Y=None):
        # If train_idx is None then use X and Y (phantom points)
        # Need to make sure test_idx stays consistent between models
        # because mini-batching permutes dataset order

        if train_idx is None: 
            if (X is None) or (Y is None): raise(ValueError, 'X and Y must be specified if using phantom points.')
            if X.shape[0] != len(Y): raise(ValueError, 'X and Y must have the same length.')
        else:
            if (X is not None) or (Y is not None): raise(ValueError, 'X and Y cannot be specified if train_idx is specified.')

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)

        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            if len(test_indices) < 10:
                test_description = test_indices
            elif len(test_indices) == self.discm_data_set.num_examples:
                test_description = "all"
            else:
                assert False

        # creates directory if it doesn't exist
        Path(self.hvp_files).mkdir(parents=True, exist_ok=True)
        approx_filename = os.path.join(self.hvp_files, '%s-schm%s-%s-%s-test-%s.npz' % (self.model_name, self.scheme_name, approx_type, loss_type, test_description))
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params)
            
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            print('Saved inverse HVP to %s' % approx_filename)

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)
        # exit(0)
        start_time = time.time()
        if train_idx is None:
            num_to_remove = len(Y)
            predicted_loss_diffs = np.zeros([num_to_remove])            
            for counter in np.arange(num_to_remove):
                single_train_feed_dict = self.fill_feed_dict_manual(X[counter, :], [Y[counter]])      
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples            

        else:            
            num_to_remove = len(train_idx)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter, idx_to_remove in enumerate(train_idx):            
                single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)      
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / self.num_train_examples
                
        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return predicted_loss_diffs


    def find_eigvals_of_hessian(self, num_iter=100, num_prints=10):

        # Setup        
        print_iterations = num_iter / num_prints
        feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, 0)

        # Initialize starting vector
        grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=feed_dict)
        initial_v = []

        for a in grad_loss_val:
            initial_v.append(np.random.random(a.shape))        
        initial_v, _ = normalize_vector(initial_v)

        # Do power iteration to find largest eigenvalue
        print('Starting power iteration to find largest eigenvalue...')

        largest_eig = norm_val
        print('Largest eigenvalue is %s' % largest_eig)

        # Do power iteration to find smallest eigenvalue
        print('Starting power iteration to find smallest eigenvalue...')
        cur_estimate = initial_v
        
        for i in range(num_iter):          
            cur_estimate, norm_val = normalize_vector(cur_estimate)
            hessian_vector_val = self.minibatch_hessian_vector_val(cur_estimate)
            new_cur_estimate = [a - largest_eig * b for (a,b) in zip(hessian_vector_val, cur_estimate)]

            if i % print_iterations == 0:
                print(-norm_val + largest_eig)
                dotp = np.dot(np.concatenate(new_cur_estimate), np.concatenate(cur_estimate))
                print("dot: %s" % dotp)
            cur_estimate = new_cur_estimate

        smallest_eig = -norm_val + largest_eig
        assert dotp < 0, "Eigenvalue calc failed to find largest eigenvalue"

        print('Largest eigenvalue is %s' % largest_eig)
        print('Smallest eigenvalue is %s' % smallest_eig)
        return largest_eig, smallest_eig


    def get_grad_of_influence_wrt_input(self, train_indices, test_indices, 
        approx_type='cg', approx_params=None, force_refresh=True, verbose=True, test_description=None,
        loss_type='normal_loss'):
        """
        If the loss goes up when you remove a point, then it was a helpful point.
        So positive influence = helpful.
        If we move in the direction of the gradient, we make the influence even more positive, 
        so even more helpful.
        Thus if we want to make the test point more wrong, we have to move in the opposite direction.
        """

        # Calculate v_placeholder (gradient of loss at test point)
        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)            

        if verbose: print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))
        
        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
        
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            if verbose: print('Loaded inverse HVP from %s' % approx_filename)
        else:            
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params,
                verbose=verbose)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            if verbose: print('Saved inverse HVP to %s' % approx_filename)            
        
        duration = time.time() - start_time
        if verbose: print('Inverse HVP took %s sec' % duration)

        grad_influence_wrt_input_val = None

        for counter, train_idx in enumerate(train_indices):
            # Put in the train example in the feed dict
            grad_influence_feed_dict = self.fill_feed_dict_with_one_ex(
                self.data_sets.train,  
                train_idx)

            self.update_feed_dict_with_v_placeholder(grad_influence_feed_dict, inverse_hvp)

            # Run the grad op with the feed dict
            current_grad_influence_wrt_input_val = self.sess.run(self.grad_influence_wrt_input_op, feed_dict=grad_influence_feed_dict)[0][0, :]            
            
            if grad_influence_wrt_input_val is None:
                grad_influence_wrt_input_val = np.zeros([len(train_indices), len(current_grad_influence_wrt_input_val)])

            grad_influence_wrt_input_val[counter, :] = current_grad_influence_wrt_input_val

        return grad_influence_wrt_input_val


    def update_train_x(self, new_train_x):
        assert np.all(new_train_x.shape == self.data_sets.train.x.shape)
        new_train = DataSet(new_train_x, np.copy(self.data_sets.train.labels))
        self.data_sets = base.Datasets(train=new_train, validation=self.data_sets.validation, test=self.data_sets.test)
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)                
        self.reset_datasets()


    def update_train_x_y(self, new_train_x, new_train_y):
        new_train = DataSet(new_train_x, new_train_y)
        self.data_sets = base.Datasets(train=new_train, validation=self.data_sets.validation, test=self.data_sets.test)
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)                
        self.num_train_examples = len(new_train_y)
        self.reset_datasets()        


    def update_test_x_y(self, new_test_x, new_test_y):
        new_test = DataSet(new_test_x, new_test_y)
        self.data_sets = base.Datasets(train=self.data_sets.train, validation=self.data_sets.validation, test=new_test)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test)                
        self.num_test_examples = len(new_test_y)
        self.reset_datasets()        

