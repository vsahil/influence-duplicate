import numpy as np
import copy


def rescale_input_numpy(inp):
    assert(inp.shape[1] == 20)  # 20 features for german credit dataset
    means_and_ranges = [(1.001, 3), (20.903, 68), (32.545, 4), (47.148, 370), (3271.258, 18174), (1.19, 4), (3.384, 4), (2.973, 3), (0.69, 1), (101.145, 2), (2.845, 3), (122.358, 3), (35.546, 56), (142.675, 2), (151.929, 2), (1.407, 3), (172.904, 3), (1.155, 1), (1.404, 1), (1.037, 1)]
    r = np.arange(20)
    out = copy.deepcopy(inp)
    for col, (mean, range_) in zip(r, means_and_ranges):
        out[:, col] = np.divide(np.subtract(out[:, col], mean), range_)
    return out


def entire_test_suite(mini=True):
    gender0 = "gender0"
    gender1 = "gender1"
    if mini:
        gender0 += "_mini"
        gender1 += "_mini"
    class0_ = np.genfromtxt("../german-credit-dataset/{}.csv".format(gender0), delimiter=",")
    class1_ = np.genfromtxt("../german-credit-dataset/{}.csv".format(gender1), delimiter=",")
    class0 = rescale_input_numpy(class0_)
    class1 = rescale_input_numpy(class1_)

    return class0, class1
    
    # length = class0.shape[0]
    # l_zero = np.eye(2)[np.zeros(length, dtype=int)]     # list of zero labels
    # l_one = np.eye(2)[np.ones(length, dtype=int)]      # list of one labels


    with tf.Session() as sess:
        out0, loss_class0_label_0 = sess.run([logits, loss_no_reg], feed_dict={X: class0, Y:l_zero})
        out0, loss_class0_label_1 = sess.run([logits, loss_no_reg], feed_dict={X: class0, Y:l_one})
        
        out1, loss_class1_label_0 = sess.run([logits, loss_no_reg], feed_dict={X: class1, Y:l_zero})
        out1, loss_class1_label_1 = sess.run([logits, loss_no_reg], feed_dict={X: class1, Y:l_one})
    
        out0_ = np.argmax(out0, axis = 1)
        out1_ = np.argmax(out1, axis = 1)
        # print("hello")
        print(sum(out0_ != out1_))
        
        idx = np.where(out0_ != out1_)[0]
        g0 = class0_[idx]
        g1 = class1_[idx]
        write = False
        if write:
            with open("tests_german_tf.csv", "w") as f:
                for x, y in zip(g0, g1):
                    x = str(tuple(x))[1:-1]
                    y = str(tuple(y))[1:-1]
                    z = (x, y)
                    f.write(str(z) + "\n")
        
        # selection of the labels is due to the sum of losses of the pair, 
        # but the final ranking should be only based on the loss of the data-point which is changing prediction
        # and the model's prediction for it
        # import ipdb; ipdb.set_trace()
        zero_labels_loss = loss_class1_label_0[idx] + loss_class0_label_0[idx]
        ones_labels_loss = loss_class1_label_1[idx] + loss_class0_label_1[idx]
        # keep the lower loss for each pair of discriminating test. 
        loss_labelling = list(map(lambda x: x[1] if x[0] > x[1] else x[0], zip(zero_labels_loss, ones_labels_loss)))
        desired_labels = list(map(lambda x: 1 if x[0] > x[1] else 0, zip(zero_labels_loss, ones_labels_loss)))
        undesired_actual_labels = list(map(lambda x: 0 if x == 1 else 1, desired_labels))      # just the inverse of it

        l00 = loss_class0_label_0[idx]
        l10 = loss_class1_label_0[idx]
        l11 = loss_class1_label_1[idx]
        l01 = loss_class0_label_1[idx]

        # for the data point whose prediction == label, choose its complement data-point (gender flipped)
        which_data_point = list(map(lambda x: (x[3], 1) if int(x[0]) == int(x[1]) else (x[2], 0), zip(out0_[idx], desired_labels, g0, g1)))
        # import ipdb; ipdb.set_trace()
        gender = [i[1] for i in which_data_point]
        which_data_point = [i[0] for i in which_data_point]

        losses_at_this_point = list(map(lambda x: x[2] if x[0] == 0 and x[1] == 0 else (x[3] if x[0] == 0 and x[1] == 1 else (x[4] if x[0] == 1 and x[1] == 0 else x[5])) , zip(gender, undesired_actual_labels, l00, l01, l10, l11)))

        # arg_ = np.argsort(loss_labelling).tolist()
        arg_ = np.argsort(losses_at_this_point).tolist()
        undesired_actual_labels_sorted = [undesired_actual_labels[i] for i in arg_]
        which_data_point_sorted = [which_data_point[i] for i in arg_]
        
        # this should not be desired labels as the prediction of the model is not the desired label and we want the point responsible for that
        # with open("scheme8_labelled_generated_tests.csv", "w") as f:
        #     for dt, la in zip(which_data_point_sorted, desired_labels_sorted):
        #         f.write(str(dt.astype(int).tolist())[1:-1] + ", " + str(la) + "\n")       # don't take the scaled and modified input

        with open("scheme8_labelled_generated_tests.csv", "w") as f:
            f.write("Checking-ccount,Months,Credit-history,Purpose,Credit-mount,Savings-ccount,Present-employment-since,Instllment-rte,Gender,Other-debtors,Present-residence-since,Property,age,Other-instllment-plns,Housing,Number-of-existing-credits,Job,Number-of-people-being-lible,Telephone,Foreign-worker, Final-label\n")
            for dt, la in zip(which_data_point_sorted, undesired_actual_labels_sorted):
                f.write(str(dt.astype(int).tolist())[1:-1] + ", " + str(la) + "\n")


