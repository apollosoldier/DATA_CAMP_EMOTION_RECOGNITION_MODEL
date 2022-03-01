'''
Author: Mohamed Traore
Description: this file will store all the useful functions that can be used
	in machine learning algorithms in order to not repeat them several
	times if they are used in different algorithms
Functions:
	bootstrap_oob
'''

# Libraries
import numpy as np
import pandas as pd
import scipy.stats


def boostrap_oob(df_input):
    """
    :param df_input: Df which contains as columns features and y
    :return boostrap_df and and oob_df
    """
    bootstrap = df_input.sample(len(df_input.index), replace=True)
    oob_index = [x for x in df_input.index if x not in bootstrap.index]
    oob = df_input.iloc[oob_index]

    return bootstrap, oob


def most_frequent_classes(X):
    """

    :param X:
    :return:
    """
    print("type_most_frequent_classes:",type(X))
    (classes, counts) = np.unique(X, return_counts=True)
    index = np.argmax(counts)
    print("return most_fr_cls:",classes[index])
    return classes[index]

def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance