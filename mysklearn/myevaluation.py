"""myevaluation.py

@author aphollier
"""
import math
import numpy as np # use numpy's random number generation

from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    if test_size >= 1:
        test_split = test_size
    else:
        test_split = int(math.ceil(len(X) * test_size))
    rng = np.random.default_rng(random_state)
    if shuffle:
        rng.shuffle(X)
        rng.shuffle(y)
    X_train = X[:len(X)-test_split]
    X_test = X[len(X)-test_split:]
    y_train = y[:len(y)-test_split]
    y_test = y[len(y)-test_split:]

    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    indices = list(range(len(X)))
    first_size = len(X) // n_splits + 1
    sec_size = len(X) // n_splits
    first_splits = len(X) % n_splits
    sec_start = first_size * first_splits
    folds = []
    if shuffle:
        myutils.randomize_in_place(indices, random_state)
    for i in range(first_splits):
        folds.append((indices[:first_size*i] + indices[first_size*i+first_size:],\
             indices[first_size*i:first_size*i+first_size]))
    for i in range(n_splits - first_splits):
        folds.append((indices[:sec_size*i + sec_start] + indices[sec_size*i+sec_size+sec_start:],\
             indices[sec_size*i+sec_start:sec_size*i+sec_size+sec_start]))

    return folds

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    indices = list(range(len(X)))
    y_splits = {}
    first_splits = len(X) % n_splits
    folds = []
    if shuffle:
        myutils.randomize_in_place(indices, random_state)
    for i in indices:
        if y[i] in y_splits:
            y_splits[y[i]].append(i)
        else:
            y_splits[y[i]] = [i]
    split_size = (len(X) // n_splits)//len(y_splits)
    for i in range(n_splits):
        train_split = []
        test_split = []
        extra = True
        for j in y_splits.values():
            test_split.extend(j[i*split_size:i*split_size+split_size])
            if i < first_splits and extra:
                if len(j) > n_splits:
                    ind = i*split_size+split_size
                    if(len(j) > ind):
                        test_split.append(j[ind])
                        j.pop(i*split_size+split_size)
                    else: 
                        test_split.append(j[-1])
                        j.pop(-1)
        for j in indices:
            if j not in test_split:
                train_split.append(j)
        folds.append((train_split, test_split))

    return folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if n_samples is None:
        n_samples = len(X)
    if y is None:
        y_samples = None
        y_out_of_bag = None
    else:
        y_samples = []
        y_out_of_bag = []
    x_samples = []
    x_out_of_bag = []
    rng = np.random.default_rng(random_state)
    samples = rng.integers(len(X), size=n_samples)
    for i in samples:
        x_samples.append(X[i])
        if y is not None:
            y_samples.append(y[i])
    for i, _ in enumerate(X):
        if i not in samples:
            x_out_of_bag.append(X[i])
            if y is not None:
                y_out_of_bag.append(y[i])

    return x_samples, x_out_of_bag, y_samples, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    for i, _ in enumerate(y_true):
        x = labels.index(y_true[i])
        y = labels.index(y_pred[i])
        matrix[x][y] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct_count = 0
    for i, _ in enumerate(y_true):
        if y_true[i] == y_pred[i]:
            correct_count += 1
    if normalize:
        return float(correct_count)/float(len(y_true))
    return correct_count


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    true_positive = 0
    false_positive = 0
    if not labels and not pos_label:
        pos_label = y_true[0]
    elif not pos_label:
        pos_label = labels[0]
    for i, _ in enumerate(y_true):
        if y_pred[i] == pos_label:
            if y_pred[i] == y_true[i]:
                true_positive += 1
            else:
                false_positive += 1
    if false_positive + true_positive == 0:
        return 0
    return true_positive/(false_positive + true_positive)

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    true_positive = 0
    false_negative = 0
    if not labels and not pos_label:
        pos_label = y_true[0]
    elif not pos_label:
        pos_label = labels[0]
    for i, _ in enumerate(y_true):
        if y_true[i] == pos_label:
            if y_pred[i] == y_true[i]:
                true_positive += 1
            else:
                false_negative += 1
    if false_negative + true_positive == 0:
        return 0
    return true_positive/(false_negative + true_positive)

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if not labels and not pos_label:
        pos_label = y_true[0]
    elif not pos_label:
        pos_label = labels[0]
    precision = binary_precision_score(y_true, y_pred, pos_label=pos_label)
    recall = binary_recall_score(y_true, y_pred, pos_label=pos_label)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)
