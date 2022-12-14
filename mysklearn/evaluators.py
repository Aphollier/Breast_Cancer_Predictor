"""test_classifier.py
@author Commander-Cross
Michael Waight
"""
from mysklearn import classifier_utils
import numpy as np # use numpy's random number generation

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

    # Copy data params
    X_copy = X.copy()
    y_copy = y.copy()

    # Get size of test instances given the test_size
    if type(test_size) == int:
        test_size_int = test_size
    elif type(test_size) == float:
        test_size_int = int(np.ceil(test_size * len(X)))

    # Shuffle if True
    if shuffle:
        X_copy, y_copy = classifier_utils.shuffle_data(random_state, X_copy, y_copy)

    # Build X_train, X_test and y_train, y_test
    X_train = X_copy
    y_train = y_copy
    X_test = []
    y_test = []
    for _ in range(test_size_int):
        X_test.insert(0, X_train[-1])
        y_test.insert(0, y_train[-1])
        X_train.pop(-1)
        y_train.pop(-1)

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

    X_copy = X.copy()
    # Shuffle if True
    if shuffle:
        X_copy = classifier_utils.shuffle_data(random_state, X_copy)

    # Split the data
    splits = classifier_utils.split_data(n_splits, X_copy)

    # "Fold" splits into tuples
    folds = classifier_utils.fold_data(n_splits, splits)

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

    X_copy = X.copy()
    y_copy = y.copy()

    # Shuffle if True
    if shuffle:
        X_copy = classifier_utils.shuffle_data(random_state, X_copy, y_copy)

    # Perform group by
    grouped_indices = classifier_utils.group_by_index(y_copy)

    # Now deal out indexes
    splits = classifier_utils.split_stratified_data(n_splits, grouped_indices)

    # "Fold" splits into tuples
    folds = classifier_utils.fold_data(n_splits, splits)

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

    if random_state is not None:
        np.random.seed(random_state)

    # Get sample indexes
    X_sample = []
    y_sample = []
    if n_samples is None:
        for _ in range(len(X)):
            rand_idx = np.random.randint(len(X))
            X_sample.append(X[rand_idx])
            y_sample.append(y[rand_idx])
    else:
        for _ in range(n_samples):
            rand_idx = np.random.randint(len(X))
            X_sample.append(X[rand_idx])
            y_sample.append(y[rand_idx])
    # Get "out of bag" indexes
    X_out_of_bag = []
    y_out_of_bag = []
    for idx in range(len(X)):
        if X[idx] not in X_sample:
            X_out_of_bag.append(X[idx])
            y_out_of_bag.append(y[idx])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def bootstrap_sample_stratified(X, y, random_state=None):
    """Split dataset into a stratified bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    """

    if random_state is not None:
        np.random.seed(random_state)

    # Split by class attribute
    split_ind = {}
    for i in range(len(X)):
        if y[i] in split_ind:
            split_ind[y[i]].append(i)
        else:
            split_ind[y[i]] = [i]

    # Get sample indexes
    X_sample = []
    y_sample = []
    X_out_of_bag = []
    y_out_of_bag = []
    for splits in list(split_ind.values()):
        n = len(splits)
        sampled_indexes = [np.random.randint(0, n) for _ in range(n)]
        X_sample.extend([X[splits[index]] for index in sampled_indexes])
        y_sample.extend([y[splits[index]] for index in sampled_indexes])
        out_of_bag_indexes = [index for index in list(range(n)) if index not in sampled_indexes]
        X_out_of_bag.extend([X[splits[index]] for index in out_of_bag_indexes])
        y_out_of_bag.extend([y[splits[index]] for index in out_of_bag_indexes])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

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

    # Create matrix architecture based on len(labels)
    matrix = []
    for _ in range(len(labels)):
        row = []
        for _ in range(len(labels)):
            row.append(0)
        matrix.append(row)

    # Fill in the matrix
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            for idx in range(len(y_true)):
                if y_true[idx] == true_label and y_pred[idx] == pred_label:
                    matrix[i][j] += 1

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

    # Find the number of correct classifications in y_pred
    num_correct = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            num_correct += 1
    # If normalize is true, then normalize the accuracy and return
    if normalize:
        return num_correct / len(y_true)
    # Otherwise, just return the number of correct classifications
    else:
        return num_correct

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

    true_pos = 0
    false_pos = 0
    # Go through the parallel lists finding true positives and false positives
    for idx in range(len(y_true)):
        if y_true[idx] == pos_label and y_pred[idx] == pos_label:
            true_pos += 1
        elif y_true[idx] != pos_label and y_pred[idx] == pos_label:
            false_pos += 1

    # Check that true positives is not zero to avoid dividing by zero
    if true_pos == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)

    return precision

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

    true_pos = 0
    false_neg = 0
    # Go through the parallel lists finding true positives and false negatives
    for idx in range(len(y_true)):
        if y_true[idx] == pos_label and y_pred[idx] == pos_label:
            true_pos += 1
        elif y_true[idx] == pos_label and y_pred[idx] != pos_label:
            false_neg += 1

    # Check that true positives is not zero to avoid dividing by zero
    if true_pos == 0:
        recall = 0
    else:
        recall = true_pos / (true_pos + false_neg)

    return recall

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

    # Get precision
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    # Get recall
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    # Check that precision or recall are not zero
    if precision == 0 or recall == 0:
        f_one = 0
    else:
        # Compute the F1 score
        f_one = (2 * (precision * recall)) / (precision + recall)

    return f_one
