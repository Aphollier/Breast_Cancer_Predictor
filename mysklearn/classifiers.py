"""classifier.py
@author aphollier, Commander-Cross
Alexander Hollier and Michael Waight
"""
from mysklearn import classifier_utils
import math

##############################################################
# KNEIGHBORS CLASSIFIER
##############################################################
class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for test in X_test:
            curr_distances = []
            for train in self.X_train:
                curr_distances.append(math.dist(train, test))
            best_distances = sorted(curr_distances)[:self.n_neighbors]
            best_indices = []
            for dis in best_distances:
                best_indices.append(curr_distances.index(dis))
            distances.append(best_distances)
            neighbor_indices.append(best_indices)

        return distances, neighbor_indices

    def predict(self, X_test, catagorical=False):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
            catagorical(bool): whether or not a dataset is catagorical or not
                This is rather rushed for pa6
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        if catagorical:
            for test in X_test:
                max_index = 0
                max_match = 0
                for i, x in enumerate(self.X_train):
                    curr_match = 0
                    for j, att in enumerate(test):
                        if att == x[j]:
                            curr_match += 1
                    if curr_match > max_match:
                        max_match = curr_match
                        max_index = i
                y_predicted.append(self.y_train[max_index])

        else:
            _, indices = self.kneighbors(X_test)
            for x in indices:
                labels = []
                for i in x:
                    labels.append(self.y_train[i])
                y_predicted.append(classifier_utils.most_frequent(labels))
        return y_predicted


##############################################################
# DUMMY CLASSIFIER
##############################################################
class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        # Find the most frequent class label
        self.most_common_label = classifier_utils.find_majority_1D(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.most_common_label is None:
            return None
        else:
            y_predicted = []
            for _ in range(len(X_test)):
                y_predicted.append(self.most_common_label)
        return y_predicted


##############################################################
# NAIVE BAYES CLASSIFIER
##############################################################
class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        priors(dict): The prior probabilities computed for each
            label in the training set.
        posteriors(dict of dict): The posterior probabilities computed for each
            attribute value/label pair in the training set.
        classifications(list): The classifications present in X_train that have
            been fit to the classifier
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        self.classifications = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """

        X_copy = X_train.copy()
        y_copy = y_train.copy()

        # Get unique class labels
        classifications_count, classifications = classifier_utils.group_by_classification(y_train)
        self.classifications = classifications

        # Compute priors
        priors = {}
        for classification in classifications:
            priors[classification] = 0
            for idx in range(len(y_copy)):
                if y_copy[idx] == classification:
                    priors[classification] += 1
            priors[classification] = priors[classification] / len(y_copy)
        # Compute posteriors
        posteriors = classifier_utils.compute_posteriors(X_copy, y_copy, classifications, classifications_count)

        self.priors = priors
        self.posteriors = posteriors

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        predictions = {}
        for instance in X_test:
            for classification in self.classifications:
                prediction = 1
                for attribute_idx , value in enumerate(instance):
                    keys_list = self.posteriors[classification][attribute_idx].keys()
                    for key in keys_list:
                        if value == key:
                            prediction *= self.posteriors[classification][attribute_idx][key]
                prediction *= self.priors[classification]
                predictions[classification] = prediction
            # Choose the correct instance
            prediction = classifier_utils.find_naive_prediction(predictions, self.classifications, self.priors)
            y_predicted.append(prediction)

        return y_predicted


##############################################################
# Decision Tree Classifier
##############################################################
class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.header = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """

        # Build string of arbitrary attribute names
        available_attributes = []
        for i in range(len(X_train[0])):
            available_attributes.append("att" + str(i))
        self.header = available_attributes.copy()
        # Build dictionary of attribute domains
        attribute_domains = {}
        for i in range(len(X_train[0])):
            attribute_domains["att" + str(i)] = []
            for row in X_train:
                if row[i] not in attribute_domains["att" + str(i)]:
                    attribute_domains["att" + str(i)].append(row[i])
        for k in attribute_domains.keys():
            attribute_domains[k] = sorted(attribute_domains[k])
        # Combine X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]

        # Call TDIDT
        tree = classifier_utils.tdidt(self, train, available_attributes, attribute_domains)
        self.tree = tree

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for instance in X_test:
            prediction = classifier_utils.find_tree_prediction(self.tree, instance)
            predictions.append(prediction)
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        # Call recursive function
        classifier_utils.find_rule(self, self.tree, "", attribute_names, class_name)

##############################################################
# Random Forest Classifier
##############################################################
class MyRandomForestClassifier:
    """
    Represents a Random Forest Classifier

    Attributes:
        remainder_set(list of list): The 2D list of training data after creating the random stratified
            test set from a third of the original dataset
        N (int): The number of total decision trees to make.
        M (int): The number of most accurate decision trees to create rand_forest.
        F (int): The number of attributes to consider splitting on for each node in a decision tree
        rand_forest(list of MyDecisionTreeClassifiers): The decision tree classifiers making up the
            random forest
        rand_state (int): Value to seed the random class
    Notes:
        Loosely based on sklearn's RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    def __init__(self, N, M, F, rand_state=None):
        """
        Class initializer
        """
        self.remainder_set = None
        self.N = N
        self.M = M
        self.F = F
        self.rand_forest = None
        self.rand_state = rand_state
        if rand_state is not None:
            classifier_utils.seed(rand_state)

    def fit(self, remainder_set_X, remainer_set_y):
        """
        Fits a random forest classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            remainder_set_X (list of list): 2D list of data to fit the classifier
            remainder_set_y (list of obj): list of class attributes
        """

        self.remainder_set = [remainder_set_X[i] + [remainer_set_y[i]] for i in range(len(remainder_set_X))]
        tree_list = []
        test_list = []
        for i in range(self.N):
            train, test = classifier_utils.compute_bootstrapped_sample(self.remainder_set)
            test_list.append(test)
            # Build string of arbitrary attribute names
            available_attributes = []
            for i in range(len(train[0]) - 1):
                available_attributes.append("att" + str(i))
            header = available_attributes.copy()
            # Build dictionary of attribute domains
            attribute_domains = {}
            for i in range(len(train[0])-1):
                attribute_domains["att" + str(i)] = []
                for row in train:
                    if row[i] not in attribute_domains["att" + str(i)]:
                        attribute_domains["att" + str(i)].append(row[i])
            for k in attribute_domains.keys():
                attribute_domains[k] = sorted(attribute_domains[k])

            # Call TDIDT
            tree = classifier_utils.tdidtforest(self, train, available_attributes, attribute_domains, header)
            tree_list.append(tree)
        self.rand_forest = classifier_utils.prune_rand_forest(tree_list, test_list, self.M)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for instance in X_test:
            prediction = classifier_utils.find_forest_prediction(self.rand_forest, instance)
            predictions.append(prediction)
        return predictions
