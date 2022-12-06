"""myclassifier.py

@author aphollier
"""
import math

from mysklearn import myutils

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
                y_predicted.append(myutils.most_frequent(labels))
        return y_predicted



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
        if len(X_train) == len(y_train): # I needed to use X_train somehow for the linter?
            self.most_common_label = myutils.most_frequent(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for _ in X_test:
            y_predicted.append(self.most_common_label)
        return y_predicted


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = {}
        self.posteriors = {}

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
        self.priors = {}
        self.posteriors = {}
        for prior in y_train:
            if prior in self.priors:
                self.priors[prior] += 1
            else:
                self.priors[prior] = 1
                self.posteriors[prior] = [dict() for _, _ in enumerate(X_train[0])]
        for i, X in enumerate(X_train):
            for j, posterior in enumerate(X):
                if posterior in self.posteriors[y_train[i]][j]:
                    self.posteriors[y_train[i]][j][posterior] += 1
                else:
                    self.posteriors[y_train[i]][j][posterior] = 1
        for prior in self.priors:
            for i, _ in enumerate(self.posteriors[prior]):
                for posterior in self.posteriors[prior][i].keys():
                    self.posteriors[prior][i][posterior] = self.posteriors[prior][i][posterior] / self.priors[prior]
            self.priors[prior] = self.priors[prior] / len(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        pred = []
        for test in X_test:
            curr_pred = {}
            for prior in self.priors:
                prior_pred = 1
                for i, X in enumerate(test):
                    if X in self.posteriors[prior][i]:
                        prior_pred *= self.posteriors[prior][i][X]
                    else:
                        prior_pred = 0
                curr_pred[prior] = prior_pred * self.priors[prior]
            max_vals = max(curr_pred.values())
            for k,v in sorted(curr_pred.items()):
                if v == max_vals:
                    pred.append(k)
                    break
        return pred

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
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = []
        header = []
        attribute_domains = {}
        for row in X_train:
            for i, col in enumerate(row):
                if len(available_attributes) != len(row):
                    header.append("att" + str(i))
                    available_attributes.append("att" + str(i))
                    attribute_domains["att" + str(i)] = []
                if col not in attribute_domains["att" + str(i)]:
                    attribute_domains["att" + str(i)].append(col)
        for k in attribute_domains.keys():
            attribute_domains[k] = sorted(attribute_domains[k])
        self.tree = myutils.tdidt(train, available_attributes, header, attribute_domains)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        header = []
        sol = []
        for i, _ in enumerate(X_test[0]):
            header.append("att" + str(i))
        for test in X_test:
            sol.append(myutils.tdidt_predict(self.tree, test, header))
        return sol

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
        myutils.tdidt_print(self.tree, "", attribute_names, class_name)
        



    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
