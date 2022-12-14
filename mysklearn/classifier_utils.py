"""classifier_utils.py
@author aphollier, Commander-Cross
Alexander Hollier and Michael Waight
"""
import numpy as np
import operator
# Need bootstrap_sample for random forest clf
from mysklearn import evaluators as eval
from mysklearn import classifiers as clf

def most_frequent(arr):
    """Finds the most common value in a list
    Args:
        arr(list of obj): The list to find
        the most frequent occurance
    Returns:
        max(obj): The most common value
    """
    return max(set(arr), key = arr.count)

def compute_euclidean_distance(v1, v2):
    """
    Function computes the distance between two values using the euclidean
    distance algorithm
    Args:
        v1 (float): point one
        v2 (float): point two
    Returns:
        float: the distance value between the two points
    """
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def find_majority_1D(classifications):
    """
    Function returns the most common classification value from a 1D list
    Args:
        classifications (list): list of classification values from y_train
    Returns:
        obj: the most common classification from the list
    """
    # Build dictionary containing frequency of each classification
    y_predicted = ""
    class_frequency = {}
    for value in classifications:
        if value in class_frequency:
            class_frequency[value] += 1
        else:
            class_frequency[value] = 1
    y_predicted = max(class_frequency, key=class_frequency.get)
    return y_predicted

def find_majority_2D(classifications):
    """
    Function returns a list of the most common classification values from each row
    in a 2D list
    Args:
        classifications (list of list): 2D list of classification values from y_train
    Returns:
        list: the most common classification from each row of classifications in 
        the list
    """
    y_predicted = []
    for row in classifications:
        class_frequency = {}
        for value in row:
            if value in class_frequency:
                class_frequency[value] += 1
            else:
                class_frequency[value] = 1
        y_predicted.append(max(class_frequency, key=class_frequency.get))

    return y_predicted

def get_indices_and_dists(MyKNeighborsClassifier, X_test, categorical):
    """
    Function creates a 2D list of tuples (index, distance) based off of the
    distances computed between and instance of X_train and X_test
    Args:
        X_test (list of list): 2D list of point values to be used in computing
        distances
    Returns:
        list of list: 2D list of tuples (index, distance)
    """
    row_indexes_dists = []
    for test_point in X_test:
        index_distances = []
        for i, train_point in enumerate(MyKNeighborsClassifier.X_train):
            if categorical:
                if test_point == train_point:
                    distance = 0
                elif test_point != train_point:
                    distance = 1
            else:
                distance = round(compute_euclidean_distance(\
                    test_point, train_point), 3)
            index_distances.append((i, distance))
        row_indexes_dists.append(index_distances)
    return row_indexes_dists

def get_kclosest_neighbors(knn_clf, row_indexes_dists):
    """
    Function finds the k closest neighbors for each list in the 2D list
    row_indexes_dists by finding the items with the shortest distances
    Args:
        knn_clf (MyKNeighborsClassifier): MyKNeighborsClassifier object used to
        obtain the n_neighbors attribute
        row_indexes_dists (list of list): 2D list of tuples (index, distance) used
        to find the k closest neighbors
    Returns:
        list of list: 2D list of tuples
    """
    total_neighbors = []
    for index_distances in row_indexes_dists:
        index_distances.sort(key=operator.itemgetter(-1))
        total_neighbors.append(index_distances[:knn_clf.n_neighbors])
    return total_neighbors

def find_unique_attributes(X_train):
    """
    This function finds all of the unique attribute values in each column
    within the dataset ignoring duplicates
    Args:
        X_train (list of list): The dataset
    Returns:
        list of list: A 2D list where eahc nested list represents a column
        and the values within each nested list reprsent the unique attribute
        values within the column
    """

    attributes = []
    for idx in range(len(X_train[0])):
        # Want a new list for each row
        row_attributes = []
        for row in X_train:
            if row[idx] not in row_attributes:
                row_attributes.append(row[idx])
        attributes.append(row_attributes)

    return attributes

def compute_posteriors(X_train, y_train, classifications, classifications_count):
    # Build posteriors dictionary architecture
    unique_attributes = find_unique_attributes(X_train)
    posteriors = {}
    for classification in classifications:
        attribute_dict = {}
        for attribute, col_attributes in enumerate(unique_attributes):
            value_dict = {}
            for value in col_attributes:
                value_dict[value] = 0
            attribute_dict[attribute] = value_dict
        posteriors[classification] = attribute_dict

    # Compute posteriors
    for class_idx, classification in enumerate(classifications):
        for attribute, col_attributes in enumerate(unique_attributes):
            for value in col_attributes:
                y_idx = 0
                for row in X_train:
                    if row[attribute] == value and y_train[y_idx] == classification:
                        posteriors[classification][attribute][value] += 1
                    y_idx += 1
                posteriors[classification][attribute][value] /= classifications_count[class_idx]
    return posteriors

def find_naive_prediction(predictions, classifications, priors):
    """
    This function is called within the the Naive Bayes Classifiers predict() function.
    Given the prediction options, it chooses the correct prediction according to the
    algorithm. First looks for the largest prediction, if there is a tie it then looks
    for the largest prior, if there is a tie it then chooses randomly
    Args:
        predictions (dictionary): A dictionary of predictions to choose from
        classifications (list): A list of all the classifications present in the dataset.
            Used for accessing the predictions and priors dictionaries
        priors (dictionary): A dictionary of prior values to choose from
    Returns:
        obj: The prediction value chosen from the predictions
    """
    # Choose largest prediction value
    max_class_value = -1    # -1 instead of 0 to avoid falsely finding a posterior match
    for classification in classifications:
        if predictions[classification] > max_class_value:
            max_class_value = predictions[classification]
            max_class = classification
        elif predictions[classification] == max_class_value:
            # Found a tie, choose prediction w/ the largest prior
            max_prior_value = -1
            for _ in range(len(priors)):
                if priors[classification] > max_prior_value:
                    max_prior_value = classification
                    max_prior = classification
                elif priors[classification] == max_prior_value:
                    # Found a tie, flip a coin
                    rand_idx = np.random.randint(0, len(classifications))
                    return classifications[rand_idx]
            return max_prior
    return max_class

def shuffle_data(random_state, X, y=None):
    """
    This function shuffles two parallel lists randomly while keeping them parellel.
    If y is none, only shuffle X

    Args:
        random_state (int): an integer used for seeding np.random
        X (list of list): A 2D list of values to be sorted parallel to y
        y (list, optional): A list of values to be sorted parallel to X. Defaults to None.

    Returns:
        X and y: returns the now shuffled parellel lists
    """
    if random_state is not None:
        np.random.seed(random_state)

    # shuffle indexes
    for i in range(len(X)):
        rand_idx = np.random.randint(len(X))
        X[i], X[rand_idx] = X[rand_idx], X[i]
        if y is not None:
            y[i], y[rand_idx] = y[rand_idx], y[i]

    return X, y

def split_data(n_splits, X):
    """
    Function splits the dataset X as evenly as possible into n partitions

    Args:
        n_splits (int): number of partitions to split data into
        X (list of list): dataset that needs to be split up

    Returns:
        list of list: A 2D list with lists of split data from X
    """
    splits = []
    for _ in range(n_splits):
        splits.append([])
    idx = 0
    for i in range(len(X)):
        splits[idx].append(i)
        # Check for looping back around
        if idx == n_splits - 1:
            idx = 0
        else:
            idx += 1

    return splits

def split_stratified_data(n_splits, group_indices):
    """
    Function splits the stratified dataset as evenly as possible into n partitions
    Differs from split_data as the original dataset has already been split into
    groups based on classification

    Args:
        n_splits (int): number of partitions to split data into
        group_indices (list of list): each row in this 2D list is a list of all of the
        indexes with corresponding attributes

    Returns:
        list of list: A 2D list with lists of stratified split data from X
    """
    splits = []
    for _ in range(n_splits):
        splits.append([])
    idx = 0
    for group in group_indices:
        for i in range(len(group)):
            splits[idx].append(group[i])
            # Check for looping back around
            if idx == n_splits - 1:
                idx = 0
            else:
                idx += 1

    return splits

def fold_data(n_splits, splits):
    """
    Function 'folds' the data into k tuples. Each tuple has a test set and a training set

    Args:
        n_splits (int): the number of splits in the lists splits
        splits (list of list): A 2D list, each row is a split from the dataset that needs to be folded

    Returns:
        list of list: a list of tuples, each with a test set and train set
    """
    folds = []
    for test_idx in range(n_splits):
        train_set = []
        for idx in range(n_splits):
            if idx == test_idx:
                test_set = splits[idx]
            elif idx != test_idx:
                train_set += splits[idx]
        folds.append((train_set, test_set))
        del train_set
        del test_set
    
    return folds

def group_by_index(y):
    """
    Function creates a list for each classification present in y.
    Then groups the indexes of the elements into their corresponding
    classification

    Args:
        y (list): listof classification values

    Returns:
        list of list: A 2D list where each list is a group of indexes with the
        same classification value
    """
    groups = []
    classes = []
    for i in range(len(y)):
        if y[i] not in classes:
            classes.append(y[i])
            groups.append([])

    for i in range(len(y)):
        for j, classification in enumerate(classes):
            if y[i] == classification:
                groups[j].append(i)
                break

    return groups

def group_by_classification(y):
    """
    Function builds a list of classifications. One for every unique classification
    in the given list. (Gets rid of dups). It also counts how many times each unique attribute
    occurs and builds a list parralel to the classification list
    Args:
        y (list): list of classifications
    Returns:
        list: list of one of each unique classification within y
    """
    classifications = []
    classifications_count = []
    for i in range(len(y)):
        if y[i] not in classifications:
            classifications.append(y[i])
            classifications_count.append(0)

    for i, classification in enumerate(classifications):
        for j in range(len(y)):
            if classification == y[j]:
                classifications_count[i] += 1

    return classifications_count, classifications

def compute_priors(classifications, cur_instance):
    """
    This function is called in the select_attribute function to build a dictionary of the priors
    corresponding to each unique attribute value for all attributes. These values are then used
    in the algorithm to find the weighted entropy of each attribute
    Args:
        classifications (list): a list of all of the classification values
        cur_instance (list of list): a portion of the training set to get the priors from
    Returns:
        dictionary: returns a nested dictionary that maps each attribute and its unique values to
        a prior
    """
    # Build priors dictionary architecture
    priors = {}
    for classification in classifications:
        priors[classification] = {}
        for i in range(len(cur_instance[0]) - 1):
            priors[classification]["att" + str(i)] = {}
            for row in cur_instance:
                priors[classification]["att" + str(i)][row[i]] = 0
    # Find the numerator for each attribute value and corresponding classification
    for classification in classifications:
        for i in range(len(cur_instance[0]) - 1):
            for row in cur_instance:
                att_keys = priors[classification]["att" + str(i)].keys()
                for key in att_keys:
                    if row[i] == key and row[-1] == classification:
                        priors[classification]["att" + str(i)][row[i]] += 1
    # Now find the denomanator for each attribute value and divide the numerator to get the priors
    for i in range(len(cur_instance[0]) - 1):
        att_keys = priors[classification]["att" + str(i)].keys()
        for key in att_keys:
            denom = 0
            for classification in classifications:
                denom += priors[classification]["att" + str(i)][key]
            if denom == 0:
                denom = 1
            for classification in classifications:
                priors[classification]["att" + str(i)][key] /= denom

    return priors

def get_accurance(key, attribute, header, cur_instance):
    """
    This function finds the number of instances for each attribute value and divides it
    by the number of total instances. This number is used in the equation to calculate the
    weighted entropy
    Args:
        key (string): This key value represent a unique attribute value
        attribute (string): The name of the attribute whose instances we want to compare
        the value of key to
        header (list): a list of all the attributes in the training data used to find the index
        of the column we want to look through
        cur_instance (list of list): the data containing all the instances
    Returns:
        float: The number of times the attribute value occurs over the total number of instances
    """
    # Get the index of the column we want to look through
    idx = 0
    for i in range(len(header)):
        if attribute == header[i]:
            idx = i
            break
    # Find the number of times that attribute value occurs in the column
    accurances = 0
    for row in cur_instance:
        if row[idx] == key:
            accurances += 1
    accurances /= len(cur_instance)

    return accurances

def select_attribute(cur_instance, header, available_attributes):
    """
    This function is called in the TDIDT function to find the attribute to split on next
    given the current instance. It chooses this attribute by calculating the weighted entropy
    of all available attributes and then choosing the smallest entropy value
    Args:
        cur_instance (list of list): data containing instances of all of the attributes
        header (list): a list of all the attributes in the training data
        available_attributes (list): A list of all the available attributes (ones that have not
        yet been split on)
    Returns:
        string: The attribute selected
    """
    # Get unique class labels
    y_train = []
    for row in cur_instance:
        y_train.append(row[-1])
    classifications_count, classifications = group_by_classification(y_train)

    priors = compute_priors(classifications, cur_instance)
    # Find the weighted average of each partition
    weighted_entropies = {}
    for attribute in available_attributes:
        weighted_entropies[attribute] = 0
        att_keys = priors[classifications[0]][attribute].keys()
        for key in att_keys:
            entropy = 0
            for classification in classifications:
                if priors[classification][attribute][key] != 0.0:
                    entropy += -(priors[classification][attribute][key]) *\
                            np.log2(priors[classification][attribute][key])
            accurance = get_accurance(key, attribute, header, cur_instance)
            weighted_entropies[attribute] += accurance * entropy

    # Select the attribute with the minimum weighted entropy
    split_attribute = min(weighted_entropies, key=weighted_entropies.get)

    return split_attribute

def partition_instances(cur_instance, header, split_attribute, att_domains):
    """
    This function creates a dictionary by partitioning the current instance based on the
    split_attribute value
    Args:
        cur_instance (list of list): data containing instances of all of the attributes
        header (list): a list of all the attributes in the training data
        split_attribute (string): The attribute name selected to split on
        att_domains (list): A list of the unique attribute values
    Returns:
        dictionary: A dictionary of partitioned instances based on the split_attribute value
    """
    # this is a group by attribute domain
    att_index = header.index(split_attribute)

    att_domain = att_domains["att" + str(att_index)]
    # Use dictionaries
    partitions = {}
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in cur_instance:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions

def same_class_label(att_partition):
    """
    This function determines whether all of the class labels are the same within a given
    partition
    Args:
        att_partition (list of list): A list of instances within a partition
    Returns:
        bool: Boolean value determining whether all class labels are the same or not
    """

    first_label = att_partition[0][-1]
    for instance in att_partition:
        if instance[-1] != first_label:
            return False
    # get here, all the same
    return True

def tdidt(self, cur_instance, available_attributes, attribute_domains):
    """
    This function implements the TDIDT algorithm using recursion
    Args:
        cur_instance (list): The current instance to split on or create leaf on
        available_attributes (list): A list of all the available attributes (ones that have not
        yet been split on)
        attribute_domains (list): A list of the unique attribute values
    Returns:
        list of list: A nested list structure representing the decision tree
    """
    # Select attribute to split on using entropy
    split_attribute = select_attribute(cur_instance, self.header, available_attributes)
    available_attributes.remove(split_attribute)

    # Root node of current tree
    tree = ["Attribute", split_attribute]

    partitions = partition_instances(cur_instance, self.header, split_attribute, attribute_domains)
    class_values = list(partitions.keys())
    class_frequency = []
    for class_value in class_values:
        class_frequency.append((class_value, len(partitions[class_value])))

    # Check for base cases, if none then recurse
    for att_value, att_partition in partitions.items():
        value_subtree = ["Value", att_value]

        if len(att_partition) > 0 and same_class_label(att_partition):
            # Case 1: Make leaf node
            value_subtree.append(["Leaf", str(att_partition[0][-1]), len(att_partition), len(cur_instance)])
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            # Case 2: Clash occured
            # Build list of classifications from partition
            class_partition = []
            for value in att_partition:
                class_partition.append(value[-1])
            value_subtree.append(["Leaf", find_majority_1D(class_partition), len(att_partition), len(cur_instance)])
        elif len(att_partition) == 0:
            # Case 3: Empty partition, backtrack
            majority_class = max(class_frequency, key=operator.itemgetter(1))
            class_count_total = sum(class_tuple[1] for class_tuple in class_frequency)
            tree = ["Leaf", majority_class[0], majority_class[1], class_count_total]
        else:
            # Recurse
            subtree = tdidt(self, att_partition, available_attributes.copy(), attribute_domains.copy())
            value_subtree.append(subtree)

        tree.append(value_subtree)

    return tree

def find_tree_prediction(tree_level, test_instance):
    """
    This function traverses the decision tree to find the correct prediction given the test_instance
    Args:
        tree_level (list of list): A portion of the decision tree
        test_instance (list): An instance of X_test
    Returns:
        string: The predicted classification
    """
    prediction = ""
    if tree_level[0] == "Attribute":
        # Parse instances split attribute index
        for i, char in enumerate(tree_level[1]):
            if i == 3:
                att_idx = int(char)
        # Find index of attribute value
        for value_idx, value_list in enumerate(tree_level):
            # Skip first two elems in list
            if value_idx >= 2 and value_list[1] == test_instance[att_idx]:
                # Recurse with next level down
                prediction = find_tree_prediction(value_list[2], test_instance)
                break
    # Once we have reached the leaf node, simply grab the predicted value
    elif tree_level[0] == "Leaf":
        prediction = tree_level[1]
        return prediction

    return prediction

def find_rule(self, tree_level, rule, attribute_names=None, class_name=None):
    """
    This function builds rules based off of the structure of the decision tree and prints
    them out once they are complete.
    Args:
        tree_level (list of list): A portion of the decision tree
        attribute_names (list): A list of all of the attribute names
        class_name (string): The class name
        rule (string): The rule to be sent through recursively
    Returns:
        string: The rule being built
    """
    # Attribute level
    if tree_level[0] == "Attribute":
        for value_idx, value_list in enumerate(tree_level):
            # Parse the index out of the string to get the actual attribute name
            if value_idx == 1 and attribute_names != None:
                for i, char in enumerate(tree_level[1]):
                    if i == 3:
                        att_idx = int(char)
            # Add to rule
            elif value_idx >= 2:
                if attribute_names != None and rule == "":
                    rule += "IF " + str(attribute_names[att_idx]) + " == " + str(value_list[1]) + " "
                elif attribute_names == None and rule == "":
                    rule += "IF " + str(value_list) + " == " + str(value_list[1]) + " "
                elif attribute_names != None and rule != "":
                    rule += "AND " + str(attribute_names[att_idx]) + " == " + str(value_list[1]) + " "
                elif attribute_names == None and rule != "":
                    rule += "AND " + str(value_list) + " == " + str(value_list[1]) + " "
                rule = find_rule(self, value_list[2], rule, attribute_names, class_name)
    # Leaf level
    elif tree_level[0] == "Leaf":
        # Finish rule
        if class_name != None:
            rule += "THEN " + class_name + " = " + str(tree_level[1])
        else:
            rule += "THEN class = " + str(tree_level[1])
        print(rule)
        # Reset the rule
        return ""

    return rule

def get_X_y(dataset):
    """
    Function gets the X and y data lists from a 2D dataset where the last
    element of each row is the y instance.

    Args:
        dataset (list of list): 2D dataset containing X and y data

    Returns:
        (list of list), (list): X and y respectively
    """
    X =[]
    y = []
    for row in dataset:
        X.append(row[:-1])
        y.append(row[-1])

    return X, y

def prune_rand_forest(tree_list, test_list, M):
    """
    Function selects to M most accurate trees from the current list of trees
    Args:
        tree_list (list of trees): list of tree's
        test_list (list of lists): list of tree's
        M (int): the amount of trees to include in our forest

    Returns:
        list of decision trees: The top M most accurate decision trees
    """
    tree_accuracy_tuples = []
    best_trees = []
    for ind, tree in enumerate(tree_list):
        test_X = [test_list[ind][i][:-1] for i in range(len(test_list[ind]))]
        test_y = [test_list[ind][i][-1] for i in range(len(test_list[ind]))]
        predictions = []
        for instance in test_X:
            prediction = find_tree_prediction(tree, instance)
            predictions.append(prediction)
        acc = eval.accuracy_score(test_y, predictions)
        tree_accuracy_tuples.append((tree, acc))
    for i in range(M):
        best = max(tree_accuracy_tuples, key=lambda x:x[1])
        best_trees.append(best[0])
        tree_accuracy_tuples.remove(best)
    return best_trees

def tdidtforest(self, cur_instance, available_attributes, attribute_domains, header):
    """
    This function implements the TDIDT algorithm using recursion but is edited to work for Random Forest Classifier
    Args:
        cur_instance (list): The current instance to split on or create leaf on
        available_attributes (list): A list of all the available attributes (ones that have not
        yet been split on)
        attribute_domains (list): A list of the unique attribute values
        header (list): A list of all attributes
    Returns:
        list of list: A nested list structure representing the decision tree
    """
    # Create Subset of available_attributes
    subset = compute_random_subset(available_attributes, self.F)

    # Select attribute to split on using entropy
    split_attribute = select_attribute(cur_instance, header, subset)
    available_attributes.remove(split_attribute)

    # Root node of current tree
    tree = ["Attribute", split_attribute]

    partitions = partition_instances(cur_instance, header, split_attribute, attribute_domains)
    class_values = list(partitions.keys())
    class_frequency = []
    for class_value in class_values:
        class_frequency.append((class_value, len(partitions[class_value])))

    # Check for base cases, if none then recurse
    for att_value, att_partition in partitions.items():
        value_subtree = ["Value", att_value]

        if len(att_partition) > 0 and same_class_label(att_partition):
            # Case 1: Make leaf node
            value_subtree.append(["Leaf", str(att_partition[0][-1]), len(att_partition), len(cur_instance)])
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            # Case 2: Clash occured
            # Build list of classifications from partition
            class_partition = []
            for value in att_partition:
                class_partition.append(value[-1])
            value_subtree.append(["Leaf", find_majority_1D(class_partition), len(att_partition), len(cur_instance)])
        elif len(att_partition) == 0:
            # Case 3: Empty partition, backtrack
            majority_class = max(class_frequency, key=operator.itemgetter(1))
            class_count_total = sum(class_tuple[1] for class_tuple in class_frequency)
            tree = ["Leaf", majority_class[0], majority_class[1], class_count_total]
        else:
            # Recurse
            subtree = tdidtforest(self, att_partition, available_attributes.copy(), attribute_domains.copy(), header)
            value_subtree.append(subtree)

        tree.append(value_subtree)

    return tree

def compute_bootstrapped_sample(table):
    """
    This function computes the random test/train split for random forest
    Args:
        table (list of lists): The Data table to be split
    Returns:
        sample (list of lists): The Train data
        out_of_bag_sample (list of lists): The Test Data
    """
    n = len(table)
    # np.random.randint(low, high) returns random integers from low (inclusive) to high (exclusive)
    sampled_indexes = [np.random.randint(0, n) for _ in range(n)]
    sample = [table[index] for index in sampled_indexes]
    out_of_bag_indexes = [index for index in list(range(n)) if index not in sampled_indexes]
    out_of_bag_sample = [table[index] for index in out_of_bag_indexes]
    return sample, out_of_bag_sample

def compute_random_subset(values, num_values):
    """
    This function computes the random subset of attributes for random forest
    Args:
        values (list of str): the list of attributes
        num_values (int): the F value or the number of values in the subset
    Returns:
        values_copy (list of str): the subset of values
    """
    # could use np.random.choice()
    # I'll use np.random.shuffle() and slicing
    values_copy = values.copy() # shallow copy
    np.random.shuffle(values_copy) # inplace shuffle
    return values_copy[:num_values]

def seed(rand_state):
    """
    This function sets the seet for random forest
    Args:
        rand_state (int)): the random state
    """
    np.random.seed(rand_state)

def find_forest_prediction(tree_list, test_instance):
    """
    This function gets the predictions for random forest by getting the majority rule of the prediction
    of each tree
    Args:
        tree_list (list of trees): the list of trees
        test_instance (list of obj): the current test set
    Returns:
        most_frequent(count) (obj): the prediction
    """
    count = []
    for tree in tree_list:
        count.append(find_tree_prediction(tree, test_instance))
    return most_frequent(count)