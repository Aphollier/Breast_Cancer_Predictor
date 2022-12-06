"""myutils.py

@author aphollier
"""
import numpy as np # use numpy's random number generation
import math
import mysklearn.myevaluation as myevaluation

def most_frequent(arr):
    """Finds the most common value in a list

    Args:
        arr(list of obj): The list to find
        the most frequent occurance

    Returns:
        max(obj): The most common value
    """
    return max(set(arr), key = arr.count)

def mpg_rating(mpg):
    """converts mpg data into ratings

    Args:
        mpg(float): A column value from mpg

    Returns:
        mpg_rated(int): the mpg data converted to a rated value
    """
    if mpg <= 13:
        return 1
    elif mpg == 14:
        return 2
    elif mpg <= 16:
        return 3
    elif mpg <= 19:
        return 4
    elif mpg <= 23:
        return 5
    elif mpg <= 27:
        return 6
    elif mpg <= 30:
        return 7
    elif mpg <= 36:
        return 8
    elif mpg <= 44:
        return 9
    else:
        return 10

def normalize(cols):
    """converts columns of data into normalized [0,1]
    data in point form

    Args:
        cols(list of lists of nums): A list of columns to me normalized

    Returns:
        normalized data(list of list of floats): the normalized data in point form
    """
    normalized_data = [[] for _ in cols[0]]
    for col in cols:
        max_val = max(col)
        min_val = min(col)
        for i, x in enumerate(col):
            normalized_data[i].append((x-min_val)/(max_val - min_val))
    return normalized_data

def randomize_in_place(alist, seed, parallel_list=None):
    """shuffles a list in place using np random

    Args:
        alist(list of objs): A list of objects to be shuffled
        seed(int): The seed for the random
        parallel_list(list of objs): A list parallel to alist to be shuffled
    """
    np.random.seed(seed)
    for i, _ in enumerate(alist):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]

def select_attribute(instances, attributes, headers):
    """select the attribute to split on using entropy

    Args:
        instances(list of list of objs): A list of instances
        attributes(list of objs): the attributes to split on
        headers(list of strs): A list of the headers

    Returns:
        information_gained(str): the attribute to split on
    """
    information_gained = {}
    for att in attributes:
        information_gained[att] = 0
        att_size = 0
        att_index = headers.index(att)
        domain = {}
        for instance in instances:
            if instance[att_index] in domain:
                domain[instance[att_index]] += 1
            else:
                domain[instance[att_index]] = 1
            att_size += 1
        for k in domain.keys():
            partition = {}
            e = 0
            for instance in instances:
                if instance[att_index] == k:
                    if instance[-1] in partition:
                        partition[instance[-1]] += 1
                    else:
                        partition[instance[-1]] = 1
            for part_k in partition.keys():
                e =+ -((partition[part_k]/domain[k]) * math.log2(partition[part_k]/domain[k]))
            information_gained[att] += (domain[k]/att_size) * e

    return min(information_gained, key=information_gained.get)

def partition_instances(instances, attribute, header, attribute_domains):
    """partitions on the attributes

    Args:
        instances(list of list of objs): A list of instances
        attributes(list of objs): the attribute to partition
        header(list of strs): A list of the headers
        attribute_domains(dictionary of lists of obj): the domains on each attribute

    Returns:
        partitions(list of tuples of str and list of list of objs): the partition
    """
    att_index = header.index(attribute)
    att_domain = attribute_domains["att" + str(att_index)]
    partitions = {}
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)
    return partitions

def same_class_label(instances):
    """check if the all the class labels are the same

    Args:
        instances(list of list of objs): A list of instances

    Returns:
        (bool): True or False
    """
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False
    return True

def tdidt(current_instances, available_attributes, header, attribute_domains):
    """Top Down Induction decision tree

    Args:
        instances(list of list of objs): A list of instances
        attributes(list of objs): the attributes
        header(list of strs): A list of the headers
        attribute_domains(dictionary of lists of obj): the domains on each attribute

    Returns:
        tree(list of objs): the tree
    """
    split_attribute = select_attribute(current_instances, available_attributes, header)
    available_attributes.remove(split_attribute)
    tree = ["Attribute", split_attribute]

    partitions = partition_instances(current_instances, split_attribute, header, attribute_domains)
    size = 0
    for att_value, att_partition in partitions.items():
        size += len(att_partition)

    for att_value, att_partition in partitions.items():
        value_subtree = ["Value", att_value]
        if len(att_partition) > 0 and same_class_label(att_partition):
            value_subtree.append(["Leaf", att_partition[0][-1], len(att_partition), size])
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            pass
        elif len(att_partition) == 0:
            pass
        else:
            subtree = tdidt(att_partition, available_attributes.copy(), header, attribute_domains)
            value_subtree.append(subtree)
        tree.append(value_subtree)

    return tree

def tdidt_predict(tree, instance, header):
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1]

    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            return tdidt_predict(value_list[2], instance, header)

def tdidt_print(tree, rule, attribute_names=None, class_name="class"):
    for i, att in enumerate(tree):
        if att == "Attribute":
            if len(rule) != 0:
                rule += " AND "
            if attribute_names is None:
                rule += "IF " + tree[i+1] + " == "
            else:
                rule += "IF " + attribute_names[int(tree[i+1][3:])] + " == "
        elif att == "Value":
            rule += tree[i+1]
        elif att == "Leaf":
            rule += " THEN " + class_name + " = " + tree[i+1]
            print(rule)
        elif type(att) is list:
            tdidt_print(att, rule, attribute_names, class_name)
