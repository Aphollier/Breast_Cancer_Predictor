"""test_classifier.py
@author aphollier, Commander-Cross
Alexander Hollier and Michael Waight
"""
import numpy as np

from mysklearn.classifiers import MyNaiveBayesClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier,\
    MyDecisionTreeClassifier,\
    MyRandomForestClassifier

# from in-class #1  (4 instances)
X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
y_train_class_example1 = ["bad", "bad", "good", "good"]

# from in-class #2 (8 instances)
# assume normalized
X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]

y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

# from Bramer
header_bramer_example = ["Attribute 1", "Attribute 2"]
X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]

y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]

# in-class Naive Bayes example (lab task #1)
header_inclass_example = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# RQ5 (fake) iPhone purchases dataset
header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
X_train_iphone = [
    [1, 3, "fair"], # no
    [1, 3, "excellent"], # no
    [2, 3, "fair"], # yes
    [2, 2, "fair"], # yes
    [2, 1, "fair"], # yes
    [2, 1, "excellent"], # no
    [2, 1, "excellent"], # yes
    [1, 2, "fair"], # no
    [1, 1, "fair"], # yes
    [2, 2, "fair"], # yes
    [1, 2, "excellent"], # yes
    [2, 2, "excellent"], # yes
    [2, 3, "fair"], # yes
    [2, 2, "excellent"], # no
    [2, 3, "fair"] # yes
]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

# Bramer 3.2 train dataset
header_train = ["day", "season", "wind", "rain", "class"]
X_train_train = [
    ["weekday", "spring", "none", "none"], # on time
    ["weekday", "winter", "none", "slight"], # on time
    ["weekday", "winter", "none", "slight"], # on time
    ["weekday", "winter", "high", "heavy"], # late
    ["saturday", "summer", "normal", "none"], # on time
    ["weekday", "autumn", "normal", "none"], # very late
    ["holiday", "summer", "high", "slight"], # on time
    ["sunday", "summer", "normal", "none"], # on time
    ["weekday", "winter", "high", "heavy"], # very late
    ["weekday", "summer", "none", "slight"], # on time
    ["saturday", "spring", "high", "heavy"], # cancelled
    ["weekday", "summer", "high", "slight"], # on time
    ["saturday", "winter", "normal", "none"], # late
    ["weekday", "summer", "high", "none"], # on time
    ["weekday", "winter", "normal", "heavy"], # very late
    ["saturday", "autumn", "high", "slight"], # on time
    ["weekday", "autumn", "none", "heavy"], # on time
    ["holiday", "spring", "normal", "slight"], # on time
    ["weekday", "spring", "normal", "none"], # on time
    ["weekday", "spring", "normal", "slight"] # on time
]
y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                 "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                 "very late", "on time", "on time", "on time", "on time", "on time"]

header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_train_interview = [
    ["Senior", "Java", "no", "no"], # False
    ["Senior", "Java", "no", "yes"], # False
    ["Mid","Python", "no","no"], # True
    ["Junior", "Python", "no", "no"], # True
    ["Junior", "R", "yes", "no"], # True
    ["Junior", "R", "yes", "yes"], # False
    ["Mid", "R", "yes", "yes"], # True
    ["Senior", "Python", "no", "no"], # False
    ["Senior", "R", "yes", "no"], # True
    ["Junior", "Python", "yes", "no"], # True
    ["Senior", "Python", "yes", "yes"], # True
    ["Mid", "Python", "no", "yes"], # True
    ["Mid", "Java", "yes", "no"], # True
    ["Junior", "Python", "no", "yes"] # False
]

y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False",
"True", "True", "True", "True", "True", "False"]

# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
tree_interview = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

forest_interview = [
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 3, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 4, 6]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 6]
                    ]
                ]
            ]
        ],
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 2, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 3, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 1, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att1",
                    ["Value", "Java",
                        ["Leaf", "False", 4, 8]
                    ],
                    ["Value", "Python",
                        ["Leaf", "False", 2, 8]
                    ],
                    ["Value", "R",
                        ["Leaf", "True", 2, 8]
                    ],
                ]
            ]
        ]
]

def high_low_discretizer(value):
    """Tells whether a y value should be considered high or low
        Args:
            value(num): a y value
        Returns:
            high_low(str): a string of either high or low
    """
    if value <= 100:
        return "low"
    return "high"

def test_kneighbors_classifier_kneighbors():
    """Tests the MyKNeighborsClassifier.fit() function using
    three data sets, a four instance set, an eight instace set
    and a set from Bramer
    """
    knn_clf = MyKNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X_train_class_example1, y_train_class_example1)
    distances, indexes = knn_clf.kneighbors([[0.33, 1]])
    desk_distances = [0.67, 1, 1.053043]
    desk_indexes = [0,2,3]
    assert np.allclose(sorted(distances[0]), sorted(desk_distances))
    assert np.allclose(sorted(indexes[0]), sorted(desk_indexes))

    knn_clf.fit(X_train_class_example2, y_train_class_example2)
    distances, indexes = knn_clf.kneighbors([[3, 2]])
    desk_distances = [0,1.414214, 2]
    desk_indexes = [0,2,4]
    assert np.allclose(sorted(distances[0]), sorted(desk_distances))
    assert np.allclose(sorted(indexes[0]), sorted(desk_indexes))

    knn_clf.fit(X_train_bramer_example, y_train_bramer_example)
    distances, indexes = knn_clf.kneighbors([[5.4, 6.6]])
    desk_distances = [4.609772,4.272002, 3.395585]
    desk_indexes = [0,1,2]
    assert np.allclose(sorted(distances[0]), sorted(desk_distances))
    assert np.allclose(sorted(indexes[0]), sorted(desk_indexes))

def test_kneighbors_classifier_predict():
    """Tests the MyKNeighborsClassifier.predict() function using
    three data sets, a four instance set, an eight instace set
    and a set from Bramer
    """
    knn_clf = MyKNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X_train_class_example1, y_train_class_example1)
    y_predicted = knn_clf.predict([[0.33, 1]])
    y_predicted_solution = ["good"]
    assert y_predicted == y_predicted_solution

    knn_clf.fit(X_train_class_example2, y_train_class_example2)
    y_predicted = knn_clf.predict([[3,2]])
    y_predicted_solution = ["no"]
    assert y_predicted == y_predicted_solution

    knn_clf.fit(X_train_bramer_example, y_train_bramer_example)
    y_predicted = knn_clf.predict([[5.4, 6.6]])
    y_predicted_solution = ["-"]
    assert y_predicted == y_predicted_solution

def test_dummy_classifier_fit():
    """Tests the MyDummyClassifier.fit() function using
    three y_train sets that each have different probabilities
    for their labels, first has the highest chance of being yes,
    second has the highest chance of being no, and third has the highest
    chance of being perhaps? the highest probability label
    should be the .most_common_label
    """
    dmy_clf = MyDummyClassifier()
    X_train = [[value, value] for value in range(100)]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dmy_clf.fit(X_train, y_train)
    desk_label = "yes"
    assert dmy_clf.most_common_label == desk_label

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dmy_clf.fit(X_train, y_train)
    desk_label = "no"
    assert dmy_clf.most_common_label == desk_label

    y_train = list(np.random.choice(["yes", "no", "maybe", "perhaps?"], 100, replace=True, p=[0.05, 0.05, 0.1, 0.8]))
    dmy_clf.fit(X_train, y_train)
    desk_label = "perhaps?"
    assert dmy_clf.most_common_label == desk_label

def test_dummy_classifier_predict():
    """Tests the MyDummyClassifier.predict() function using
    three y_train sets that each have different probabilities
    for their labels, first has the highest chance of being yes,
    second has the highest chance of being no, and third has the highest
    chance of being perhaps? the predict method should return the highest
    probability label.
    """
    dmy_clf = MyDummyClassifier()
    X_train = [[value, value] for value in range(100)]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dmy_clf.fit(X_train, y_train)
    y_predicted = dmy_clf.predict([[1,1],[20,20],[99,99]])
    y_predicted_solution = ["yes","yes","yes"]
    assert y_predicted == y_predicted_solution

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dmy_clf.fit(X_train, y_train)
    y_predicted = dmy_clf.predict([[1,1],[20,20],[99,99]])
    y_predicted_solution = ["no","no","no"]
    assert y_predicted == y_predicted_solution

    y_train = list(np.random.choice(["yes", "no", "maybe", "perhaps?"], 100, replace=True, p=[0.05, 0.05, 0.1, 0.8]))
    dmy_clf.fit(X_train, y_train)
    y_predicted = dmy_clf.predict([[1,1],[20,20],[99,99]])
    y_predicted_solution = ["perhaps?","perhaps?","perhaps?"]
    assert y_predicted == y_predicted_solution

def test_naive_bayes_classifier_fit():
    """Tests the MyNaiveBayesClassifier.fit() function using
    three X_train and y_train sets that we covered in class
    and in the textbook. After being fitted, the priors and
    posteriors of the Classifier are tested against desk
    calculations
    """
    nvb_clf = MyNaiveBayesClassifier()
    nvb_clf.fit(X_train_inclass_example, y_train_inclass_example)
    actual_priors = {"no": 3/8, "yes": 5/8}
    actual_posteriors = {"no" : {0: {1: 2/3, 2: 1/3}, 1: {5: 2/3, 6: 1/3,}}, \
        "yes" : {0: {1: 4/5, 2: 1/5}, 1: {5: 2/5, 6: 3/5}}}
    assert nvb_clf.priors == actual_priors
    assert nvb_clf.posteriors == actual_posteriors

    nvb_clf.fit(X_train_iphone, y_train_iphone)
    actual_priors = {"no": 5/15, "yes": 10/15}
    actual_posteriors = {"no" : {0 :{1: 3/5, 2: 2/5}, 1: {1: 1/5,2: 2/5,3: 2/5}, 2: {"fair": 2/5,"excellent": 3/5}}, \
        "yes" : {0: {1: 2/10, 2: 8/10}, 1: {1: 3/10,2: 4/10,3: 3/10}, 2: {"fair": 7/10,"excellent": 3/10}}}
    assert nvb_clf.priors == actual_priors
    assert nvb_clf.posteriors == actual_posteriors

def test_naive_bayes_classifier_predict():
    """Tests the MyNaiveBayesClassifier.predict() function using
    three X_test sets derived from the train sets that we covered
    in class and in the textbook. After being fitted, the predictions
    are tested against desk calculations
    """
    nvb_clf = MyNaiveBayesClassifier()
    nvb_clf.fit(X_train_inclass_example, y_train_inclass_example)
    X_test = [[1,5], [2,6]]
    X_pred_act = ["yes", "yes"]
    assert nvb_clf.predict(X_test) == X_pred_act

    nvb_clf.fit(X_train_iphone, y_train_iphone)
    X_test = [[1,3,"excellent"], [2,1,"fair"]]
    X_pred_act = ["no", "yes"]
    assert nvb_clf.predict(X_test) == X_pred_act

    nvb_clf.fit(X_train_train, y_train_train)
    X_test = [["weekday", "winter", "high", "heavy"], ["weekday","summer","high","heavy"], ["sunday","summer","normal", "slight"]]
    X_pred_act = ["very late", "on time", "on time"]
    assert nvb_clf.predict(X_test) == X_pred_act

def test_decision_tree_classifier_fit():
    """Tests the MyDecisionTreeClassifier.fits() function using
    one X_test
    """
    dct_clf = MyDecisionTreeClassifier()
    dct_clf.fit(X_train_interview, y_train_interview)
    assert dct_clf.tree == tree_interview

def test_decision_tree_classifier_predict():
    """Tests the MyDecisionTreeClassifier.predict() function using
    one X_test
    """
    dct_clf = MyDecisionTreeClassifier()
    dct_clf.fit(X_train_interview, y_train_interview)
    dct_predict = dct_clf.predict([["Senior", "R", "yes", "no"], ["Senior", "Python", "no", "yes"],
        ["Mid", "Python", "yes", "yes",], ["Junior", "R", "yes", "no"]])
    dct_sol = ["True", "False", "True", "True"]
    assert dct_predict == dct_sol

def test_random_forest_classifier_fit():
    """Tests the MyRandomForest.fit() function using
    one X_test
    """
    rnf_clf = MyRandomForestClassifier(6, 2, 2, 1)
    rnf_clf.fit(X_train_interview, y_train_interview)
    assert rnf_clf.rand_forest == forest_interview

def test_decision_tree_classifier_predict():
    """Tests the MyRandomForestClassifier.predict() function using
    one X_test
    """
    rnf_clf = MyRandomForestClassifier(6, 3, 2, 1)
    rnf_clf.fit(X_train_interview, y_train_interview)
    rnf_predict = rnf_clf.predict([["Senior", "R", "yes", "no"], ["Senior", "Python", "no", "yes"],
        ["Mid", "Python", "yes", "yes",], ["Junior", "R", "yes", "no"]])
    rnf_sol = ["True", "False", "True", "True"]
    assert rnf_predict == rnf_sol
