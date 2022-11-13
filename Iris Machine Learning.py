'''
This machine learning program classifies and learns to differentiate between 
three species of flowers (in iris dataset) by learning from the data provided. 
It then can make a prediction based on patterns it found and save a pdf of the
decision tree.
'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# indices for testing, the first of each different flower
test_idx = [0, 50, 100]

# training data
# makes a list of all the flowers excluding the 
# first of each new flower (reserved for testing)

train_target = np.delete(iris.target, test_idx) # the name of the flowers
train_data = np.delete(iris.data, test_idx, axis=0) # the data of the flowers

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier() # creates new classifier
clf.fit(train_data, train_target) # trains it with training data

# The actual label of the flowers
print(test_target)

# What the program predicts
print(clf.predict(test_data))

print(iris.feature_names)
print(iris.target_names)
print(test_data)

# Code to visualize the decision tree
# import graphviz

# dot_data = tree.export_graphviz(clf, out_file=None,
#                             feature_names=iris.feature_names,
#                             class_names=iris.target_names,
#                             filled=True, rounded=True,
#                             impurity=False)
# graph = graphviz.Source(dot_data) 
# graph.render("irisML") 