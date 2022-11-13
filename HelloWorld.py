'''
This machine learning program classifies and learns to differentiate between 
apples and oranges by learning from the data provided. It then can make a 
prediction based on patterns it found.
'''

from sklearn import tree

# training data that will be used to train the classifier with

# the "features" of each fruit are certain facts that can be used to 
# describe it
# the first num in each sublist is the weight in g
# the second is the texture, 0=bumpy, 1=smooth
features = [[150, 0], [170, 0], [140, 1], [130, 1]]

# "answers" or the fruit names that the above datasets are from
labels = ["orange", "orange", "apple", "apple"] 

# creates a new classifier
clf = tree.DecisionTreeClassifier()

# classifier is trained with above data
clf = clf.fit(features, labels)

# predicts what a 160g, bumpy fruit is
print(clf.predict([[160, 0]]))