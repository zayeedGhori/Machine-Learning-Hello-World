'''
This program uses a hard coded NearestNeighbours machine learning algorithm to predict a certain number of test
points and then calculatest the accuracy of the classifier.
'''

from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# returns the euclidean distance between two points
def euc(a, b):
    return distance.euclidean(a, b)

# Classifier that I made based on KNearestNeighbours algorithm
class BarebonesNN:

    # Saves the training data to the class
    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

    # Predicts every label based on test_data by identifying each
    # the label of the closest point according to training data
    def predict(self, test_data):
        predictions = []

        # for every test data point, check which training 
        # data point is closest and append to predictions
        for row in test_data:
            label = self.closest(row)
            predictions.append(label)

        return predictions
    
    # Finds the closest point in training data to test point
    def closest(self, test_point):
        closest_dist = euc(test_point, self.train_data[0]) # sets dist to dist from first training point
        closest_index = 0

        # checks every other training point and records index and dist if less than closest_dist
        for i in range(1, len(self.train_data)):
            dist = euc(test_point, self.train_data[i])

            if (closest_dist > dist):
                closest_dist = dist
                closest_index = i
        
        return self.train_labels[closest_index]


# creates new iris dataset
iris = load_iris()

data = iris.data
labels = iris.target

# splits half our dataset into train and test features and labels
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.5)

# creates new classifier from BarebonesNN
clf = BarebonesNN()

# saves training data to model (as x and y components)
clf.fit(train_data, train_labels)

# saves predictions made from test_data
predictions = clf.predict(test_data)

# prints the accuracy of classifier
print(accuracy_score(test_labels, predictions))