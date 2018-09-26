import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from preprocess import process_training_data

dataset = 'audio/'
num_chunks = 50

print("Preprocessing dataset.")
features, labels = process_training_data(dataset, num_chunks)

X = features.T
Y = labels.flatten()

print("Creating an 80-20 train-test split.")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

print("Fitting the logistic regression model.")
clf = LogisticRegressionCV()
clf.fit(X_train, Y_train)

print("Evaluating the model.")
print("Training set accuracy =", np.mean(clf.predict(X_train) == Y_train) * 100)
print("Testing set accuracy =", np.mean(clf.predict(X_test) == Y_test) * 100)
