##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#    
#    References:
#        Official Documentation
#
#
##############################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# iloc gets data via numerical indexes
# .values converts from python dataframe to numpy object
dataset = pd.read_csv('Circles.csv')
X = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 3].values

from matplotlib.colors import ListedColormap
for i, j in enumerate(np.unique(y)):
    plt.scatter(X[y == j, 0], X[y == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Dataset')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
plt.clf()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred).flatten()

# Making the Confusion Matrix and predicting accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Getting evaluation report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Visualizing
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('lightsalmon', 'greenyellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SGDClassifier Train Set')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
plt.clf()


# Visualizing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('lightsalmon', 'greenyellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SGDClassifier Test Set')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

