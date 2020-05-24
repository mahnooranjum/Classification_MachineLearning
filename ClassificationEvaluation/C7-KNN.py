##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#    
#    References:
#        SuperDataScience,
#        Official Documentation
#
#
##############################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def plot_map(X_set, y_set, classifier, text):
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('pink', 'cyan','cornflowerblue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    plt.title(text)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()
    plt.clf()
    
def plot_model(X_set, y_set, classifier, text):
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('pink', 'cyan','cornflowerblue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'blue','midnightblue'))(i), label = j)
    plt.title(text)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()
    plt.clf()
    
def evaluate(y_test,y_pred):
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print("============================================================")
    print("Accuracy Score: " + str(accuracy))
    # Getting evaluation report
    from sklearn.metrics import classification_report
    print("============================================================")
    print(classification_report(y_test, y_pred))
    print("============================================================")


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

# ========================= K NEAREST NEIGHBORS ===============================
#    
#    Classifier implementing the k-nearest neighbors vote.
#    n_neighbors : int, optional (default = 5)
#    Number of neighbors to use by default for kneighbors queries.
#    
#    p : integer, optional (default = 2)
#    Power parameter for the Minkowski metric. When p = 1, this is equivalent to 
#    using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. 
#    For arbitrary p, minkowski_distance (l_p) is used.
#    
#    metric : string or callable, default ‘minkowski’
#    the distance metric to use for the tree. The default metric is minkowski, 
#    and with p=2 is equivalent to the standard Euclidean metric. 

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

evaluate(y_test, y_pred)
plot_map(X_train, y_train, classifier, 'Boundary')
plot_model(X_train, y_train, classifier, 'Train')
plot_model(X_test, y_test, classifier, 'Test')

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(classifier, X_test, y_test,cmap=plt.cm.Blues)

