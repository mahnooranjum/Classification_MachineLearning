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
dataset = pd.read_csv('Moons.csv')
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

# ================================== SVM ======================================
#    
#    Support Vector Classification.
#    The implementation is based on libsvm. 
#    The fit time complexity is more than quadratic with the number of 
#    samples which makes it hard to scale to dataset with more than a couple 
#    of 10000 samples.
#    
#    kernel : string, optional (default=’rbf’)
#    Specifies the kernel type to be used in
#    the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, 
#    precomputed’ or a callable. If none is given, ‘rbf’ will be used. 


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


evaluate(y_test, y_pred)
plot_map(X_train, y_train, classifier, 'SVC Boundary')
plot_model(X_train, y_train, classifier, 'SVC Train')
plot_model(X_test, y_test, classifier, 'SVC Test')

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(classifier, X_test, y_test,cmap=plt.cm.Blues)
