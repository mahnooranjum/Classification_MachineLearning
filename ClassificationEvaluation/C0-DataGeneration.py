###############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#        
#    References:
#        SuperDataScience,
#        Official Documentation
#
###############################################################################

from sklearn.datasets.samples_generator import make_moons
from sklearn.datasets.samples_generator import make_circles
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


######### CLUSTERS ##############

# generate 2d classification dataset
X, y = make_blobs(n_samples=10000, centers=3,cluster_std=2.5, n_features=2)
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue',2:'green'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()


datadict = {'X1': X[:,0],'X2' : X[:,1], 'y': y}
df = pd.DataFrame(data=datadict)
df.to_csv('Clusters.csv')


########## MOONS ########33

# generate 2d classification dataset
X, y = make_moons(n_samples=10000, noise=0.1)
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()


datadict = {'X1': X[:,0],'X2' : X[:,1], 'y': y}
df = pd.DataFrame(data=datadict)
df.to_csv('Moons.csv')

########## MOONSANN ########33

# generate 2d classification dataset
X, y = make_moons(n_samples=1000000, noise=0.1)
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()


datadict = {'X1': X[:,0],'X2' : X[:,1], 'y': y}
df = pd.DataFrame(data=datadict)
df.to_csv('MoonsANN.csv')

###### CIRCLES #########
# generate 2d classification dataset
X, y = make_circles(n_samples=100, noise=0.05)
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()
datadict = {'X1': X[:,0],'X2' : X[:,1], 'y': y}
df = pd.DataFrame(data=datadict)
df.to_csv('Circles.csv')