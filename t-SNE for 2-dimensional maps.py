#t-SNE for 2-dimensional maps
'''
● t-SNE = “t-distributed stochastic neighbor embedding”
● Maps samples to 2D space (or 3D)
● Map approximately preserves nearness of samples
● Great for inspecting datasets
'''
import sklearn
import pandas as pd
from sklearn.cluster import KMeans


from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target

#URL = ''
#data = pd.read_csv(URL)


samples = X
species = y



import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(samples)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()


##
normalized_movements

# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]
#'''
# Scatter plot
plt.scatter(xs,ys,alpha=0.5)
plt.show()
'''
# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
'''
