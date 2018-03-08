import pandas as pd

URL = 'https://assets.datacamp.com/production/course_2072/datasets/wine.csv'
df = pd.read_csv(URL)

d = {'total_phenols': df['total_phenols'], 'od280': df['od280']}
samples = pd.DataFrame(data=d)


#Decorrelating your data and dimension reduction
#Visualizing the PCA transformation
from sklearn.decomposition import PCA
model = PCA()
model.fit(samples)
transformed = model.transform(samples)
print(transformed)
print(model.components_)




#Plotting the variances of PCA features
from sklearn import datasets
iris = datasets.load_iris()
samples = iris.data
species = iris.target
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(samples)
features = range(pca.n_components_)

#Plotting the variances of PCA features
plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()
'''
URL = 'https://assets.datacamp.com/production/course_2072/datasets/wine.csv'
df = pd.read_csv(URL)

d = {'total_phenols': df['total_phenols'], 'od280': df['od280']}
samples = pd.DataFrame(data=d)

#The first principal component
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])
# Create a PCA instance: model
model = PCA()
# Fit model to points
model.fit(grains)
# Get the mean of the grain samples: mean
mean = model.mean_
# Get the first principal component: first_pc
first_pc = model.components_[0,:]
# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)
# Keep axes on same scale
plt.axis('equal')
plt.show()
'''

#Dimension reduction
# Represents same data, using less features
# Important part of machine-learning pipelines
# Can be performed using PCA
PCA(n_components=2) #keep 2 features

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(samples)
transformed = pca.transform(samples)
print(transformed.shape)


import matplotlib.pyplot as plt
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
#'''
mean = model.mean_
first_pc = model.components_[0,:]
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)
plt.axis('equal')
#'''
plt.show()

