import sklearn
import pandas as pd
from sklearn.cluster import KMeans
'''

from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target


model = KMeans(n_clusters=3)
model.fit(X)
labels = model.predict(X)
print(labels)




#Cluster labels for new samples
new_samples = [[ 5.7, 4.4, 1.5, 0.4],
               [ 6.5, 3., 5.5, 1.8],
               [ 5.8, 2.7, 5.1, 1.9]]
new_labels = model.predict(new_samples)
print(new_labels)




#Scatter plots the predicted labels will have a different color
import matplotlib.pyplot as plt
xs = X[:,0]
ys = X[:,2]
plt.scatter(xs, ys, c=labels)
plt.show()

#crosstabulation
df = pd.DataFrame({'labels': labels, 'actual': y})
print(df)
ct = pd.crosstab(df['labels'], df['actual'])
print(ct)


#Measuring inertia
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)
print(model.inertia_)
 
#from sklearn.cluster import KMeans
#from sklearn import datasets
#iris = datasets.load_iris()
#X, y = iris.data, iris.target
 
cs = range(1, 8)
inertias = []
for clusters in cs:
    clusters +=1
    model = KMeans(n_clusters=clusters)
    model.fit(X)
    inertias.append(model.inertia_)

plt.plot(cs, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(cs)
plt.show()
#Choose lowest part of elbow
'''

#Pipeline and StandardScaling
data = pd.read_csv('https://assets.datacamp.com/production/course_2072/datasets/wine.csv')

samples = data.drop(labels=['class_label','class_name'], axis=1)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
labels = model.fit_predict(samples)
df = pd.DataFrame({'labels': labels,'varieties': data['class_name']})
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)
#Feature variances are high, needs to be scaled

#Pipelines combine StandardScaler, then KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)
labels = pipeline.predict(samples)

df = pd.DataFrame({'labels': labels, 'varieties': data['class_name']})
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)


#Principal Component Analysis
● PCA = "Principal Component Analysis"
● Fundamental dimension reduction technique
● First step "decorrelation" (considered here)
● Second step reduces dimension (considered later)








