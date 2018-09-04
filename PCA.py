# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 12:31:17 2017

@author: Dr Alina
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData, PlyElement
from sklearn.decomposition import PCA
from sklearn import preprocessing


df = pd.read_csv('D:\COURSE programming with python for Data Science\DAT210x-master\Module4\Datasets\kidney_disease.csv')
df.head()
df=df.dropna(axis=0)
df.drop(labels = ['id', 'classification', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'], axis=1, inplace=True)


new = df[['bgr','wc','rc']]
new.dtypes


new.rc=pd.to_numeric(new.rc,errors ="coerce")
new.wc=pd.to_numeric(new.wc,errors ="coerce")
new.dtypes
new.describe()
np.var(new)

plt.style.use('ggplot')
pca = PCA(n_components = 2, svd_solver ='full')
pca.fit(new)
pca = pca.transform(new)

print (pca.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Full PCA') 
ax.scatter(pca[:,0], pca[:,1], c='blue', marker='.', alpha=0.75)
plt.show()
#%%

def scaleFeaturesDF():
    # Feature scaling is a type of transformation that only changes the
    # scale, but not number of features. Because of this, we can still
    # use the original dataset's column names... so long as we keep in
    # mind that the _units_ have been altered:

    scaled = preprocessing.StandardScaler().fit_transform(pca)
    scaled = pd.DataFrame(scaled, columns=pca.columns)
    
    print("New Variances:\n", scaled.var())
    print("New Describe:\n", scaled.describe())
    return scaled
scaleFeaturesDF(pca)


#%%
def drawVectors(transformed_features, components_, columns, plt, scaled):
    if not scaled:
        return plt.axes() # No cheating ;-)

    num_columns = len(columns)

    # This funtion will project your *original* feature (columns)
    # onto your principal component feature-space, so that you can
    # visualize how "important" each one was in the
    # multi-dimensional scaling

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ## visualize projections

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Features by importance:\n", important_features)

    ax = plt.axes()

    for i in range(num_columns):
        # Use an arrow to project each original feature as a
        # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax


ax = drawVectors(new, pca.components_, df.columns.values, plt, scaleFeatures)
T  = pd.DataFrame(new)

T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)

plt.show()

