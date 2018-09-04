# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 11:10:06 2017

@author: Dr Alina
"""





from sklearn.datasets import load_iris

from pandas.plotting import parallel_coordinates

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')
x = 'D:\COURSE programming with python for Data Science\DAT210x-master\Module3\Datasets\wheat.data'

data = pd.read_csv(x, index_col = 0 ) 

data.drop(labels=['area', 'perimeter'], axis=1) #drop area and perimeter

parallel_coordinates(data, 'wheat_type',alpha = 0.4)

plt.show()



#%%


from sklearn.datasets import load_iris

from pandas.tools.plotting import andrews_curves

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')
x = 'D:\COURSE programming with python for Data Science\DAT210x-master\Module3\Datasets\wheat.data'

data = pd.read_csv(x, index_col = 0 ) 

data.drop(labels=['area', 'perimeter'], axis=1) #drop area and perimeter

andrews_curves(data, 'wheat_type')

plt.show()