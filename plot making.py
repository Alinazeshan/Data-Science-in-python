# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:54:05 2017

@author: Dr Alina
"""
x = 'D:\COURSE programming with python for Data Science\DAT210x-master\Module3\Datasets\wheat.data'

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
filepath = x
student_dataset = pd.read_csv(filepath, index_col = 0 ) 
#%%
#scatter plot
student_dataset.plot.scatter(x = 'G1',y = 'G2')
plt.subtitle()
plt.xlabel()
plt.ylabel()
plt.show()

##
##
### histogram plot
my_series = student_dataset.G3
my_dataframe = student_dataset[['G3','G2','G1']]


my_series.plot.hist(alpha = .50)
my_dataframe.plot.hist(alpha = .50)
#%%
#3D Scatter plot

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt


matplotlib.style.use('ggplot')
filepath = 'D:\COURSE programming with python for Data Science\DAT210x-master\students.data' 
student_dataset = pd.read_csv(filepath, index_col = 0 ) 

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax=fig.add_subplot(111,projection = '3d')
ax.set_xlabel('Final Grade')
ax.set_ylabel('First Grade')
ax.set_zlabel('Alchohol')


ax.scatter(student_dataset.G1,student_dataset.G3,student_dataset['Dalc'],c = 'r', marker = 'o')
