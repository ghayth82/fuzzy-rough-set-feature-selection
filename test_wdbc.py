import sys
from instance_selection import InstanceSelection
from feature_selection import FeatureSelection

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
import time
from pprint import pprint as pp 
import numpy as np 

data = pd.read_csv("data/wdbc.data",header=None)

labelEncoder = LabelEncoder()
minMaxscaler = MinMaxScaler()
standardScaler= StandardScaler()

#WDBC dataset has categorical labels.
#2nd column is target label. We encode it
data.iloc[:,1] =labelEncoder.fit_transform(data.iloc[:,1])

#Since the algorithm expects label to be the last column
data.iloc[:,1],data.iloc[:,31] = data.iloc[:,31],data.iloc[:,1]


#Standardise features except label
data.iloc[:,:-1] = standardScaler.fit_transform(data.iloc[:,:-1])

data = np.asarray(data)

time_taken = {}

if len(sys.argv)<2:
    MAX_ITERATIONS = len(data)
else:
    MAX_ITERATIONS = int(sys.argv[1])

for size in range(1,MAX_ITERATIONS):
    np.random.shuffle(data)
    test_data = data[:size]
    start_time = time.time()
    InstanceSelector = InstanceSelection(test_data)
    InstanceSelector.apply()
    feature_selection_obj = FeatureSelection(InstanceSelector.representative_instances_list[0])
    feature_selection_obj.apply(InstanceSelector)
    print("Instance set : ",InstanceSelector.representative_instances_list)
    print("Feature set : ",feature_selection_obj.rep_feature_set)
    end_time = time.time()
    time_taken[size] = end_time-start_time
    print(len(InstanceSelector.representative_instances_list[0]))


dataset_size,running_time = list(time_taken.keys()),list(time_taken.values())


plt.plot(dataset_size,running_time)
plt.xlabel("No. of rows")
plt.ylabel("Time taken")
plt.show()
plt.savefig("plots/running_time_wdbc_%d.png"%MAX_ITERATIONS)