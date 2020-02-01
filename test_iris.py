import sys
from instance_selection import InstanceSelection
from feature_selection import FeatureSelection

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
import time

import numpy as np 
np.random.seed(0)

from test_algos import *

data = pd.read_csv("data/iris.csv")
labelEncoder = LabelEncoder()
minMaxscaler = MinMaxScaler()
standardScaler= StandardScaler()

#Iris dataset has categorical labels.
data["species"] = labelEncoder.fit_transform(data["species"])
data.iloc[:,:-1] = standardScaler.fit_transform(data.iloc[:,:-1])

data = np.asarray(data)
time_taken = {}

if len(sys.argv)<2:
    MAX_ITERATIONS = len(data)
else:
    MAX_ITERATIONS = int(sys.argv[1])

for size in range(MAX_ITERATIONS-1,MAX_ITERATIONS):
    np.random.shuffle(data)
    test_data = data[:size]
    
    start_time1 = time.time()
    InstanceSelector = InstanceSelection(test_data)
    InstanceSelector.apply()
    end_time1 = time.time()
    algo1_time = end_time1-start_time1

    start_time2 = time.time()
    feature_selection_obj = FeatureSelection(InstanceSelector.representative_instances_list[0])
    feature_selection_obj.apply(InstanceSelector)
    end_time2 = time.time()

    representative_instances = InstanceSelector.representative_instances_list
    feature_set = list(feature_selection_obj.rep_feature_set)
    
    print("{:27s} |{:8s}|{:8s}|{:8s}|{:8s}\n".format("Model","Accuracy","Precision","Recall","F1-score"))
    for model in models:
        accuracy,precision,recall,f1 = test_model(data[:,feature_set],data[:,-1],model)
        print("{:27s} |{:8.2f}|{:8.2f}|{:8.2f}|{:8.2f}".format(type(model).__name__,accuracy,precision,recall,f1))

    # print("Algo 1 time : ",algo1_time)
    # print("Algo 2 time : ",end_time2-start_time2)
    # print("Instance set : ",representative_instances)
    # print("Feature set : ",feature_set)
    # end_time = time.time()
    # time_taken[size] = end_time-start_time


# dataset_size,running_time = list(time_taken.keys()),list(time_taken.values())


# plt.plot(dataset_size,running_time)
# plt.xlabel("No. of rows")
# plt.ylabel("Time taken")
# plt.show()
# plt.savefig("plots/running_time_iris_%d.png"%MAX_ITERATIONS)