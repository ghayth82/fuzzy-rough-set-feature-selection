import sys
from instance_selection import InstanceSelection
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
import time

import numpy as np 

data = pd.read_csv("data/iris.csv")
labelEncoder = LabelEncoder()
minMaxscaler = MinMaxScaler()
standardScaler= StandardScaler()

#Iris dataset has categorical labels.
data["species"] = labelEncoder.fit_transform(data["species"])
data = standardScaler.fit_transform(data)

data = np.asarray(data)

time_taken = {}
MAX_ITERATIONS = int(sys.argv[1]) or len(data)
for size in range(1,MAX_ITERATIONS):
    np.random.shuffle(data)
    start_time = time.time()
    test_data = data[:size]
    print("Dataset shape : ",test_data.shape)
    # print(test_data)
    InstanceSelector = InstanceSelection(test_data)
    InstanceSelector.apply()
    end_time = time.time()
    time_taken[size] = end_time-start_time
    # print("representative_instances_list : ")
    print(len(InstanceSelector.representative_instances_list[0]))


dataset_size,running_time = list(time_taken.keys()),list(time_taken.values())
# for size,time in zip(dataset_size,running_time):
#     print(size,time)

plt.plot(dataset_size,running_time)
plt.xlabel("No. of rows")
plt.ylabel("Time taken")
plt.show()
plt.savefig("plots/plot_%d.png"%MAX_ITERATIONS)