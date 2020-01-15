import sys

from instance_selection import InstanceSelection
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import numpy as np 

data = pd.read_csv("data/iris.csv")
labelEncoder = LabelEncoder()

#Iris dataset has categorical labels.
data["species"] = labelEncoder.fit_transform(data["species"])

data = np.asarray(data)[:3]
print("Dataset shape : ",data.shape)

InstanceSelector = InstanceSelection(data)
InstanceSelector.apply()

print("representative_instances_list : ")
print(InstanceSelector.representative_instances_list)

