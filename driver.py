from instance_selection import InstanceSelection
from feature_selection import FeatureSelection

def __main__():

	matrix = [[0.7, 0.9, 0.4, 0.6, 1],
			  [0.6, 0.9, 0.3, 0.7, 1],
			  [0.6, 0.8, 0.3, 0.5, 1],
			  [0.3, 0.5, 0.7, 0.2, 1],
			  [0.3, 0.4, 0.8, 0.3, 1],
			  [0.4, 0.5, 0.6, 0.3, 1],
			  [0.9, 0.4, 0.5, 0.9, 0],
			  [0.8, 0.5, 0.4, 0.8, 0],
			  [0.2, 0.6, 0.7, 1.0, 0],
			  [0.1, 0.7, 0.8, 0.8, 0]]

	instance_selection_obj = InstanceSelection(matrix)
	instance_selection_obj.apply()
	feature_selection_obj = FeatureSelection(instance_selection_obj.representative_instances_list[0])
	feature_selection_obj.apply(instance_selection_obj)
	print(instance_selection_obj.representative_instances_list)
	print(feature_selection_obj.rep_feature_set)

__main__()