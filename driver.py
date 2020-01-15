from instance_selection import InstanceSelection

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
	print(instance_selection_obj.representative_instances_list)


__main__()