from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter


def one_label_dataset(label_name):
	'''
	input : one label name (ex. LKneeAngles (1))
	output : dataset (shape : 324,101)
	'''
	dataset = []


	file_name1 = os.listdir("dataset")
	file_name1.sort()

	for one_label in file_name1:

		file_name2 = os.listdir("dataset/" + "%s" %one_label)
		file_name2.sort()
		
		cnt = 0
		for one_excel_file in file_name2:
			
			#try:
			excel_file = r'dataset/%s/%s' %(one_label, one_excel_file)
			excel_array = pd.read_excel(excel_file, sheet_name='Data')

			one_label_data = excel_array[label_name]
			dataset.append(one_label_data[2:])


			#except:
			#	print("=== Not Excel File ===")
			#	continue

			cnt += 1

	
	return dataset

def data_generation(dataset): # 108 101
	
	data_gen_num = 5
	new_dataset = []

	std_dataset = np.std(dataset, axis=0, dtype=np.float64)

	for one_data in dataset:
		new_dataset.append(one_data)
		for i in range(data_gen_num-1):
			plus_minus = np.random.uniform()
			random_value = np.random.rand(101)

			if plus_minus > 0.5:
				one_data_gen = one_data + 3 * random_value
			else:
				one_data_gen = one_data - 3 * random_value

			one_data_gen = savgol_filter(one_data_gen, 51, 3)

			new_dataset.append(one_data_gen)

	new_dataset = np.array(new_dataset)

	return new_dataset
	'''
	num = 0
	for one in new_dataset:
		plt.figure()
		plt.plot(one)
		plt.savefig("save_fig/test/%d.png" %num)
		plt.close()
		num += 1
	'''

	'''
	mean_dataset = np.mean(dataset, axis=0)
	std_dataset = np.std(dataset, axis=0, dtype=np.float64)

	plus_minus = np.random.uniform()
	random_value = np.random.rand(101)

	if plus_minus > 0.5:
		test_dataset = mean_dataset + std_dataset * random_value
	else:
		test_dataset = mean_dataset - std_dataset * random_value

	test_dataset = savgol_filter(test_dataset, 51, 3)

	mean_under_bound = mean_dataset - std_dataset
	mean_upper_bound = mean_dataset + std_dataset


	print(mean_dataset)
	print(std_dataset)
	plt.plot(mean_dataset)
	plt.plot(test_dataset)
	plt.plot(mean_under_bound,'r--')
	plt.plot(mean_upper_bound,'r--')
	plt.show()
	'''

def pickle_file_write(dataset, file_name):
	'''
	input : dataset to write / file_name
	output : pickle file
	'''
	with open("pickle_data/%s.pickle" %file_name, 'wb') as f:
		pickle.dump(dataset,f)
	
def pickle_file_read(file_name):
	with open("pickle_data/%s.pickle" %file_name, 'rb') as f:
		data = pickle.load(f)

	# print(data)
	return data

def save_all_figure(dataset, save_figure_file_path):
	'''
	input : dataset [[1 figure],[2 figure],[3 figure], ... , []]
	output : save figure img
	'''
	cnt = 0
	for i in dataset:
		plt.figure()
		plt.plot(i)
		plt.savefig('%s/%d.png' %(save_figure_file_path,cnt))
		plt.close()
		cnt += 1
		# if cnt == 10:
		# 	break