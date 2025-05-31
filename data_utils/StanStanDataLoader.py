import os
import numpy as np
import glob
import tqdm
import torch
from torch.utils.data import Dataset
import functools

class CMRDataLoader(Dataset):
	def __init__(self, data_path_list, args):
		self.data_path_list = data_path_list
		self.args=args

	def __len__(self):
		return len(self.data_path_list)

	def __getitem__(self, index):
		file_path = self.data_path_list[index]

		data = np.load(file_path)
		points = data['points']
		labels = data['labels']

		points_tensor = torch.from_numpy(points).float()
		labels_tensor = torch.from_numpy(labels).long()

		return points_tensor, labels_tensor

def custom_collate_fn(batch_of_scans, TARGET_NUM_POINTS):
	processed_points_list = []
	processed_labels_list = []

	for (raw_points, raw_labels) in batch_of_scans:
		current_num_points = raw_points.shape[0]

		if current_num_points == TARGET_NUM_POINTS:
			final_points = raw_points
			final_labels = raw_labels
		elif current_num_points > TARGET_NUM_POINTS:
			selected_indices = np.random.choice(current_num_points, size=TARGET_NUM_POINTS, replace=False)
			final_points = raw_points[selected_indices]
			final_labels = raw_labels[selected_indices]
		else:
			num_to_pad = TARGET_NUM_POINTS - current_num_points
			padding_for_points = torch.zeros((num_to_pad, 3))
			final_points = torch.cat((raw_points, padding_for_points))

			padding_for_labels = torch.zeros((num_to_pad,))
			final_labels = torch.cat((raw_labels, padding_for_labels))

		processed_points_list.append(final_points)
		processed_labels_list.append(final_labels)
	
	batched_points = torch.stack(processed_points_list)
	batched_labels = torch.stack(processed_labels_list)

	return batched_points, batched_labels
	
def CreateDataLoaders(root_dir, args, train_split_ratio):
	target_num_points = args.num_points

	file_scan_path = os.path.join(root_dir, 'scan_*.npz')
	data_file_paths = glob.glob(file_scan_path)
	if not len(data_file_paths):
		raise FileNotFoundError

	np.random.shuffle(data_file_paths)

	split_index = int(len(data_file_paths) * train_split_ratio)
	train_files = data_file_paths[:split_index]
	test_files = data_file_paths[split_index:]

	train_dataset = CMRDataLoader(train_files, args)
	test_dataset = CMRDataLoader(test_files, args)

	collate_fn_with_args = functools.partial(custom_collate_fn, TARGET_NUM_POINTS=target_num_points)

	trainDataLoader = torch.utils.data.DataLoader(
		dataset=train_dataset, 
		batch_size=args.batch_size, 
		shuffle=True, 
		num_workers=0,
		collate_fn=collate_fn_with_args,
		drop_last=True
	)
	
	testDataLoader = torch.utils.data.DataLoader(
		dataset=test_dataset, 
		batch_size=args.batch_size, 
		shuffle=False, 
		num_workers=0,
		collate_fn=collate_fn_with_args
	)

	return trainDataLoader, testDataLoader

# Testing
# if __name__=='__main__':
# 	class DummyArgs:
# 		def __init__(self):
# 			self.data_file_path='lidar_scans_20250531_014003' # slurm
# 			self.train_split_ratio=0.7 # train
# 			self.batch_size=20 # slurm
# 			self.num_workers=0 # train
# 			self.num_points=7000 # slurm
			
# 	args = DummyArgs()

# 	trainDataLoader, testDataLoader = CreateDataLoaders(args.data_file_path, args, args.train_split_ratio)

# 	for i, (batch_points, batch_labels) in enumerate(trainDataLoader):
# 		print(f"Batch {i}: Points batch shape: {batch_points.shape}, Labels batch shape: {batch_labels.shape}")

# 	for i, (batch_points, batch_labels) in enumerate(testDataLoader):
# 		print(f"Batch {i}: Points batch shape: {batch_points.shape}, Labels batch shape: {batch_labels.shape}")