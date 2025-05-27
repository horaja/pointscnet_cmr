'''
CustomSceneDataLoader.py
Loads custom .npz point cloud data with per-point labels for semantic segmentation.
'''
import os
import numpy as np
import warnings
import torch
from torch.utils.data import Dataset
from tqdm import tqdm # Used for progress bar during data processing

def pc_normalize(pc):
  """
  Normalize point cloud to a unit sphere.
  Input: pc [N_points, 3]
  Output: normalized_pc [N_points, 3]
  """
  centroid = np.mean(pc, axis=0)
  pc = pc - centroid
  m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
  pc = pc / m
  return pc

class CustomSceneDataLoader(Dataset):
  def __init__(self, root, args, split='train', process_data=False, train_split_ratio=0.8, random_seed=None):
    """
    Args:
      root (str): Path to the directory containing your custom .npz file.
      args (argparse.Namespace): Arguments including num_point, use_normals, data_file_path.
      split (str): 'train' or 'test'.
      process_data (bool): If True, loads and pre-processes data once for the current split.
      train_split_ratio (float): Ratio of data to use for training (e.g., 0.8 for 80%).
      random_seed (int, optional): Seed for random number generator for reproducible splits.
    """
    self.root = root
    self.npoints = args.num_point
    self.process_data = process_data
    self.use_normals = args.use_normals
    self.split = split

    # Use the data_file_path argument directly if it's the full path to the NPZ file.
    # Otherwise, construct it from root and a base filename.
    if os.path.isfile(args.data_file_path): # Check if it's a full path to a file
      self.data_filepath = args.data_file_path
    # else: # Assume args.data_file_path might be just a filename, and root is the directory
		# 	# This fallback might be needed if train.py changes how it passes paths.
		# 	# For your current train.py, args.data_file_path is full, so the 'if' branch is taken.
		#   base_filename = 'my_custom_scenes_4096pts.npz' # Default if not a full path
		# 	self.data_filepath = os.path.join(self.root, base_filename)


    if not os.path.exists(self.data_filepath):
      raise FileNotFoundError(f"Custom data file not found: {self.data_filepath}. Please generate it first.")

    self.all_scenes_data_npz = np.load(self.data_filepath, allow_pickle=True) # allow_pickle=True if needed for some npz structures
    
    all_scene_keys = sorted([f for f in self.all_scenes_data_npz.files if f.startswith('scene_')])
    all_label_keys = sorted([f for f in self.all_scenes_data_npz.files if f.startswith('labels_')])

    assert len(all_scene_keys) == len(all_label_keys), "Mismatch between number of scenes and label sets."
    if len(all_scene_keys) < 2 and (train_split_ratio < 1.0 and train_split_ratio > 0.0) : # Need at least 2 for a meaningful split unless ratio is 0 or 1
        warnings.warn(f"Only {len(all_scene_keys)} scene(s) found. Train/test split might not be meaningful or possible as configured.")

    scene_label_key_pairs = list(zip(all_scene_keys, all_label_keys))
    
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(scene_label_key_pairs) # Shuffle before splitting

    total_scenes = len(scene_label_key_pairs)
    split_idx = int(total_scenes * train_split_ratio)

    current_selected_pairs = []
    if self.split == 'train':
        current_selected_pairs = scene_label_key_pairs[:split_idx]
    elif self.split == 'test':
        current_selected_pairs = scene_label_key_pairs[split_idx:]
    else:
        warnings.warn(f"Split '{self.split}' is not 'train' or 'test'. Using all {total_scenes} scenes for this instance.")
        current_selected_pairs = scene_label_key_pairs
    
    self.current_scene_keys = [pair[0] for pair in current_selected_pairs]
    self.current_label_keys = [pair[1] for pair in current_selected_pairs]
    
    self.len = len(self.current_scene_keys)
    
    if self.len == 0 and total_scenes > 0 and (train_split_ratio > 0 and train_split_ratio < 1):
        # This condition indicates an issue if a split was expected to yield samples but didn't
        warnings.warn(f"Split '{self.split}' with train_split_ratio={train_split_ratio} resulted in 0 samples from {total_scenes} total scenes. Check configuration.")

    if self.process_data:
      self.list_of_points = []
      self.list_of_labels = []
      if self.len > 0:
          print(f'Processing data for {self.split} split from {self.data_filepath} ({self.len} scenes)...')
          for i in tqdm(range(self.len), total=self.len, desc=f"Processing {self.split} data"):
            scene_key = self.current_scene_keys[i]
            label_key = self.current_label_keys[i]

            points = self.all_scenes_data_npz[scene_key].astype(np.float32)
            labels = self.all_scenes_data_npz[label_key].astype(np.int32)

            if points.shape[0] != self.npoints:
              choice = np.random.choice(points.shape[0], self.npoints, replace=True)
              points = points[choice]
              labels = labels[choice]

            points[:, 0:3] = pc_normalize(points[:, 0:3])

            if self.use_normals:
              # Assuming normals are expected as 3 channels and data doesn't have them
              warnings.warn("Custom data does not have normals. Concatenating zeros for normals as use_normals is True.")
              points_xyz = points[:, 0:3]
              dummy_normals = np.zeros_like(points_xyz)
              points = np.concatenate((points_xyz, dummy_normals), axis=1)
            else:
                points = points[:, 0:3] # Ensure only XYZ if not using normals
            
            self.list_of_points.append(points)
            self.list_of_labels.append(labels)
      else:
          print(f"No data to process for {self.split} split ({self.len} scenes).")

      # Close the NPZ file handle only if it was successfully opened and data is processed
      if hasattr(self, 'all_scenes_data_npz') and self.all_scenes_data_npz.zip is not None:
          self.all_scenes_data_npz.close()
          # print(f"NPZ file closed for {self.split} split after processing.")
    
    log_message_len = self.len if self.len > 0 else "0 (check split ratio and total scenes)"
    print(f'The size of {self.split} custom data is {log_message_len}')

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    if index >= self.len: # Should not happen if DataLoader uses __len__ correctly
        raise IndexError(f"Index {index} out of bounds for dataset split '{self.split}' of size {self.len}")

    if self.process_data:
      if not self.list_of_points or not self.list_of_labels: # Should not happen if len > 0
          raise RuntimeError(f"Data not processed for split '{self.split}', but process_data is True and accessed via __getitem__.")
      point_set_np = self.list_of_points[index]
      label_np = self.list_of_labels[index]
    else:
      # This block runs if process_data is False.
      # Ensure self.all_scenes_data_npz is accessible. It should remain open if process_data=False.
      if not hasattr(self, 'all_scenes_data_npz') or self.all_scenes_data_npz.zip is None :
          # This state indicates an issue, as the npz file should be open if not processed.
           raise RuntimeError("NPZ file is closed or not loaded, but process_data is False. Initialize with process_data=True or manage NPZ file handle carefully.")

      scene_key = self.current_scene_keys[index]
      label_key = self.current_label_keys[index]
      
      point_set_np = self.all_scenes_data_npz[scene_key].astype(np.float32)
      label_np = self.all_scenes_data_npz[label_key].astype(np.int32)

      if point_set_np.shape[0] != self.npoints:
        choice = np.random.choice(point_set_np.shape[0], self.npoints, replace=True)
        point_set_np = point_set_np[choice]
        label_np = label_np[choice]
      
      point_set_np[:, 0:3] = pc_normalize(point_set_np[:, 0:3])

      if self.use_normals:
        # Assuming normals are expected as 3 channels
        points_xyz = point_set_np[:, 0:3]
        dummy_normals = np.zeros_like(points_xyz)
        point_set_np = np.concatenate((points_xyz, dummy_normals), axis=1)
      else:
        point_set_np = point_set_np[:, 0:3]

    return torch.from_numpy(point_set_np).float(), torch.from_numpy(label_np).long()


# # Example usage (for testing this DataLoader locally)
# if __name__ == '__main__':
# 	# This requires a dummy args object and a custom_scenes_4096pts.npz file
# 	class DummyArgs:
# 		def __init__(self):
# 				self.num_point = 4096 # Matches N in data generation
# 				self.use_normals = False # Your custom data has no normals, so this should be False
# 				self.num_category = 2 # Ground (0) and Cone (1)
# 				self.train_split_ratio = 0.8
# 				self.random_seed = 42
# 				self.data_file_path = 'data/my_custom_scenes_4096pts.npz'

# 	dummy_args = DummyArgs()
	
# 	# Ensure you have 'my_custom_scenes_4096pts.npz' in a 'data' folder
# 	# relative to where you run this script, e.g.:
# 	# ./data/my_custom_scenes_4096pts.npz
	
# 	# You might need to temporarily generate the data locally if you haven't yet
# 	# from generate_custom_data import generate_and_save_custom_scenes
# 	# generate_and_save_custom_scenes(num_scenes=5, N=4096, save_path='./data/my_custom_scenes_4096pts.npz')

# 	try:
# 		dataset = CustomSceneDataLoader(root='./data/', args=dummy_args, split='train', process_data=True)
# 		print(f"\nDataLoader loaded {len(dataset)} scenes.")
		
# 		# Test fetching an item
# 		points, labels = dataset[0]
# 		print(f"Sample 0 - Points shape: {points.shape}, Labels shape: {labels.shape}")
# 		print(f"Sample 0 - Unique labels: {np.unique(labels)}, Points data type: {points.dtype}, Labels data type: {labels.dtype}")

# 		# Test with a PyTorch DataLoader
# 		import torch
# 		data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 for local testing
# 		for i, (batch_points, batch_labels) in enumerate(data_loader):
# 			print(f"Batch {i}: Points batch shape: {batch_points.shape}, Labels batch shape: {batch_labels.shape}")
# 			if i > 2: # Print a few batches
# 				break

# 	except FileNotFoundError as e:
# 		print(f"Error: {e}")
# 		print("Please ensure 'my_custom_scenes_4096pts.npz' exists in a 'data' subdirectory relative to this script.")
# 	except Exception as e:
# 		print(f"An unexpected error occurred during DataLoader test: {e}")