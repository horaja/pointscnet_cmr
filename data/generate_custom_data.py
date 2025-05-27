import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_custom_scene(N=10000, scene_size=(1.0, 1.0), cone_height=0.1, cone_radius=0.05):
	num_cones = np.random.randint(1, 6)
	cone_centers = np.random.uniform(low=0.2, high=0.8, size=(num_cones, 2))
	
	# Target distribution of points
	target_cone_ratio = 0.4 # 40% of points for cones
	
	points_list = []
	labels_list = []

	# --- Cone Points Generation ---
	for center_x, center_y in cone_centers:
		# Generate more points than potentially needed to ensure enough valid points
		# Then, randomly sample from these to maintain target count
		num_initial_cone_pts = int(N * target_cone_ratio / num_cones * 1.5) # Generate 50% more than target average
		
		r_gen = np.sqrt(np.random.uniform(0, 1, num_initial_cone_pts)) * cone_radius
		theta_gen = np.random.uniform(0, 2 * np.pi, num_initial_cone_pts)
		z_gen = np.random.uniform(0, 1, num_initial_cone_pts) * cone_height
		
		max_r_at_z = (1 - z_gen / cone_height) * cone_radius
		valid = r_gen <= max_r_at_z

		x_cone = center_x + r_gen[valid] * np.cos(theta_gen[valid])
		y_cone = center_y + r_gen[valid] * np.sin(theta_gen[valid])
		z_cone = z_gen[valid]

		cone_pts_current = np.stack([x_cone, y_cone, z_cone], axis=1)
		cone_lbls_current = np.ones(cone_pts_current.shape[0], dtype=np.int32) # Label 1 for cone

		points_list.append(cone_pts_current)
		labels_list.append(cone_lbls_current)

	# Concatenate all generated cone points
	all_cone_points = np.concatenate(points_list, axis=0) if points_list else np.empty((0, 3))
	all_cone_labels = np.concatenate(labels_list, axis=0) if labels_list else np.empty((0,), dtype=np.int32)
	
	# --- Ground Points Generation ---
	# Determine how many ground points are needed to reach N total points
	num_ground_points_needed = N - all_cone_points.shape[0]
	
	if num_ground_points_needed < 0: # If too many cone points, resample from cones
		print(f"Warning: Generated {all_cone_points.shape[0]} cone points, more than N={N*target_cone_ratio}. Resampling cones.")
		idx = np.random.choice(all_cone_points.shape[0], int(N * target_cone_ratio), replace=False)
		all_cone_points = all_cone_points[idx]
		all_cone_labels = all_cone_labels[idx]
		num_ground_points_needed = N - all_cone_points.shape[0] # Recalculate

	x_ground = np.random.uniform(0, scene_size[0], num_ground_points_needed)
	y_ground = np.random.uniform(0, scene_size[1], num_ground_points_needed)
	z_ground = np.zeros(num_ground_points_needed)
	ground_pts = np.stack([x_ground, y_ground, z_ground], axis=1)
	ground_labels = np.zeros(num_ground_points_needed, dtype=np.int32) # Label 0 for ground

	# Combine all points and labels
	point_cloud = np.concatenate([ground_pts, all_cone_points], axis=0)
	point_labels = np.concatenate([ground_labels, all_cone_labels], axis=0)

	# Ensure point_cloud and point_labels always have exactly N points by resampling/padding
	if point_cloud.shape[0] != N:
		# print(f"Warning: Final scene has {point_cloud.shape[0]} points, expected {N}. Resampling to N.")
		idx = np.random.choice(point_cloud.shape[0], N, replace=True) # Use replace=True for padding if necessary
		point_cloud = point_cloud[idx]
		point_labels = point_labels[idx]

	# Shuffle points and labels together to mix ground and cone points
	shuffled_indices = np.arange(point_cloud.shape[0])
	np.random.shuffle(shuffled_indices)
	point_cloud = point_cloud[shuffled_indices]
	point_labels = point_labels[shuffled_indices]

	return point_cloud, point_labels

def generate_and_save_custom_scenes(num_scenes=5, N=10000, save_path='custom_scenes.npz'):
	scenes_data = {}
	
	for i in range(num_scenes):
		scene_points, scene_lbls = generate_custom_scene(N=N)
		
		scenes_data[f'scene_{i}'] = scene_points.astype(np.float32)
		scenes_data[f'labels_{i}'] = scene_lbls.astype(np.int32)
		print(f"Generated scene {i} with {scene_points.shape[0]} points and {scene_lbls.shape[0]} labels")
			
	np.savez_compressed(save_path, **scenes_data) # Pass the combined dictionary
	print(f"\nSaved {num_scenes} scenes with labels to '{save_path}'")

# def visualize_scene(scene_data, scene_labels=None, title="Point Cloud Scene"):
# 	fig = plt.figure(figsize=(10, 7))
# 	ax = fig.add_subplot(111, projection='3d')

# 	if scene_labels is not None:
# 			cmap = plt.get_cmap('viridis', 2) # Use direct access for colormaps if new matplotlib
# 			# For older matplotlib, get_cmap might still be okay, but this is the modern way:
# 			# cmap = mpl.colormaps['viridis'].resampled(2)
			
# 			scatter = ax.scatter(scene_data[:, 0], scene_data[:, 1], scene_data[:, 2], 
# 														c=scene_labels.astype(float), # Cast to float if labels are ints, sometimes helps cmap
# 														cmap=cmap, s=1)
# 			cbar = fig.colorbar(scatter, ax=ax, ticks=[0.25, 0.75])
# 			cbar.set_ticklabels(['Ground (0)', 'Cone (1)'])
# 	else:
# 			# Original visualization for height coloring
# 			scatter = ax.scatter(scene_data[:, 0], scene_data[:, 1], scene_data[:, 2], c=scene_data[:, 2], cmap='plasma', s=1)


# 	ax.set_xlim(0, 1)
# 	ax.set_ylim(0, 1)
# 	ax.set_zlim(0, 0.4)
# 	ax.set_title(title)
# 	ax.set_xlabel("X (m)")
# 	ax.set_ylabel("Y (m)")
# 	ax.set_zlabel("Z (m)")
# 	plt.show()

# Example usage
# if __name__ == '__main__':
# 	generate_and_save_custom_scenes(num_scenes=10, N=5000, save_path='short_cone_scenes.npz')

# 	# Load and visualize one scene
# 	data = np.load('short_cone_scenes.npz')
# 	visualize_scene(data['scene_0'], data['labels_0'], title="Scene 0 with Short Cones, colored by label")