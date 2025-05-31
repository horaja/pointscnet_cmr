#!/bin/bash

## Job name
#SBATCH --job-name=stanley_data_train
## Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=horaja@andrew.cmu.edu

## Run on a single GPU
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=mind-1-9
# Submit job to GPU queue
#SBATCH -p gpu

## Job memory request
#SBATCH --mem=32gb
## Time limit days-hrs:min:sec
#SBATCH --time 00-4:00:00

## Standard output and error log
#SBATCH --output="/user_data/horaja/workspace/PointSCNet/logs/custom_seg_test_run/job_%j.out"
#SBATCH --error="/user_data/horaja/workspace/PointSCNet/logs/custom_seg_test_run/job_%j.err"

echo "--- Starting Slurm Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID) ---"
echo "Running on host $HOSTNAME"
echo "Allocated gpus: $CUDA_VISIBLE_DEVICES"

module purge
module load anaconda3-2023.03
module load cuda-11.3
module load gcc-6.3.0
echo "Modules loaded: anaconda3-2023.03, cuda-11.3, gcc-6.3.0"

cd /user_data/horaja/workspace/PointSCNet
echo "Current working directory: $(pwd)"

# Create and Activate Conda Environment (only creates if it doesn't exist)
CONDA_ENV_NAME="stanley_inf_env"
if [ ! -d "$CONDA_PREFIX/envs/${CONDA_ENV_NAME}" ]; then
	echo "Creating new conda environment '${CONDA_ENV_NAME}' with Python 3.6.9..."
	conda create -n ${CONDA_ENV_NAME} python=3.6.9 -y
fi
source activate ${CONDA_ENV_NAME}
echo "Conda environment '${CONDA_ENV_NAME}' activated."
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

echo "Installing exact PyTorch 1.10.0 and dependencies..."
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio===0.10.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install numpy h5py tqdm matplotlib tensorboard
echo "Dependencies installed."

echo "--- Starting Custom Data Generation ---"
DATASET_SAVE_PATH="data/lidar_scans_batch_20250531_051901"
# NUM_SCENES_TO_GENERATE=2400
# POINTS_PER_SCENE=4096

mkdir -p data

# python -c "from data.generate_custom_data import generate_and_save_custom_scenes; generate_and_save_custom_scenes(num_scenes=${NUM_SCENES_TO_GENERATE}, N=${POINTS_PER_SCENE}, save_path='${DATASET_SAVE_PATH}')"
echo "--- Custom Data Generation Completed ---"

# # copying the right files (cuz no training...)
# LOG_DIR_FOR_TEST_PY="custom_seg_eval_logs_with_modelnet_weights"
# EXPERIMENT_FULL_PATH="log/classification/${LOG_DIR_NAME}"
# CHECKPOINT_DEST_PATH="${EXPERIMENT_FULL_PATH}/checkpoints"
# MODEL_SRC_PATH="checkpoint/best_model.pth"

# echo "Creating experiment log directory: ${EXPERIMENT_FULL_PATH}"
# mkdir -p "${CHECKPOINT_DEST_PATH}"

# echo "Copying checkpoint from ${MODEL_SRC_PATH} to ${CHECKPOINT_DEST_PATH}/best_model.pth"
# cp "${MODEL_SRC_PATH}" "${CHECKPOINT_DEST_PATH}/best_model.pth"

# we shall see eh
# MODEL_DEF_SRC="models/SCNet.py"
# MODEL_DEF_DEST_PATH="${EXPERIMENT_FULL_PATH}/logs"
# echo "Creating model definition log directory: ${MODEL_DEF_DEST_PATH}"
# mkdir -p "${MODEL_DEF_DEST_PATH}"
# echo "Copying model definition file ${MODEL_DEF_SRC} to ${MODEL_DEF_DEST_PATH}"
# cp "${MODEL_DEF_SRC}" "${MODEL_DEF_DEST_PATH}/SCNet.py"

# Run the Python evaluation script
echo "--- Starting training on Custom Data ---"

python test.py \
	--batch_size 20 \
	--num_point 6500 \
	--log_dir "job_2093767/" \
	--data_root_dir ${DATASET_SAVE_PATH} \
	--visualize


echo "Python script completed"

# Deactivate conda environment
conda deactivate
echo "--- Slurm Job Finished ---"