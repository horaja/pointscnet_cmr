"""
Author: Benny
Date: Nov 2019
"""
# from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.StanStanDataLoader import CreateDataLoaders
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import sys
import importlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=2, type=int, choices=[2], help='training on custom data set')
    parser.add_argument('--num_points', type=int, default=4096, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--data_root_dir', type=str, required=True, help='Path to your custom .npz data root')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize first scene in first batch')
    return parser.parse_args()

def visualize_scene(scene_data, scene_labels=None, title="Point Cloud Scene", save_path=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if scene_labels is not None:
        cmap = plt.get_cmap('viridis', 2)
        # For older matplotlib, get_cmap might still be okay, but this is the modern way:
        # cmap = mpl.colormaps['viridis'].resampled(2)
        
        scatter = ax.scatter(scene_data[:, 0], scene_data[:, 1], scene_data[:, 2], 
                                                    c=scene_labels.astype(float), # Cast to float if labels are ints, sometimes helps cmap
                                                    cmap=cmap, s=5)
        cbar = fig.colorbar(scatter, ax=ax, ticks=[0.25, 0.75])
        cbar.set_ticklabels(['Ground (0)', 'Cone (1)'])
    else:
        # Original visualization for height coloring
        scatter = ax.scatter(scene_data[:, 0], scene_data[:, 1], scene_data[:, 2], c=scene_data[:, 2], cmap='plasma', s=1)

    ax.set_xlim(scene_data[:, 0].min(), scene_data[:, 0].max())
    ax.set_ylim(scene_data[:, 1].min(), scene_data[:, 1].max())
    ax.set_zlim(scene_data[:, 2].min(), scene_data[:, 2].max())
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=30, azim=45)
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close(fig)


def test(model, loader, num_class=2, vote_num=1, experiment_dir=None, visualize=False):
    # mean_correct = []
    classifier = model.eval()
    # class_acc = np.zeros((num_class, 3))

    # Initialize Segmentation Metrics
    total_correct_points = 0
    total_points = 0

    # Initialize Intersection over Union Metrics
    total_intersection = np.zeros(num_class)
    total_union = np.zeros(num_class)

    visualized_once = False

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points_vis, target = points.cuda(), target.cuda()

        # input()
        points = points_vis.transpose(2, 1)
        # vote_pool = torch.zeros(target.size()[0], num_class).cuda()
        first_pred_example = None
        
        for _ in range(vote_num):
            pred_output, _, = classifier(points)
            if first_pred_example is None:
                first_pred_example = pred_output
                vote_pool = torch.zeros_like(first_pred_example).cuda()
            vote_pool += pred_output
        pred = vote_pool / vote_num

        pred_choice = pred.data.max(1)[1]

        if not visualized_once and j == 0 and visualize:
            points_single_np = points_vis[0].cpu().numpy()
            target_single_np = target[0].cpu().numpy()
            pred_choice_single_np = pred_choice[0].cpu().numpy()

            vis_dir = os.path.join(experiment_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            visualize_scene(points_single_np, target_single_np,
                            title='Ground Truth Labels (First Scene)',
                            save_path=os.path.join(vis_dir, "ground_truth_visualization.png"))
            visualize_scene(points_single_np, pred_choice_single_np,
                            title='Prediction Labels (First Scene)',
                            save_path=os.path.join(vis_dir, "pred_visualization.png"))
            visualized_once = True


        # All classification stuff below
        # print(pred_choice)
        # print("pred_choice.shape", pred_choice.shape)
        # for cat in np.unique(target.cpu()):
        #     # print('\n',cat.shape)
        #     # print(pred_choice[target == cat])
        #     classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
        #     # print("classacc.shape", classacc.shape)
        #     class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
        #     class_acc[cat, 1] += 1

        

        correct_points = pred_choice.eq(target.long().data).cpu().sum().item()
        total_correct_points += correct_points
        total_points += target.numel()

        target_np = target.cpu().numpy()
        pred_choice_np = pred_choice.cpu().numpy()

        for class_id in range(num_class):
            intersection = np.sum((pred_choice_np == class_id) & (target_np == class_id))
            union = np.sum((pred_choice_np == class_id) | (target_np == class_id))

            total_intersection[class_id] += intersection
            total_union[class_id] += union
        
        # mean_correct.append(correct.item() / float(points.size()[0]))

    overall_accuracy = total_correct_points / total_points if total_points > 0 else 0.0

    # Calculate per-class IoU and mean IoU
    per_class_iou = np.zeros(num_class)
    for class_id in range(num_class):
        if total_union[class_id] > 0:
            per_class_iou[class_id] = total_intersection[class_id] / total_union[class_id]
        else:
            per_class_iou[class_id] = 0.0

    mean_iou = np.mean(per_class_iou)

    return overall_accuracy, mean_iou, per_class_iou

    # class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]

    # class_mean_acc = np.mean(class_acc[:, 2])
    # instance_acc = np.mean(mean_correct)
    # return instance_acc, class_mean_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir
    sys.path.append(experiment_dir)
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    _, testDataLoader = CreateDataLoaders(
        root_dir=args.data_root_dir,
        args=args,
        train_split_ratio = 0.7
    )

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    log_string(model_name)
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is NOT available.")

    # for name, parameters in classifier.named_parameters():
    #     print(name, ':', parameters)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # if num_class == 10:
    #     catfile = os.path.join('./data/modelnet40_normal_resampled', 'modelnet10_shape_names.txt')
    # else:
    #     catfile = os.path.join('./data/modelnet40_normal_resampled', 'modelnet40_shape_names.txt')

    # cats = [line.rstrip() for line in open(catfile)]
    # cls_to_tag = dict(zip(range(len(cats)), cats))

    cls_to_tag = {0: 'Ground', 1: 'Cone'}

    with torch.no_grad():
        overall_accuracy, mean_iou, per_class_iou = test(classifier.eval(), testDataLoader, vote_num=args.num_votes,
                                                       num_class=num_class, experiment_dir=experiment_dir, visualize=args.visualize)
        log_string('Test Overall Accuracy: %f, Mean IoU: %f' % (overall_accuracy, mean_iou))
        for i in range(num_class):
            log_string('Class %s IoU: %f' % (cls_to_tag[i], per_class_iou[i]))
        log_string(f"Visualizations (if generated) are saved in: {os.path.join(experiment_dir, 'visualizations')}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
