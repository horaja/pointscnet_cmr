"""
Author: Benny
Date: Nov 2019
"""
# from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.CustomSceneDataLoader import CustomSceneDataLoader
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=2, type=int, choices=[2], help='training on custom data set')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to your custom .npz data file')
    return parser.parse_args()


def test(model, loader, num_class=2, vote_num=1):
    # mean_correct = []
    classifier = model.eval()
    # class_acc = np.zeros((num_class, 3))

    # Initialize Segmentation Metrics
    total_correct_points = 0
    total_points = 0

    # Initialize Intersection over Union Metrics
    total_intersection = np.zeros(num_class)
    total_union = np.zeros(num_class)

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        # input()
        points = points.transpose(2, 1)
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
    # data_path = 'data/modelnet40_normal_resampled/'
    data_root_dir = os.path.dirname(args.data_file_path)
    if not data_root_dir:
        data_root_dir = './'

    SEED_FOR_SPLIT = 42

    test_dataset = CustomSceneDataLoader(
        root=data_root_dir, 
        args=args, 
        split='test', 
        process_data=False,
        train_split_ratio=0.8,
        random_seed=SEED_FOR_SPLIT
    )

    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    log_string(model_name)
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)

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
                                                       num_class=num_class)
        log_string('Test Overall Accuracy: %f, Mean IoU: %f' % (overall_accuracy, mean_iou))
        for i in range(num_class):
            log_string('Class %s IoU: %f' % (cls_to_tag[i], per_class_iou[i]))

if __name__ == '__main__':
    args = parse_args()
    main(args)
