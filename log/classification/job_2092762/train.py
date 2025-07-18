"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.CustomSceneDataLoader import CustomSceneDataLoader
# from data_utils.ModelNetDataLoader import ModelNetDataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# from torchstat import stat
# from thop import profile
# from thop import clever_format

# default `log_dir` is "runs" - we'll be more specific here


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='SCNet', help='model name [default: SCNet]')
    # parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--num_category', default=2, type=int, choices=[2], help='training on custom data set')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to your custom .npz data file')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, loader, num_class=2):
    mean_correct = []
    # Classification metric :(
    # class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    # Initialize Segmentation Metrics
    total_correct_points = 0
    total_points = 0

    # Initialize Intersection over Union Metrics
    total_intersection = np.zeros(num_class)
    total_union = np.zeros(num_class)

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        # for cat in np.unique(target.cpu()):
        #     classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
        #     class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
        #     class_acc[cat, 1] += 1

        correct_points = pred_choice.eq(target.long().data).cpu().sum().item()
        total_correct_points += correct_points
        total_points += target.numel()

        target_np = target.cpu().numpy()
        pred_choice_np = pred_choice.cpu().numpy()
        # mean_correct.append(correct.item() / float(points.size()[0]))

        for class_id in range(num_class):
            # True positives:
            intersection = np.sum((pred_choice_np == class_id) & (target_np == class_id))
            # union:
            union = np.sum((pred_choice_np == class_id) | (target_np == class_id))

            total_intersection[class_id] += intersection
            total_union[class_id] += union
    
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

    # classification metrics
    # class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    # class_acc = np.mean(class_acc[:, 2])
    # instance_acc = np.mean(mean_correct)

    # return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    writer_dir = exp_dir.joinpath('SummaryWriter/')
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(writer_dir)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
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
    
    train_dataset = CustomSceneDataLoader(root=data_root_dir, args=args, split='train', process_data=args.process_data)
    test_dataset = CustomSceneDataLoader(root=data_root_dir, args=args, split='test', process_data=args.process_data)

    # train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=0, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/utils.py', str(exp_dir))
    shutil.copy('./train.py', str(exp_dir))

    shutil.copy('models/SCNet.py', str(exp_dir))
    shutil.copy('models/z_order.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_overall_accuracy = 0.0
    best_mean_iou = 0.0
    # best_instance_acc = 0.0
    # best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()
        total_loss = 0
        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                               smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            # flops, params = profile(classifier, (points,))
            # flops, params = clever_format([flops, params], "%.3f")
            # print(flops, params)
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            total_loss += loss.item()

            pred_choice = pred.data.max(1)[1]

            # correct = pred_choice.eq(target.long().data).cpu().sum()
            # mean_correct.append(correct.item() / float(points.size()[0]))

            loss.backward()
            optimizer.step()
            global_step += 1

        # train_instance_acc = np.mean(mean_correct)
        # log_string('Train Instance Accuracy: %f' % train_instance_acc)
        log_string('Total loss: %f' % total_loss)
        # writer.add_scalar('Accuracy/Train Instance Accuracy', train_instance_acc, epoch + 1)
        writer.add_scalar('Total loss', total_loss, epoch + 1)
        with torch.no_grad():
            # instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
            overall_accuracy, mean_iou, per_class_iou = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (mean_iou >= best_mean_iou):
                best_mean_iou = mean_iou
                best_overall_accuracy = overall_accuracy
                best_epoch = epoch + 1

            # if (class_acc >= best_class_acc):
            #     best_class_acc = class_acc
            log_string('Test Overall Accuracy: %f, Mean IoU: %f' % (overall_accuracy, mean_iou))
            writer.add_scalar('Accuracy/Test Overall Accuracy', overall_accuracy, epoch + 1)
            writer.add_scalar('Accuracy/Test Mean IoU', mean_iou, epoch + 1)
            log_string('Best Overall Accuracy: %f, Best Mean IoU: %f' % (best_overall_accuracy, best_mean_iou))

            if (mean_iou >= best_mean_iou):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'overall_accuracy': overall_accuracy,
                    'mean_iou': mean_iou,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
