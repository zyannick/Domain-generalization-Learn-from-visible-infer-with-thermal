import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

import sys
import random
from tqdm import tqdm
import setproctitle, colorama
import torch.nn.functional as F

import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import PIL
import pandas
import videotransforms

from flownet2.infer import call_inference, inference

import models as models
from train_loop import TrainLoop
import utils
from cmd_data_helpers import Target_Datasets, Source_Datasets
import os
import numpy as np
import torch
from tqdm import tqdm
from utils import LabelSmoothingLoss, get_values_from_batch, system_info
from torch.autograd import Variable
from glob import glob
import json

from args_parser import g2dm_parser

parser = g2dm_parser()

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
args.logging = True if not args.no_logging else False

assert args.alpha >= 0. and args.alpha <= 1.

print('Source domains: {}, {}, {}'.format(args.source1, args.source2, args.source3))
print('Orignal mode :{}'.format(args.original_mode))
print('Middle mode :{}'.format(args.middle_mode))
print('Target domain:', args.target)
print('Cuda Mode: {}'.format(args.cuda))
print('Batch size: {}'.format(args.batch_size))
print('LR task: {}'.format(args.lr_task))
print('LR domain: {}'.format(args.lr_domain))
print('L2: {}'.format(args.l2))
print('Alpha: {}'.format(args.alpha))
print('Momentum task: {}'.format(args.momentum_task))
print('Momentum domain: {}'.format(args.momentum_domain))
print('Nadir slack: {}'.format(args.nadir_slack))
print('RP size: {}'.format(args.rp_size))
print('Patience: {}'.format(args.patience))
print('Smoothing: {}'.format(args.smoothing))
print('Warmup its: {}'.format(args.warmup_its))
print('LR factor: {}'.format(args.factor))
print('Ablation: {}'.format(args.ablation))
print('Train mode: {}'.format(args.train_mode))
print('Train model: {}'.format(args.train_model))
print('Seed: {}'.format(args.seed))


def get_preds(logits):
    class_output = F.softmax(logits, dim=1)
    pred_task = class_output.data.max(1, keepdim=True)[1]
    return pred_task


def get_max_epoch(checkpoint_path):
    files = glob(os.path.join(checkpoint_path, ))

    return


class InferLoop(object):

    def __init__(self, models_dict, optimizer_task, infer_loader, nadir_slack,
                 alpha, patience, factor, label_smoothing, warmup_its, lr_threshold, args, batch_size=1, verbose=-1,
                 cp_name=None,
                 save_cp=True, checkpoint_path=None, checkpoint_epoch=None, cuda=True, logging=False, ablation='no',
                 train_mode='hv'):

        assert (checkpoint_path is not None)
        self.checkpoint_path = checkpoint_path

        self.save_epoch_fmt_task = os.path.join(self.checkpoint_path, 'task' + cp_name) if cp_name else os.path.join(
            self.checkpoint_path, 'task_checkpoint_{}ep.pt')
        self.save_epoch_fmt_domain = os.path.join(self.checkpoint_path,
                                                  'Domain_{}' + cp_name) if cp_name else os.path.join(
            self.checkpoint_path, 'Domain_{}.pt')

        self.cuda_mode = cuda
        self.batch_size = batch_size
        self.feature_extractor = models_dict['feature_extractor']
        # self.flow_net = None

        self.task_classifier = models_dict['task_classifier']
        self.domain_discriminator_list = models_dict['domain_discriminator_list']
        self.optimizer_task = optimizer_task
        self.test_source_loader = infer_loader
        self.target_loader = target_loader
        self.history = {'loss_task': [], 'hypervolume': [], 'loss_domain': [], 'accuracy_source': [],
                        'accuracy_target': []}
        self.cur_epoch = 0
        self.total_iter = 0
        self.nadir_slack = nadir_slack
        self.alpha = alpha
        self.ablation = ablation
        self.train_mode = train_mode
        self.device = next(self.feature_extractor.parameters()).device

        self.verbose = verbose
        self.save_cp = save_cp

        if checkpoint_epoch is not None:
            self.load_checkpoint(checkpoint_epoch)

        logging = False
        self.logging = logging
        if self.logging:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()

        if label_smoothing > 0.0:
            self.ce_criterion = LabelSmoothingLoss(label_smoothing, lbl_set_size=7)
        else:
            # self.ce_criterion = torch.nn.CrossEntropyLoss()	#torch.nn.NLLLoss()#
            self.ce_criterion = F.binary_cross_entropy_with_logits

        # loss_domain_discriminator = F.binary_cross_entropy_with_logits(y_predict, curr_y_domain)
        weight = torch.tensor([2.0 / 3.0, 1.0 / 3.0]).to(self.device)
        # d_cr=torch.nn.CrossEntropyLoss(weight=weight)
        self.d_cr = torch.nn.NLLLoss(weight=weight)

    # self.d_cr=  F.binary_cross_entropy_with_logits()
    #### Edit####
    def adjust_learning_rate(self, optimizer, epoch=1, every_n=700, In_lr=0.01):
        """Sets the learning rate to the initial LR decayed by 10 every n epoch epochs"""
        every_n_epoch = every_n  # n_epoch/n_step
        lr = In_lr * (0.1 ** (epoch // every_n_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def load_checkpoint(self, epoch):
        ckpt = self.save_epoch_fmt_task.format(epoch)

        if os.path.isfile(ckpt):

            ckpt = torch.load(ckpt)
            # Load model state
            self.feature_extractor.load_state_dict(ckpt['feature_extractor_state'])
            self.task_classifier.load_state_dict(ckpt['task_classifier_state'])
            # self.domain_classifier.load_state_dict(ckpt['domain_classifier_state'])
            # Load optimizer state
            self.optimizer_task.load_state_dict(ckpt['optimizer_task_state'])
            # Load scheduler state
            # self.scheduler_task.load_state_dict(ckpt['scheduler_task_state'])
            # Load history
            self.history = ckpt['history']
            self.cur_epoch = ckpt['cur_epoch']

            for i, disc in enumerate(self.domain_discriminator_list):
                ckpt = torch.load(self.save_epoch_fmt_domain.format(i + 1))
                disc.load_state_dict(ckpt['model_state'])
                disc.optimizer.load_state_dict(ckpt['optimizer_disc_state'])
                # self.scheduler_disc_list[i].load_state_dict(ckpt['scheduler_disc_state'])
        else:
            print('No checkpoint found at: {}'.format(ckpt))
            raise ValueError('----------Unable to load model for inference----------\n\n\n')

    def print_grad_norms(self, model):
        norm = 0.0
        for params in list(filter(lambda p: p.grad is not None, model.parameters())):
            norm += params.grad.norm(2).item()
        print('Sum of grads norms: {}'.format(norm))

    def update_nadir_point(self, losses_list):
        self.nadir = float(np.max(losses_list) * self.nadir_slack + 1e-8)


def infer(dataloader, feature_extractor, task_classifier, disc_list, device, source_target, epoch=0, tb_writer=None,
          Loss=F.binary_cross_entropy_with_logits):
    feature_extractor = feature_extractor.eval()
    task_classifier = task_classifier.eval()
    for disc in disc_list:
        disc = disc.eval()

    ground_truth_values = []
    predict_values = []

    with torch.no_grad():

        feature_extractor = feature_extractor.to(device)
        task_classifier = task_classifier.to(device)

        for disc in disc_list:
            disc = disc.to(device)

        target_iter = tqdm(enumerate(dataloader))

        n_total = 0
        n_correct = 0
        predictions_domain = []
        labels_domain = []

        for t, batch in target_iter:

            # if source_target == 'source':
            #     inputs, labels, y_domain = get_values_from_batch(batch)
            #     y_domain.to(device)
            # else:
            #     inputs, labels, _ = batch

            inputs, labels, y_domain = get_values_from_batch(batch)

            taille = inputs.size(2)

            inputs = inputs.to(device)
            labels = labels.to(device)

            features = feature_extractor.forward(inputs)

            # Task
            task_out = task_classifier.forward(features)
            task_out = F.upsample(task_out, taille, mode='linear')

            pred_task = get_preds(task_out)
            target_task = get_preds(labels)
            n_correct += pred_task.eq(target_task.data.view_as(pred_task)).cpu().sum() / target_task.size(2)
            # class_output = F.softmax(task_out, dim=1)
            # pred_task = class_output.data.max(1, keepdim=True)[1]
            # n_correct += pred_task.eq(y.data.view_as(pred_task)).cpu().sum()

            pred_labels = task_out.cpu().numpy()
            true_labels = target_task.cpu().numpy()

            pred_labels = np.amax(pred_labels, axis=2)
            pred_labels = np.argmax(pred_labels, axis=1)
            pred_labels = pred_labels[0]

            #print(true_labels.shape)
            true_labels = np.amax(true_labels, axis=2)
            true_labels = true_labels[0]

            # true_labels = np.argmax(true_labels, axis=1)

            ground_truth_values.append(true_labels)
            predict_values.append(pred_labels)

            task_loss = Loss(task_out, labels)
            n_total += inputs.size(0)

            if source_target == 'source':
                # Domain classification
                for i, disc in enumerate(disc_list):
                    pred_domain = disc.forward(features).squeeze()
                    curr_y_domain = torch.where(y_domain == i, torch.ones(y_domain.size(0)),
                                                torch.zeros(y_domain.size(0))).float().to(device)
                    try:
                        predictions_domain[i] = torch.cat(predictions_domain[i], pred_domain)
                        labels_domain[i] = torch.cat(labels_domain[i], curr_y_domain)
                    except:
                        predictions_domain.append(pred_domain)
                        labels_domain.append(curr_y_domain)
            try:
                predictions_task = torch.cat((predictions_task, pred_task), 0)
            except:
                predictions_task = pred_task

        acc = n_correct.item() * 1.0 / n_total

        if tb_writer is not None:
            predictions_task_numpy = predictions_task.cpu().numpy()
            tb_writer.add_histogram('Test/' + source_target, predictions_task_numpy, epoch)
            tb_writer.add_scalar('Test/' + source_target + '_accuracy', acc, epoch)

            if source_target == 'source':
                for i, disc in enumerate(disc_list):
                    predictions_domain_numpy = predictions_domain[i].cpu().numpy()
                    labels_domain_numpy = labels_domain[i].cpu().numpy()
                # tb_writer.add_histogram('Test/source-D{}-pred'.format(i), predictions_domain[i], epoch)
                # tb_writer.add_pr_curve('Test/ROC-D{}'.format(i), labels = labels_domain_numpy, predictions = predictions_domain_numpy, global_step = epoch)
        return acc, np.asarray(ground_truth_values), np.asarray(predict_values)





acc_runs = []
acc_blind = []
seeds = [1, 10, 100]


def get_model_name():

    if os.path.exists('dict_params.json'):
        with open('dict_params.json') as json_file:
            dict_params = json.load(json_file)

        args.ablation = dict_params['ablation']

    if args.ablation == 'model1' or args.ablation == 'baseline':
        args.data_aug = False

    train_mode =  args.ablation

    return train_mode

for run in range(args.n_runs):
    print('Run {}'.format(run))

    args.infer_target = 'cmd_fall'

    
    train_mode = get_model_name()

    if not os.path.exists('infer_results'):
        os.makedirs('infer_results')

    # Setting seed
    if args.seed is None:
        random.seed(seeds[run])
        torch.manual_seed(seeds[run])
        if args.cuda:
            torch.cuda.manual_seed(seeds[run])
        checkpoint_path = os.path.join(args.checkpoint_path, args.target + '_' + train_mode + '_seed' + str(seeds[run]))
    else:
        seeds[run] = args.seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        checkpoint_path = os.path.join(args.checkpoint_path, args.target + '_' + train_mode + '_seed' + str(args.seed))

    system_info()

    if not os.path.exists(checkpoint_path):
        print(checkpoint_path)
        raise ValueError('----------Unable to load model----------\n\n\n')
    else:
        list_checkpoints = glob(os.path.join(checkpoint_path, 'task_checkpoint_*'))
        max_checkpoint = -1
        for tas in list_checkpoints:
            ck = int((tas.split('task_checkpoint_')[-1]).split('ep.pt')[0])
            max_checkpoint = max(ck, max_checkpoint)
        args.checkpoint_epoch = max_checkpoint

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    nb_workers = 0

    print(args.batch_size)

    args.infer_target = 'cmd_fall'

    setproctitle.setproctitle('domain_generalization inference')

    flownet, args_flownet, _ = call_inference()

    if flownet is None and args.middle_mode == 'flow':
        raise ValueError('----------Unable to load the flownet model----------\n\n\n')

    target_dataset = Target_Datasets(split='testing', transforms=test_transforms, args=args, flownet=flownet,
                                     args_flow_net=args_flownet)

    val_dataset = Source_Datasets(split='testing', transforms=test_transforms,
                                  args=args, flownet=flownet, args_flow_net=args_flownet)

    if args.infer_target == 'cmd_fall':
        target = val_dataset
    else:
        target = target_dataset

    target_loader = torch.utils.data.DataLoader(target, batch_size=args.batch_size, shuffle=True,
                                                num_workers=nb_workers, pin_memory=True)

    task_classifier = models.task_classifier()
    domain_discriminator_list = []
    for i in range(3):
        if args.rp_size == 4096:
            disc = models.domain_discriminator_ablation_RP(optim.SGD, args.lr_domain, args.momentum_domain,
                                                           args.l2).train()
        else:
            disc = models.domain_discriminator(args.rp_size, optim.SGD, args.lr_domain, args.momentum_domain,
                                               args.l2).train()
        domain_discriminator_list.append(disc)

    feature_extractor = models.get_pretrained_model(args)

    optimizer_task = optim.SGD(list(feature_extractor.parameters()) + list(task_classifier.parameters()),
                               lr=args.lr_task, momentum=args.momentum_task, weight_decay=args.l2)
    models_dict = {}

    models_dict['feature_extractor'] = feature_extractor
    models_dict['task_classifier'] = task_classifier
    models_dict['domain_discriminator_list'] = domain_discriminator_list
    if args.cuda:
        for key in models_dict.keys():
            if key != 'domain_discriminator_list':
                models_dict[key] = models_dict[key].cuda()
            else:
                for k, disc in enumerate(models_dict[key]):
                    models_dict[key][k] = disc.cuda()
        torch.backends.cudnn.benchmark = True

    inferloop = InferLoop(models_dict, optimizer_task, target_loader, args.nadir_slack,
                          args.alpha, args.patience, args.factor, args.smoothing, args.warmup_its, args.lr_threshold,
                          args, batch_size=args.batch_size,
                          checkpoint_path=checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda,
                          ablation=args.ablation, logging=args.logging, train_mode=args.train_mode)

    history = {'loss_task': [], 'hypervolume': [], 'loss_domain': [], 'accuracy_source': [],
               'accuracy_target': []}

    acc, ground_truth_values, predict_values = infer(inferloop.target_loader, inferloop.feature_extractor,
                                                     inferloop.task_classifier,
                                                     inferloop.domain_discriminator_list,
                                                     inferloop.device, source_target='target',
                                                     epoch=inferloop.cur_epoch,
                                                     tb_writer=inferloop.writer if inferloop.logging else None)

    # print(predict_values)
    # print(ground_truth_values)

    np.savetxt(os.path.join('infer_results', 'predict_' + args.target + '_' + train_mode + '_' + str(run)) + '.csv',
               np.squeeze(predict_values) )
    np.savetxt(
        os.path.join('infer_results', 'ground_truth_' + args.target + '_' + train_mode + '_' + str(run)) + '.csv',
        np.squeeze(ground_truth_values))


