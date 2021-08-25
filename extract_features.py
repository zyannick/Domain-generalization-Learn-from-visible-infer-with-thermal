import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1,3'

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
from cmd_data_helpers import Source_Dataset
import os
import numpy as np
import torch
from tqdm import tqdm
from utils import LabelSmoothingLoss, get_values_from_batch
from torch.autograd import Variable
from glob import glob
import pandas as pd
import h5py

parser = argparse.ArgumentParser(description='RP for domain generalization')
parser.add_argument('--original_mode', type=str, default='rgb', help='raw or flow')
parser.add_argument('--middle_mode', type=str, default='raw', help='rgb or flow')
parser.add_argument('--save_model', default='checkpoint', type=str)
parser.add_argument('--edge_type', default='no_edge', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_classes', default=8, type=int)
parser.add_argument('--nb_frames', default=16, type=int)
parser.add_argument('--blur_kernel', type=int, default=0)
parser.add_argument('--operator_kernel', type=int, default=3)
parser.add_argument('--number_of_domain', type=int, default=3)
parser.add_argument('--domains_list', type=list, default=['raw_rgb', 'sobel_0_3', 'laplace_0_3'])
parser.add_argument('--model_name', default='i3d', type=str)

parser.add_argument('--root', default='../five_fps_cme_sep/', type=str)
parser.add_argument('--dataset_name', default='cme_sep', type=str)
parser.add_argument('--split_file', default='./maj_cmd_fall.json', type=str)

parser.add_argument('--target_root', default='./baga_balanced/videos/', type=str)
parser.add_argument('--target_split_file', default='./baga_balanced/baga.json', type=str)

parser.add_argument('--data_aug', type=bool, default=True)
parser.add_argument('--flow_net', type=bool, default=True)
parser.add_argument('--continue_training', type=bool, default=True)
parser.add_argument('--affine_transform', type=bool, default=True)
parser.add_argument('--is_extended', type=bool, default=False)
parser.add_argument('--is_inference', type=bool, default=False)
parser.add_argument('--label_smoothing', type=bool, default=False)
parser.add_argument('--distance_transform', type=int, default=0)
parser.add_argument('--type_img', type=str, default='visible')
parser.add_argument('--gpus', type=str, default='0,1,2,3')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 50)')
parser.add_argument('--lr-task', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--lr-domain', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--lr-threshold', type=float, default=1e-4, metavar='LRthrs', help='learning rate (default: 1e-4)')
parser.add_argument('--momentum-task', type=float, default=0.9, metavar='m', help='momentum (default: 0.9)')
parser.add_argument('--momentum-domain', type=float, default=0.9, metavar='m', help='momentum (default: 0.9)')
parser.add_argument('--l2', type=float, default=1e-5, metavar='L2', help='Weight decay coefficient (default: 0.00001')
parser.add_argument('--factor', type=float, default=0.1, metavar='f', help='LR decrease factor (default: 0.1')
parser.add_argument('--checkpoint-epoch', type=int, default=24, metavar='N',
                    help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default='checkpoint', metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='../data/pacs/prepared_data/', metavar='Path', help='Data path')

parser.add_argument('--source1', type=str, default='sobel_0_3', metavar='Path', help='Path to source1 file')
parser.add_argument('--source2', type=str, default='sobel_3_5', metavar='Path', help='Path to source2 file')
parser.add_argument('--source3', type=str, default='laplace_0_3', metavar='Path', help='Path to source3 file')

parser.add_argument('--target', type=str, default='tir_sobel', metavar='Path', help='Path to target data')
parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: None)')
parser.add_argument('--nadir-slack', type=float, default=1.5, metavar='nadir',
                    help='factor for nadir-point update. Only used in hyper mode (default: 1.5)')
parser.add_argument('--alpha', type=float, default=0.8, metavar='alpha',
                    help='balance losses to train encoder. Should be within [0,1]')
parser.add_argument('--rp-size', type=int, default=3000, metavar='rp',
                    help='Random projection size. Should be smaller than 4096')
parser.add_argument('--patience', type=int, default=20, metavar='N',
                    help='number of epochs to wait before reducing lr (default: 20)')
parser.add_argument('--smoothing', type=float, default=0.0, metavar='l', help='Label smoothing (default: 0.2)')
parser.add_argument('--warmup-its', type=float, default=500, metavar='w', help='LR warm-up iterations (default: 500)')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--save-every', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging training status. Default is 5')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--no-logging', action='store_true', default=False, help='Deactivates logging')
parser.add_argument('--ablation', choices=['all', 'RP', 'no'], default='no',
                    help='Ablation study (removing only RPs (option: RP), RPs+domain classifier (option: all), (default: no))')
parser.add_argument('--train-mode', choices=['hv', 'avg'], default='hv',
                    help='Train mode (options: hv, avg), (default: hv))')
parser.add_argument('--train-model', choices=['alexnet', 'resnet18', 'i3d', 'r2p1d'], default='i3d',
                    help='Train model (options: alexnet, resnet18, i3d, r2p1d), (default: i3d))')

parser.add_argument('--n-runs', type=int, default=1, metavar='n', help='Number of repetitions (default: 3)')

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

acc_runs = []
acc_blind = []
seeds = [1, 10, 100]

train_mode = args.original_mode + '_' + args.middle_mode

nb_domains = [1, 2, 3, 4]

class ExtractFeaturesParams(object):

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
        self.target_loader = infer_loader
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


def extracted_features(dataloader, feature_extractor, device, dom):
    feature_extractor = feature_extractor.eval()

    with torch.no_grad():

        feature_extractor = feature_extractor.to(device)

        target_iter = tqdm(enumerate(dataloader))

        for t, batch in target_iter:
            savepath = os.path.join(os.getcwd(), train_mode + str(dom) +  '.h5')

            inputs, labels, domains = batch

            inputs = inputs.to(device)

            features = feature_extractor.forward(inputs)

            features = features.cpu().numpy()

            labels = labels.numpy()


            if t == 0:
                if os.path.exists(savepath):
                    print("\n%s have existed!\n" % (savepath))
                    return False
                else:
                    hf = h5py.File(savepath, 'w')
                    features_h5 = hf.create_dataset("features", (1, 1, 1024, 2, 7, 7),
                                                    maxshape=(None, 1, 1024, 2, 7, 7),
                                                    chunks=(1, 1, 1024, 2, 7, 7), dtype='float32')
                    labels_h5 = hf.create_dataset("labels", (1, 1, 8, 15),
                                                  maxshape=(None, 1, 8, 15),
                                                  chunks=(1, 1, 8, 15), dtype='float32')
                    domains_h5 = hf.create_dataset("domains", (1, 1),
                                                  maxshape=(None, 1),
                                                  chunks=(1, 1), dtype='float32')
            else:
                hf = h5py.File(savepath, 'a')
                features_h5 = hf["features"]
                labels_h5 = hf["labels"]
                domains_h5 = hf["domains"]

            features_h5.resize([t + 1, 1, 1024, 2, 7, 7])
            features_h5[t: t + 1] = features

            labels_h5.resize([t + 1, 1, 8, 15])
            labels_h5[t: t + 1] = labels

            domains_h5.resize([t + 1, 1])
            domains_h5[t: t + 1] = domains


def system_info():
    import torch.cuda as cuda
    import torchvision
    print(sys.version, "\n")
    print("PyTorch {}".format(torch.__version__), "\n")
    print("Torch-vision {}".format(torchvision.__version__), "\n")
    print("Available devices:")
    if cuda.is_available():
        for i in range(cuda.device_count()):
            print("{}: {}".format(i, cuda.get_device_name(i)))
    else:
        print("CPUs")




for dom in nb_domains:
    run = 0

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

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    nb_workers = 0

    print(args.batch_size)

    setproctitle.setproctitle('domain_generalization inference')

    flownet, args_flownet, _ = call_inference()

    if flownet is None and args.middle_mode == 'flow':
        raise ValueError('----------Unable to load the flownet model----------\n\n\n')

    if dom != 4:
        dataset = Source_Dataset(split='testing', split_file=args.split_file, root=args.root,
                                 transforms=test_transforms, args=args, flownet=flownet,
                                     args_flow_net=args_flownet, domain = dom)
    else:
        dataset = Source_Dataset(split='testing', split_file=args.target_split_file, root=args.target_root,
                                 transforms=test_transforms, args=args, flownet=flownet,
                                 args_flow_net=args_flownet, domain=dom)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
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

    inferloop = ExtractFeaturesParams(models_dict, optimizer_task, loader, args.nadir_slack,
                                      args.alpha, args.patience, args.factor, args.smoothing, args.warmup_its, args.lr_threshold,
                                      args, batch_size=args.batch_size,
                                      checkpoint_path=checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda,
                                      ablation=args.ablation, logging=args.logging, train_mode=args.train_mode)



    extracted_features(dataloader = inferloop.target_loader, feature_extractor=inferloop.feature_extractor,
                       device=inferloop.device,dom = dom)


