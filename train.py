import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from cmd_data_helpers import Source_Datasets, Target_Datasets
import utils
import torchvision.models as models_tv
from data_loader import Loader_source, Loader_validation, Loader_unif_sampling
from train_loop import TrainLoop
import models as models
import videotransforms
import pandas
import PIL
from torchvision import transforms
from torchvision import datasets
import torch.utils.data
import torch.optim as optim
import argparse
import sys
import random
from tqdm import tqdm
import setproctitle
import colorama
from utils import system_info


from args_parser import g2dm_parser

parser = g2dm_parser()

args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
args.logging = True if not args.no_logging else False

assert args.alpha >= 0. and args.alpha <= 1.

# print('Source domains: {}, {}, {}'.format(
#     args.source1, args.source2, args.source3))
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



print('here ------------------')
print(args.gpus)
from flownet2.infer import call_inference, inference


acc_runs = []
acc_blind = []
seeds = [1, 10, 100]

args.n_runs = 1

import json


def get_model_name():

    if os.path.exists('dict_params.json'):
        with open('dict_params.json') as json_file:
            dict_params = json.load(json_file)

        args.ablation = dict_params['ablation']

    if args.ablation == 'model1' or args.ablation == 'baseline':
        args.data_aug = False

    train_mode = args.original_mode + '_' + args.middle_mode

    # if args.batch_size > 1:
    #     train_mode = train_mode + '_' + str(args.batch_size)


    train_mode =  args.ablation

    return train_mode 


for run in range(args.n_runs):
    print('Run {}'.format(run))

    

    train_mode = get_model_name()

    print('data augmentation')
    print(args.data_aug)

    # Setting seed
    if args.seed is None:
        random.seed(seeds[run])
        torch.manual_seed(seeds[run])
        if args.cuda:
            torch.cuda.manual_seed(seeds[run])
        checkpoint_path = os.path.join(
            args.checkpoint_path, args.target + '_' + train_mode + '_seed' + str(seeds[run]))
    else:
        seeds[run] = args.seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        checkpoint_path = os.path.join(
            args.checkpoint_path, args.target + '_' + train_mode + '_seed' + str(args.seed))

    system_info()

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    nb_workers = 0

    print(args.batch_size)

    setproctitle.setproctitle(train_mode)

    flownet, args_flownet, _ = call_inference()

    if flownet is None and args.middle_mode == 'flow':
        raise ValueError(
            '----------Unable to load the flownet model----------\n\n\n')

    dataset = Source_Datasets(split='training', args=args,
                              transforms=train_transforms, flownet=flownet, args_flow_net=args_flownet)
    source_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=nb_workers,
                                                pin_memory=True)

    val_dataset = Source_Datasets(split='testing', transforms=test_transforms,
                                  args=args, flownet=flownet, args_flow_net=args_flownet)
    test_source_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=nb_workers, pin_memory=True)

    target_dataset = Target_Datasets(
        split='testing', transforms=test_transforms, args=args, flownet=flownet, args_flow_net=args_flownet)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
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

    trainer = TrainLoop(models_dict, optimizer_task, source_loader, test_source_loader, target_loader, args.nadir_slack,
                        args.alpha, args.patience, args.factor, args.smoothing, args.warmup_its, args.lr_threshold, args, batch_size=args.batch_size,
                        checkpoint_path=checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, cuda=args.cuda,
                        ablation=args.ablation, logging=args.logging, train_mode=args.train_mode)
    _, _ = trainer.train(n_epochs=args.epochs, save_every=1)



