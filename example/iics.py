from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import shutil
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

import gc 

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from reid.loss import TripletLoss, LabelSmoothing

from torchvision.transforms import  InterpolationMode

from reid import datasets
from reid import models
from reid.trainers import IntraCameraTrainer
from reid.trainers import InterCameraTrainer
from reid.evaluators_cos import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.cluster_utils import get_intra_cam_cluster_result, get_inter_cam_cluster_result
from reid.utils.data.sampler import RandomIdentitySampler
torch.multiprocessing.set_sharing_strategy('file_system')


def get_data(
    name,
    split_id,
    data_dir,
    height,
    width,
    batch_size,
    workers,
):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval
    num_classes = dataset.num_trainval_ids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(Preprocessor(train_set,
                                           root=dataset.images_dir,
                                           transform=train_transformer),
                              batch_size=batch_size,
                              num_workers=workers,
                              shuffle=False,
                              pin_memory=False,
                              drop_last=False)

    val_loader = DataLoader(Preprocessor(dataset.val,
                                         root=dataset.images_dir,
                                         transform=test_transformer),
                            batch_size=batch_size,
                            num_workers=workers,
                            shuffle=False,
                            pin_memory=False)

    test_loader = DataLoader(Preprocessor(
        list(set(dataset.query) | set(dataset.gallery)),
        root=dataset.images_dir,
        transform=test_transformer),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=False)

    return dataset, num_classes, train_loader, val_loader, test_loader


def make_params(model, lr, weight_decay):
    params = []
    for key, value in model.model.named_parameters():
        if not value.requires_grad:
            continue

        params += [{
            "params": [value],
            "lr": lr * 0.1,
            "weight_decay": weight_decay
        }]
    for key, value in model.classifier.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    return params


def get_mix_rate(mix_rate, epoch, num_epoch, power=0.6):
    return mix_rate * (1 - epoch / num_epoch) ** power


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print(args)
    shutil.copy(sys.argv[0], osp.join(args.logs_dir,
                                      osp.basename(sys.argv[0])))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (256, 128)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size * 8, args.workers,
                 )

    # Create model
    model = models.create("ft_net_inter",
                          num_classes=num_classes, stride=args.stride)

    # Load from checkpoint
    start_epoch = 0
    best_top1 = 0
    top1 = 0
    is_best = False
    if args.checkpoint is not None:
        if args.evaluate:
            checkpoint = load_checkpoint(args.checkpoint)
            param_dict = model.state_dict()
            for k, v in checkpoint['state_dict'].items():
                if 'model' in k:
                    param_dict[k] = v
            model.load_state_dict(param_dict)
        else:
            model.model.load_param(args.checkpoint)
            # checkpoint_continue = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
            # new_state_dict = {}
            # for k in checkpoint_continue['state_dict'].keys():
            #     if 'model' in k:
            #         new_state_dict[k] = checkpoint_continue['state_dict'][k]
            # model.load_state_dict(new_state_dict, strict=False)

    model = model.cuda()

    # Distance metric
    metric = None

    # Evaluator
    evaluator = Evaluator(model, use_cpu=args.use_cpu)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)
        return

    train_transformer = [
        T.Resize((args.height, args.width), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=0.5),
    ]
    train_transformer = T.Compose(train_transformer)
    losses_entropy = []
    precisions_entropy = []
    losses_triple = []
    precisions_triple = []
    
    for cluster_epoch in range(args.cluster_epochs):
        # -------------------------Stage 1 intra camera training--------------------------
        # Cluster and generate new dataset and model
        
        # if cluster_epoch != 0:
        #   model = model_end
        #   del model_end
        #   gc.collect()
        #   gc.collect() 
        
        cluster_result = get_intra_cam_cluster_result(model, train_loader,
                                                      args.class_number_stage1,
                                                      args.linkage)
        cluster_datasets = [
            datasets.create("cluster", osp.join(args.data_dir, args.dataset),
                            cluster_result[cam_id], cam_id)
            for cam_id in cluster_result.keys()
        ]

        cluster_dataloaders = [
            DataLoader(Preprocessor(dataset.train_set,
                                    root=dataset.images_dir,
                                    transform=train_transformer),
                       batch_size=args.batch_size,
                       num_workers=args.workers,
                       shuffle=True,
                       pin_memory=False,
                       drop_last=True) for dataset in cluster_datasets
        ]
        param_dict = model.model.state_dict()
        model = models.create("ft_net_intra",
                              num_classes=[
                                  args.class_number_stage1
                                  for cam_id in cluster_result.keys()
                              ],
                              stride=args.stride)

        model_param_dict = model.model.state_dict()
        for k, v in model_param_dict.items():
            if k in param_dict.keys():
                model_param_dict[k] = param_dict[k]
        model.model.load_state_dict(model_param_dict)

        model = model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()

        # Optimizer
        param_groups = make_params(model, args.lr, args.weight_decay)
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
        # Trainer
        trainer = IntraCameraTrainer(
            model, criterion, warm_up_epoch=args.warm_up)
        print("start training")
        # Start training
        for epoch in range(0, args.epochs_stage1):
            trainer.train(cluster_epoch,
                          epoch,
                          cluster_dataloaders,
                          optimizer,
                          print_freq=args.print_freq,
                          )
        torch.cuda.empty_cache()
        # -------------------------------------------Stage 2 inter camera training-----------------------------------
        mix_rate = get_mix_rate(
            args.mix_rate, cluster_epoch, args.cluster_epochs, power=args.decay_factor)

        cluster_result = get_inter_cam_cluster_result(
            model,
            train_loader,
            args.class_number_stage2,
            args.linkage,
            mix_rate,
            use_cpu=args.use_cpu)

        cluster_dataset = datasets.create(
            "cluster", osp.join(args.data_dir, args.dataset), cluster_result,
            0)

        cluster_dataloaders = DataLoader(
            Preprocessor(cluster_dataset.train_set,
                         root=cluster_dataset.images_dir,
                         transform=train_transformer),
            batch_size=args.batch_size_stage2,
            num_workers=args.workers,
            sampler=RandomIdentitySampler(cluster_dataset.train_set,
                                          args.batch_size_stage2,
                                          args.instances),
            pin_memory=False,
            drop_last=True)

        param_dict = model.model.state_dict()
        model = models.create("ft_net_inter",
                              num_classes=args.class_number_stage2,
                              stride=args.stride)
        model.model.load_state_dict(param_dict)

        model = model.cuda()
        # Criterion
        criterion_entropy = nn.CrossEntropyLoss().cuda()  #交叉熵损失
        criterion_triple = TripletLoss(margin=args.margin).cuda()   #

        # Optimizer
        param_groups = make_params(model,
                                   args.lr * args.batch_size_stage2 / 32,
                                   args.weight_decay)

        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
        # Trainer
        trainer = InterCameraTrainer(model,
                                     criterion_entropy,
                                     criterion_triple,
                                     warm_up_epoch=args.warm_up,
                                     )

        print("start training")
        # Start training
        a = b = c = d = 0
        for epoch in range(0, args.epochs_stage2):
            a, b, c, d = trainer.train(cluster_epoch,
                          epoch,
                          cluster_dataloaders,
                          optimizer,
                          print_freq=args.print_freq)
        #训练完后将最后的损失或者准确率加入列表
        # losses_entropy.append(a)
        # precisions_entropy.append(b)
        # losses_triple.append(c)
        # precisions_triple.append(d)
        
        
        
        if (cluster_epoch + 1) % 5 == 0:

            evaluator = Evaluator(model, use_cpu=args.use_cpu)
            top1, mAP = evaluator.evaluate(
                test_loader, dataset.query, dataset.gallery, metric, return_mAP=True)

            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)

            save_checkpoint(
                {
                    'state_dict': model.state_dict(),
                    'epoch': cluster_epoch + 1,
                    'best_top1': best_top1,
                    'cluster_epoch': cluster_epoch + 1,
                },
                is_best,
                fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        if cluster_epoch == (args.cluster_epochs - 1):
            save_checkpoint(
                {
                    'state_dict': model.state_dict(),
                    'epoch': cluster_epoch + 1,
                    'best_top1': best_top1,
                    'cluster_epoch': cluster_epoch + 1,
                },
                False,
                fpath=osp.join(args.logs_dir, 'latest.pth.tar'))

        print('\n * cluster_epoch: {:3d} top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(cluster_epoch, top1, best_top1, ' *' if is_best else ''))




    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    best_rank1, mAP = evaluator.evaluate(
        test_loader, dataset.query, dataset.gallery, metric, return_mAP=True)


    #draw the loss and precision
    # dev_cpu = torch.device("cpu")
    # print(losses_entropy)
    # print(precisions_entropy)
    # print(losses_triple)
    # print(precisions_triple)
    #
    # len_p_e = len(precisions_entropy)
    # for i in range(len_p_e):
    #   if isinstance(precisions_entropy[i],int):
    #     precisions_entropy[i] = np.array(precisions_entropy[i])
    #   else:
    #     precisions_entropy[i] = precisions_entropy[i].cpu().numpy()
    # len_p_t = len(precisions_triple)
    # for i in range(len_p_t):
    #   if isinstance(precisions_triple[i],int):
    #     precisions_triple[i] = np.array(precisions_triple[i])
    #   else:
    #     precisions_triple[i] = precisions_triple[i].cpu().numpy()
    #
    # x1 = range(0,40)
    # x2 = range(0,40)
    # x3 = range(0,40)
    # x4 = range(0,40)
    # y1 = losses_entropy
    # print(y1)
    # y2 = precisions_entropy
    #
    # print(y2)
    # y3 = losses_triple
    # print(y3)
    # y4 = precisions_triple
    #
    # print(y4)
    #
    #
    # plt.subplot(2,2,1)
    # plt.plot(x1,y1,'o-')
    # plt.title('losses_entropy vs epoches')
    # plt.ylabel('losses_entropy')
    # plt.savefig('./loss_entropy.png')
    #
    # plt.clf()
    # plt.subplot(2, 2, 1)
    # plt.plot(x2, y2, 'o-')
    # plt.title('precisions_entropy vs epoches')
    # plt.ylabel('precisions_entropy')
    # plt.savefig('./precisions_entropy.png')
    #
    # plt.clf()
    # plt.subplot(2, 2, 1)
    # plt.plot(x3, y3, 'o-')
    # plt.title('losses_triple vs epoches')
    # plt.ylabel('losses_triple')
    # plt.savefig('./losses_triple.png')
    #
    # plt.clf()
    # plt.subplot(2, 2, 1)
    # plt.plot(x4, y4, 'o-')
    # plt.title('precisions_triple vs epoches')
    # plt.ylabel('precisions_triple')
    # plt.savefig('./precisions_triple.png')
    # # plt.savefig('./loss.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('--checkpoint', type=str, metavar='PATH')
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        default='market1501',
                        choices=datasets.names())
    parser.add_argument('--class_number_stage1', type=int, default=600)
    parser.add_argument('--class_number_stage2', type=int, default=800)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--mix_rate', type=float,
                        default=0.02, help="mu in Eq (5)")
    parser.add_argument('--decay_factor', type=float, default=0.6)

    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-b2', '--batch-size-stage2', type=int,
                        default=64)
    parser.add_argument('--instances', default=4)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height',
                        type=int,
                        help="input height, default: 256 for resnet*, "
                        "144 for inception")
    parser.add_argument('--width',
                        type=int,
                        help="input width, default: 128 for resnet*, "
                        "56 for inception")
    # optimizer
    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help="learning rate of new parameters, for pretrained "
                        "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--evaluate',
                        action='store_true',
                        help="evaluation only")
    parser.add_argument(
        '--use_cpu',
        action='store_true',
        help='use cpu to calculate dist to prevent from GPU OOM')
    parser.add_argument('--epochs_stage1', type=int, default=2)
    parser.add_argument('--epochs_stage2', type=int, default=2)
    parser.add_argument('--cluster_epochs', type=int, default=40)
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--start_save',
                        type=int,
                        default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--linkage', type=str, default="average")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--multi_task_weight', type=float, default=1.)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir',
                        type=str,
                        metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir',
                        type=str,
                        metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
