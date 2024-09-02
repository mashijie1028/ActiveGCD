import os
import argparse
from copy import deepcopy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from utils_simgcd.general_utils_al import AverageMeter, init_experiment   # NOTE!!!
from utils_simgcd.cluster_and_log_utils_al import log_accs_from_preds
from config import exp_root
from model import DINOHead_feature, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups   # NOTE!!! DINOHead_feature not DINOHead

from query_strategies import strategy
from utils_al.handler import DataHandler, ImageNetDataHandler, CUBDataHandler, AircraftDataHandler, CarsDataHandler, Herbarium19DataHandler
from utils_al.al_data import AL_Data, AL_Data_ImageNet, AL_Data_CUB, AL_Data_Aircraft, AL_Data_Cars, AL_Data_Herb19
from utils_al.al_net import AL_Net
from utils_al.get_al_strategy import get_strategy


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Active Learning Training (fine-tuning)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aircraft, herbarium_19')

    # dataset labels NOTE!!!
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--num_old_classes', type=int, default=-1)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--logits_temp', default=0.1, type=float, help='temperature for logits')   # NOTE!!!

    # args for AL
    parser.add_argument('--strategy', type=str, default=None, help='Active Learning: strategy')
    parser.add_argument('--num_round', default=1, type=int, help='Active Learning: training rounds')
    parser.add_argument('--adaptive_round', default=2, type=int, help='For Adaptive NovelMarginSamplingAdaptive, transfer from NovelSamplingRandom to NovelSampling (Entropy) at this round')
    parser.add_argument('--num_query', default=100, type=int, help='Active Learning: number of query per round')
    parser.add_argument('--base_exp_root', type=str, default='dev_outputs_base')
    parser.add_argument('--base_ckpts_date', type=str, default=None, help='base (initial) checkpoints directories, e.g. 20231007-212237')
    parser.add_argument('--base_exp_id', default=None, type=str)
    parser.add_argument('--epochs', default=20, type=int, help='Active Learning training epochs per round')
    parser.add_argument('--al_batch_size', default=128, type=int, help='bathc size for AL labeled data')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--al_weight', type=float, default=1)
    parser.add_argument('--al_supcon_weight', type=float, default=1)
    parser.add_argument('--al_cls_weight', type=float, default=1)


    # ema
    parser.add_argument('--ema_decay', type=float, default=0.9)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--print_freq_al', default=5, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--exp_id', default=None, type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.exp_root = 'dev_outputs_al_ema'
    if args.num_round > 1:
        args.exp_root = args.exp_root + '_' + str(args.num_round) + 'rounds'
    args.exp_name = args.dataset_name + '_simgcd_al'
    args.base_exp_id = 'old' + str(args.num_labeled_classes) + '_' + 'ratio' + str(args.prop_train_labels)

    init_experiment(args, runner_name=['simgcd-al'], exp_id=args.exp_id)
    args.logger.info('number of old and novel classes: (%d)-(%d)' % (args.num_labeled_classes, args.num_unlabeled_classes))
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    backbone_ema = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    for m in backbone_ema.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone_ema.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True


    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # NOTE !!! ind_mapping dataset
    train_dataset_ind_mapping = deepcopy(datasets['train_labelled'])
    train_dataset_ind_mapping.transform = test_transform

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # ind_mapping dataset
    train_labeled_loader_ind_mapping = DataLoader(train_dataset_ind_mapping, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead_feature(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projector_ema = DINOHead_feature(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)
    model_ema = nn.Sequential(backbone_ema, projector_ema).to(device)

    # load checkpoints from the base model
    if args.base_ckpts_date is not None:
        if args.base_exp_id is not None:
            args.base_model_dir = os.path.join(args.base_exp_root, args.dataset_name, args.base_exp_id + '_' + args.base_ckpts_date, 'checkpoints', 'model.pt')
        else:
            args.base_model_dir = os.path.join(args.base_exp_root, args.dataset_name, args.base_ckpts_date, 'checkpoints', 'model.pt')
        args.logger.info(f'Loading weights from {args.base_model_dir}')
        ckpts = torch.load(args.base_model_dir)
        ckpts = ckpts['model']
        model.load_state_dict(ckpts)
        print('Successfully load checkpoints.')
        model_ema.load_state_dict(ckpts)
        print('Successfully load checkpoints (ema).')

    # detach ema params
    for param in model_ema.parameters():
        param.detach_()

    # ----------------------
    # Active Learning Settings
    # ----------------------
    al_source_dataset = deepcopy(unlabelled_train_examples_test)
    #al_dataset = AL_Data(X_train=al_source_dataset.data, Y_train=al_source_dataset.targets, handler=DataHandler)
    if args.dataset_name == 'cifar100' or args.dataset_name == 'cifar10':
        al_dataset = AL_Data(X_train=al_source_dataset.data, Y_train=torch.LongTensor(al_source_dataset.targets), handler=DataHandler)   # NOTE!!! LongTensor
    elif args.dataset_name == 'imagenet_100':
        al_dataset = AL_Data_ImageNet(Imgs=al_source_dataset.imgs, Samples=al_source_dataset.samples, Targets=al_source_dataset.targets,
                                      Uq_idxs=al_source_dataset.uq_idxs, target_transform=al_source_dataset.target_transform, handler=ImageNetDataHandler)   # NOTE!!! LongTensor
    elif args.dataset_name == 'cub':
        al_dataset = AL_Data_CUB(Data=al_source_dataset.data, Uq_idxs=al_source_dataset.uq_idxs, target_transform=al_source_dataset.target_transform, handler=CUBDataHandler)
    elif args.dataset_name == 'scars':
        al_dataset = AL_Data_Cars(Data=al_source_dataset.data, Target=al_source_dataset.target, Uq_idxs=al_source_dataset.uq_idxs, target_transform=al_source_dataset.target_transform, handler=CarsDataHandler)
    elif args.dataset_name == 'fgvc_aircraft':
        al_dataset = AL_Data_Aircraft(Samples=al_source_dataset.samples, Uq_idxs=al_source_dataset.uq_idxs, target_transform=al_source_dataset.target_transform, handler=AircraftDataHandler)
    elif args.dataset_name == 'herbarium_19':
        al_dataset = AL_Data_Herb19(Samples=al_source_dataset.samples, Targets=al_source_dataset.targets,
                                      Uq_idxs=al_source_dataset.uq_idxs, target_transform=al_source_dataset.target_transform, handler=Herbarium19DataHandler)
    else:
        al_dataset = AL_Data(X_train=al_source_dataset.data, Y_train=torch.LongTensor(al_source_dataset.targets), handler=DataHandler)   # NOTE!!! LongTensor
    al_net = AL_Net(net=model, net_ema=model_ema, args=args, device=device)
    al_strategy = get_strategy(args.strategy)(al_dataset, train_loader, test_loader_labelled, test_loader_unlabelled,
                                              None, train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args)

    # evaluate before AL
    args.logger.info('Evaluate before AL training...')
    all_acc_initial, old_acc_initial, new_acc_initial, ind_map_test_initial = al_strategy.test()

    # accuracy measure list
    all_acc_round_list = []
    old_acc_round_list = []
    new_acc_round_list = []
    # novelty measure list
    coverage_round_list = []
    ratio_round_list = []
    entropy_round_list = []


    # begin AL training of various rounds
    for rd in range(1, args.num_round + 1):
        args.logger.info('\n\nBegin Active Learning Round {}'.format(rd))

        # query
        query_idxs = al_strategy.query(args.num_query, rd)

        # measure query acc
        _, _, _, ind_map_test_ = al_strategy.al_net.test(al_strategy.original_test_loader, 'Test ACC')
        al_strategy.measure_acc(query_idxs, ind_map_test_)

        # update labels in unlabeled training data
        al_strategy.update(query_idxs)

        # metric the novelty of query
        args.logger.info('Evaluating the novelty metrics on AL selected data (Current Round)...')
        coverage_round, ratio_round, entropy_round, upper_bound_round = al_strategy.measure_novelty(query_idxs)
        args.logger.info('Evaluating the novelty metrics on AL selected data (Overall across All Rounds)...')
        coverage_overall, ratio_overall, entropy_overall, upper_bound_overall = al_strategy.measure_novelty_overall()

        # train and evaluate
        best_test_acc_all_round, best_test_acc_lab_round, best_test_acc_ubl_round = al_strategy.train(rd)

        # logs of current round
        all_acc_round_list.append(best_test_acc_all_round)
        old_acc_round_list.append(best_test_acc_lab_round)
        new_acc_round_list.append(best_test_acc_ubl_round)
        coverage_round_list.append(coverage_round)
        ratio_round_list.append(ratio_round)
        entropy_round_list.append(entropy_round)


        # state dict:
        save_round_name = 'state_at_the_end_of_round{}.pt'.format(rd)
        save_round_path = os.path.join(args.model_dir, save_round_name)
        labeled_idxs_, labeled_data_ = al_strategy.al_dataset.get_labeled_data(train_transform)
        save_round_dict = {
            'round': rd,
            'labeled_idxs': labeled_idxs_,
            'all_acc_round_list': all_acc_round_list,
            'old_acc_round_list': old_acc_round_list,
            'new_acc_round_list': new_acc_round_list,
            'coverage_round_list': coverage_round_list,
            'ratio_round_list': ratio_round_list,
            'entropy_round_list': entropy_round_list,
        }
        args.logger.info('Saving the state of round {}...'.format(rd))
        torch.save(save_round_dict, save_round_path)

    # logger results of acc across whole AL process
    args.logger.info('\n\nFinal results:')
    args.logger.info('='*150)
    args.logger.info('='*150)
    args.logger.info(f'Initial Accuracies on test set before AL: All: {all_acc_initial:.4f} Old: {old_acc_initial:.4f} New: {new_acc_initial:.4f}')
    for rd_ in range(0, args.num_round):
        args.logger.info(f'Accuracies on test set at AL Round {rd_+1}: All: {all_acc_round_list[rd_]:.4f} Old: {old_acc_round_list[rd_]:.4f} New: {new_acc_round_list[rd_]:.4f}')
    for rd_ in range(0, args.num_round):
        args.logger.info(f'Novelty at AL Round {rd_+1}: Novel Coverage {coverage_round_list[rd_]:.4f} | Novel Ratio {ratio_round_list[rd_]:.4f} | Novel Entropy {entropy_round_list[rd_]:.4f} (upper bound {upper_bound_round:.4f})')
    # overall novelty
    args.logger.info(f'Novelty Overall: Novel Coverage {coverage_overall:.4f} | Novel Ratio {ratio_overall:.4f} | Novel Entropy {entropy_overall:.4f} (upper bound {upper_bound_round:.4f})')


    # print info
    print('='*150)
    print('='*150)
    print(f'Initial Accuracies on test set before AL: All: {all_acc_initial:.4f} Old: {old_acc_initial:.4f} New: {new_acc_initial:.4f}')
    for rd_ in range(0, args.num_round):
        print(f'Accuracies on test set at AL Round {rd_+1}: All: {all_acc_round_list[rd_]:.4f} Old: {old_acc_round_list[rd_]:.4f} New: {new_acc_round_list[rd_]:.4f}')
    for rd_ in range(0, args.num_round):
        print(f'Novelty at AL Round {rd_+1}: Novel Coverage {coverage_round_list[rd_]:.4f} | Novel Ratio {ratio_round_list[rd_]:.4f} | Novel Entropy {entropy_round_list[rd_]:.4f} (upper bound {upper_bound_round:.4f})')
    # overall novelty
    print(f'Novelty Overall: Novel Coverage {coverage_overall:.4f} | Novel Ratio {ratio_overall:.4f} | Novel Entropy {entropy_overall:.4f} (upper bound {upper_bound_round:.4f})')
