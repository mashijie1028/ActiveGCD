from data.data_utils import MergedDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.herbarium_19 import get_herbarium_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_datasets, get_imagenet_1k_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets

from copy import deepcopy
import pickle
import os

from config import osr_split_dir


get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'imagenet_1k': get_imagenet_1k_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets
}


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False)
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)
        if args.num_old_classes > 0:
            args.train_classes = range(args.num_old_classes)
            args.unlabeled_classes = range(args.num_old_classes, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)
        if args.num_old_classes > 0:
            args.train_classes = range(args.num_old_classes)
            args.unlabeled_classes = range(args.num_old_classes, 100)


    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']
        if args.num_old_classes > 0:
            train_classes0 = class_splits['Old']
            unlabeled_classes0 = class_splits['New']
            if args.num_old_classes <= len(train_classes0):
                args.train_classes = train_classes0[:args.num_old_classes]
                args.unlabeled_classes = train_classes0[args.num_old_classes:] + unlabeled_classes0
            else:
                args.train_classes = train_classes0 + unlabeled_classes0[:(args.num_old_classes - len(train_classes0))]
                args.unlabeled_classes = unlabeled_classes0[(args.num_old_classes - len(train_classes0)):]


    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)
        if args.num_old_classes > 0:
            args.train_classes = range(args.num_old_classes)
            args.unlabeled_classes = range(args.num_old_classes, 100)

    elif args.dataset_name == 'imagenet_1k':

        args.image_size = 224
        args.train_classes = range(500)
        args.unlabeled_classes = range(500, 1000)

    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
            if args.num_old_classes > 0:
                train_classes0 = class_info['known_classes']
                unlabeled_classes0 = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
                if args.num_old_classes <= len(train_classes0):
                    args.train_classes = train_classes0[:args.num_old_classes]
                    args.unlabeled_classes = train_classes0[args.num_old_classes:] + unlabeled_classes0
                else:
                    args.train_classes = train_classes0 + unlabeled_classes0[:(args.num_old_classes - len(train_classes0))]
                    args.unlabeled_classes = unlabeled_classes0[(args.num_old_classes - len(train_classes0)):]

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)
            if args.num_old_classes > 0:
                args.train_classes = range(args.num_old_classes)
                args.unlabeled_classes = range(args.num_old_classes, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
            if args.num_old_classes > 0:
                train_classes0 = class_info['known_classes']
                unlabeled_classes0 = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
                if args.num_old_classes <= len(train_classes0):
                    args.train_classes = train_classes0[:args.num_old_classes]
                    args.unlabeled_classes = train_classes0[args.num_old_classes:] + unlabeled_classes0
                else:
                    args.train_classes = train_classes0 + unlabeled_classes0[:(args.num_old_classes - len(train_classes0))]
                    args.unlabeled_classes = unlabeled_classes0[(args.num_old_classes - len(train_classes0)):]

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)
            if args.num_old_classes > 0:
                args.train_classes = range(args.num_old_classes)
                args.unlabeled_classes = range(args.num_old_classes, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
            if args.num_old_classes > 0:
                train_classes0 = class_info['known_classes']
                unlabeled_classes0 = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']
                if args.num_old_classes <= len(train_classes0):
                    args.train_classes = train_classes0[:args.num_old_classes]
                    args.unlabeled_classes = train_classes0[args.num_old_classes:] + unlabeled_classes0
                else:
                    args.train_classes = train_classes0 + unlabeled_classes0[:(args.num_old_classes - len(train_classes0))]
                    args.unlabeled_classes = unlabeled_classes0[(args.num_old_classes - len(train_classes0)):]

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)
            if args.num_old_classes > 0:
                args.train_classes = range(args.num_old_classes)
                args.unlabeled_classes = range(args.num_old_classes, 200)

    else:

        raise NotImplementedError

    return args
