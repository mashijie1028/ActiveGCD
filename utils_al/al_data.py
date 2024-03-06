import numpy as np
import torch
from torchvision import datasets
from .handler import ImageNetDataHandler, CUBDataHandler, AircraftDataHandler, CarsDataHandler, Herbarium19DataHandler


'''
AGCD datasets

generic: CIFAR10, CIFAR100, ImageNet-100
fine-grained: CUB, Stanford Cars, FGVC-Aircraft, Herbarium19
'''


class AL_Data:
    '''
    Active Learning Unlabeled_Dataset Data_Structure
    maintain the AL selected indices for labeling on-the-fly

    reference: https://github.com/ej0cl6/deep-active-learning/blob/master/data.py
    '''
    def __init__(self, X_train, Y_train, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        #self.X_test = X_test
        #self.Y_test = Y_test
        self.handler = handler

        self.n_pool = len(X_train)
        #self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)


    def initialize_labels(self, num=0):
        # generate initial labeled pool
        # tmp_idxs = np.arange(self.n_pool)
        # np.random.shuffle(tmp_idxs)
        # self.labeled_idxs[tmp_idxs[:num]] = True
        pass


    def get_labeled_data(self, data_transform):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs], data_transform)


    def get_unlabeled_data(self, data_transform):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], data_transform)


    def get_indexed_data(self, indexes, data_transform):
        return self.handler(self.X_train[indexes], self.Y_train[indexes], data_transform)


    def get_train_data(self, data_transform):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, data_transform)


    # def get_test_data(self):
    #     return self.handler(self.X_test, self.Y_test)


    # def cal_test_acc(self, preds):
    #     return 1.0 * (self.Y_test==preds).sum().item() / self.n_test



class AL_Data_ImageNet:
    '''
    Active Learning Unlabeled_Dataset Data_Structure
    maintain the AL selected indices for labeling on-the-fly

    reference: https://github.com/ej0cl6/deep-active-learning/blob/master/data.py
    '''
    def __init__(self, Imgs, Samples, Targets, Uq_idxs, target_transform, handler=ImageNetDataHandler):
        self.Imgs = Imgs
        self.Samples = Samples
        self.Targets = Targets
        self.Uq_idxs = Uq_idxs
        self.target_transform = target_transform
        self.handler = handler

        self.n_pool = len(Targets)
        #self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)


    def initialize_labels(self, num=0):
        # generate initial labeled pool
        # tmp_idxs = np.arange(self.n_pool)
        # np.random.shuffle(tmp_idxs)
        # self.labeled_idxs[tmp_idxs[:num]] = True
        pass


    def get_labeled_data(self, data_transform):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        #return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs], data_transform)
        return labeled_idxs, self.handler(labeled_idxs, self.Imgs, self.Samples, self.Targets, self.Uq_idxs, data_transform, self.target_transform)


    def get_unlabeled_data(self, data_transform):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        #return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], data_transform)
        return unlabeled_idxs, self.handler(unlabeled_idxs, self.Imgs, self.Samples, self.Targets, self.Uq_idxs, data_transform, self.target_transform)


    def get_indexed_data(self, indexes, data_transform):
        return self.handler(indexes, self.Imgs, self.Samples, self.Targets, self.Uq_idxs, data_transform, self.target_transform)


    def get_train_data(self, data_transform):
        #return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, data_transform)
        return self.labeled_idxs.copy(), self.handler(None, self.Imgs, self.Samples, self.Targets, self.Uq_idxs, data_transform, self.target_transform)


    # def get_test_data(self):
    #     return self.handler(self.X_test, self.Y_test)


    # def cal_test_acc(self, preds):
    #     return 1.0 * (self.Y_test==preds).sum().item() / self.n_test




class AL_Data_CUB:
    '''
    Active Learning Unlabeled_Dataset Data_Structure
    maintain the AL selected indices for labeling on-the-fly

    reference: https://github.com/ej0cl6/deep-active-learning/blob/master/data.py
    '''
    def __init__(self, Data, Uq_idxs, target_transform, handler=CUBDataHandler):
        self.Data = Data
        self.Uq_idxs = Uq_idxs
        self.target_transform = target_transform
        self.handler = handler

        self.n_pool = len(Data)
        #self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)


    def initialize_labels(self, num=0):
        # generate initial labeled pool
        # tmp_idxs = np.arange(self.n_pool)
        # np.random.shuffle(tmp_idxs)
        # self.labeled_idxs[tmp_idxs[:num]] = True
        pass


    def get_labeled_data(self, data_transform):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        #return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs], data_transform)
        return labeled_idxs, self.handler(labeled_idxs, self.Data, self.Uq_idxs, data_transform, self.target_transform)


    def get_unlabeled_data(self, data_transform):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        #return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], data_transform)
        return unlabeled_idxs, self.handler(unlabeled_idxs, self.Data, self.Uq_idxs, data_transform, self.target_transform)


    def get_indexed_data(self, indexes, data_transform):
        return self.handler(indexes, self.Data, self.Uq_idxs, data_transform, self.target_transform)


    def get_train_data(self, data_transform):
        #return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, data_transform)
        return self.labeled_idxs.copy(), self.handler(None, self.Data, self.Uq_idxs, data_transform, self.target_transform)


    # def get_test_data(self):
    #     return self.handler(self.X_test, self.Y_test)


    # def cal_test_acc(self, preds):
    #     return 1.0 * (self.Y_test==preds).sum().item() / self.n_test



class AL_Data_Aircraft:
    '''
    Active Learning Unlabeled_Dataset Data_Structure
    maintain the AL selected indices for labeling on-the-fly

    reference: https://github.com/ej0cl6/deep-active-learning/blob/master/data.py
    '''
    def __init__(self, Samples, Uq_idxs, target_transform, handler=AircraftDataHandler):
        self.Samples = Samples
        self.Uq_idxs = Uq_idxs
        self.target_transform = target_transform
        self.handler = handler

        self.n_pool = len(Samples)
        #self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)


    def initialize_labels(self, num=0):
        # generate initial labeled pool
        # tmp_idxs = np.arange(self.n_pool)
        # np.random.shuffle(tmp_idxs)
        # self.labeled_idxs[tmp_idxs[:num]] = True
        pass


    def get_labeled_data(self, data_transform):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(labeled_idxs, self.Samples, self.Uq_idxs, data_transform, self.target_transform)


    def get_unlabeled_data(self, data_transform):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        #return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], data_transform)
        return unlabeled_idxs, self.handler(unlabeled_idxs, self.Samples, self.Uq_idxs, data_transform, self.target_transform)


    def get_indexed_data(self, indexes, data_transform):
        return self.handler(indexes, self.Samples, self.Uq_idxs, data_transform, self.target_transform)


    def get_train_data(self, data_transform):
        #return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, data_transform)
        return self.labeled_idxs.copy(), self.handler(None, self.Samples, self.Uq_idxs, data_transform, self.target_transform)


    # def get_test_data(self):
    #     return self.handler(self.X_test, self.Y_test)


    # def cal_test_acc(self, preds):
    #     return 1.0 * (self.Y_test==preds).sum().item() / self.n_test




class AL_Data_Cars:
    '''
    Active Learning Unlabeled_Dataset Data_Structure
    maintain the AL selected indices for labeling on-the-fly

    reference: https://github.com/ej0cl6/deep-active-learning/blob/master/data.py
    '''
    def __init__(self, Data, Target, Uq_idxs, target_transform, handler=CarsDataHandler):
        self.Data = Data
        self.Target = Target
        self.Uq_idxs = Uq_idxs
        self.target_transform = target_transform
        self.handler = handler

        self.n_pool = len(Data)
        #self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)


    def initialize_labels(self, num=0):
        # generate initial labeled pool
        # tmp_idxs = np.arange(self.n_pool)
        # np.random.shuffle(tmp_idxs)
        # self.labeled_idxs[tmp_idxs[:num]] = True
        pass


    def get_labeled_data(self, data_transform):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(labeled_idxs, self.Data, self.Target, self.Uq_idxs, data_transform, self.target_transform)


    def get_unlabeled_data(self, data_transform):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        #return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], data_transform)
        return unlabeled_idxs, self.handler(unlabeled_idxs, self.Data, self.Target, self.Uq_idxs, data_transform, self.target_transform)


    def get_indexed_data(self, indexes, data_transform):
        return self.handler(indexes, self.Data, self.Target, self.Uq_idxs, data_transform, self.target_transform)


    def get_train_data(self, data_transform):
        #return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, data_transform)
        return self.labeled_idxs.copy(), self.handler(None, self.Data, self.Target, self.Uq_idxs, data_transform, self.target_transform)


    # def get_test_data(self):
    #     return self.handler(self.X_test, self.Y_test)


    # def cal_test_acc(self, preds):
    #     return 1.0 * (self.Y_test==preds).sum().item() / self.n_test




class AL_Data_Herb19:
    '''
    Active Learning Unlabeled_Dataset Data_Structure
    maintain the AL selected indices for labeling on-the-fly

    reference: https://github.com/ej0cl6/deep-active-learning/blob/master/data.py
    '''
    def __init__(self, Samples, Targets, Uq_idxs, target_transform, handler=Herbarium19DataHandler):
        self.Samples = Samples
        self.Targets = Targets
        self.Uq_idxs = Uq_idxs
        self.target_transform = target_transform
        self.handler = handler

        self.n_pool = len(Targets)
        #self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)


    def initialize_labels(self, num=0):
        # generate initial labeled pool
        # tmp_idxs = np.arange(self.n_pool)
        # np.random.shuffle(tmp_idxs)
        # self.labeled_idxs[tmp_idxs[:num]] = True
        pass


    def get_labeled_data(self, data_transform):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        #return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs], data_transform)
        return labeled_idxs, self.handler(labeled_idxs, self.Samples, self.Targets, self.Uq_idxs, data_transform, self.target_transform)


    def get_unlabeled_data(self, data_transform):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        #return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], data_transform)
        return unlabeled_idxs, self.handler(unlabeled_idxs, self.Samples, self.Targets, self.Uq_idxs, data_transform, self.target_transform)


    def get_indexed_data(self, indexes, data_transform):
        return self.handler(indexes, self.Samples, self.Targets, self.Uq_idxs, data_transform, self.target_transform)


    def get_train_data(self, data_transform):
        #return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, data_transform)
        return self.labeled_idxs.copy(), self.handler(None, self.Samples, self.Targets, self.Uq_idxs, data_transform, self.target_transform)


    # def get_test_data(self):
    #     return self.handler(self.X_test, self.Y_test)


    # def cal_test_acc(self, preds):
    #     return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

