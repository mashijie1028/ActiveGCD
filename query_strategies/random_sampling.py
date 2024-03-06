import numpy as np
from .strategy import Strategy


class RandomSampling(Strategy):
    def __init__(self, al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args):
        super(RandomSampling, self).__init__(al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                                             original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args)

    def query(self, n, current_round):
        return np.random.choice(np.where(self.al_dataset.labeled_idxs==0)[0], n, replace=False)
