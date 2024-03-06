import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy

'''
randomly sampling from novel classes via prediction
'''



class NovelSamplingRandom(Strategy):
    def __init__(self, al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args):
        super(NovelSamplingRandom, self).__init__(al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args)

    def query(self, n, current_round):
        num_old_classes = self.args.num_labeled_classes
        num_novel_classes = self.args.num_unlabeled_classes
        num_per_class = int(n/num_novel_classes)
        unlabeled_idxs, unlabeled_data = self.al_dataset.get_unlabeled_data(self.test_transform)
        probs = self.predict_prob(unlabeled_data)
        #log_probs = torch.log(probs)
        #uncertainties = (probs*log_probs).sum(1)
        preds = probs.max(1)[1]
        novel_idxs = unlabeled_idxs[preds>=num_old_classes]

        final_idxs = np.random.choice(novel_idxs, n, replace=False)

        return final_idxs

