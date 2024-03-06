import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy


'''
sampling from novel classes via prediction
criterion: margin
class-wise sampling
'''


class NovelMarginSampling(Strategy):
    def __init__(self, al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args):
        super(NovelMarginSampling, self).__init__(al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args)

    def query(self, n, current_round):
        num_old_classes = self.args.num_labeled_classes
        num_novel_classes = self.args.num_unlabeled_classes
        num_per_class = int(n/num_novel_classes)
        unlabeled_idxs, unlabeled_data = self.al_dataset.get_unlabeled_data(self.test_transform)
        probs = self.predict_prob(unlabeled_data)

        # Margin
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:, 1]
        preds = probs.max(1)[1]

        # remapping with sorted indexes by margin
        sorted_idxs = uncertainties.sort()[1]
        probs = probs[sorted_idxs]
        uncertainties = uncertainties[sorted_idxs]
        preds = preds[sorted_idxs]
        unlabeled_idxs = unlabeled_idxs[sorted_idxs]

        # select max entropy samples from novel clusters
        final_idxs_list = []
        for i in range(num_novel_classes):
            novel_idx_i = unlabeled_idxs[preds==(i+num_old_classes)]
            min_select_i = min(len(novel_idx_i), num_per_class)
            final_idxs_list.append(novel_idx_i[:min_select_i])

        final_idxs = np.concatenate(final_idxs_list)
        # append more large entropy data if selected less (some novel clusters a small)
        if len(final_idxs) < n:
            novel_idxs = unlabeled_idxs[preds>=num_old_classes]
            diff_idxs = np.setdiff1d(novel_idxs, final_idxs, True)
            diff_num = n - len(final_idxs)
            final_idxs = np.concatenate([final_idxs, diff_idxs[:diff_num]])

        # further check the selected number
        assert len(final_idxs)==n, 'not enough (pseudo-) novel data!'

        return final_idxs
