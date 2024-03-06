import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils_al.novelty_metrics import novel_coverage, novel_ratio, novel_uniformity


'''
adapted from: https://github.com/ej0cl6/deep-active-learning/blob/master/query_strategies/strategy.py
update: 2023-10-19
'''


def map_labels(original_labels, ind_map):
    mapped_labels = torch.tensor([ind_map[int(i)] for i in original_labels], dtype=original_labels.dtype, device=original_labels.device)
    return mapped_labels


class Strategy:
    def __init__(self, al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args):
        """_summary_

        Args:
            unlabeled_dataset (_type_): original unlabeled part
            original_train_loader (_type_): _description_
            net (_type_): _description_
        """
        self.al_dataset = al_dataset   # AL dataset data structure (initial unlabeled training set)
        self.original_train_loader = original_train_loader   # original labeled+unlabeled training data
        self.original_test_loader = original_test_loader   # original test data
        self.original_unlabelled_train_loader = original_unlabelled_train_loader   # original unlabeled train loader for test (transductive evaluation)
        self.original_train_labeled_loader_ind_mapping = original_train_labeled_loader_ind_mapping
        self.al_net = al_net
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.args = args


    def query(self, n, current_round):
        pass


    def update(self, pos_idxs, neg_idxs=None):
        self.al_dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.al_dataset.labeled_idxs[neg_idxs] = False


    def measure_novelty(self, query_idxs):
        #self.args.logger.info('Evaluating the novelty metrics on AL selected data (Current Round)...')
        if self.args.dataset_name == 'cifar100' or self.args.dataset_name == 'cifar10':
            selected_labels = self.al_dataset.Y_train[query_idxs]
        elif self.args.dataset_name == 'imagenet_100':
            selected_labels = torch.LongTensor(self.al_dataset.Targets)[query_idxs]   # NOTE!!! LongTensor
        elif self.args.dataset_name == 'cub':
            selected_labels = self.al_dataset.Data.iloc[query_idxs]
            selected_labels = selected_labels.target-1   # NOTE!!! -1
            selected_labels = torch.from_numpy(np.array(selected_labels))
            if self.al_dataset.target_transform is not None:
                selected_labels = torch.LongTensor([self.al_dataset.target_transform(selected_labels[i].item()) for i in range(len(selected_labels))])
        elif self.args.dataset_name == 'scars':
            selected_labels = self.al_dataset.Target
            selected_labels = [selected_labels[i]-1 for i in range(len(selected_labels))]   # NOTE!!! -1
            selected_labels = torch.LongTensor(selected_labels)[query_idxs]   # NOTE query_idxs
            if self.al_dataset.target_transform is not None:
                selected_labels = torch.LongTensor([self.al_dataset.target_transform(selected_labels[i].item()) for i in range(len(selected_labels))])
            else:
                #selected_labels = torch.LongTensor(selected_labels)
                pass
        elif self.args.dataset_name == 'aircraft':
            #path_, selected_labels = self.al_dataset.Samples[query_idxs]
            selected_labels = [self.al_dataset.Samples[i][1] for i in query_idxs]
            if self.al_dataset.target_transform is not None:
                selected_labels = torch.LongTensor([self.al_dataset.target_transform(selected_labels[i]) for i in range(len(selected_labels))])
            else:
                selected_labels = torch.LongTensor(selected_labels)
        elif self.args.dataset_name == 'herbarium_19':
            selected_labels = torch.LongTensor(self.al_dataset.Targets)[query_idxs]   # NOTE!!! LongTensor
            if self.al_dataset.target_transform is not None:
                selected_labels = torch.LongTensor([self.al_dataset.target_transform(selected_labels[i].item()) for i in range(len(selected_labels))])
        else:
            selected_labels = self.al_dataset.Y_train[query_idxs]

        coverage = novel_coverage(selected_labels, self.args.num_labeled_classes, self.args.num_unlabeled_classes)
        ratio = novel_ratio(selected_labels, self.args.num_labeled_classes, self.args.num_unlabeled_classes)
        entropy, upper_bound = novel_uniformity(selected_labels, self.args.num_labeled_classes, self.args.num_unlabeled_classes)
        self.args.logger.info('Novel Coverage {:.4f} | Novel Ratio {:.4f} | Novel Entropy {:.4f} (upper bound {:.4f})'.format(coverage, ratio, entropy, upper_bound))

        return coverage, ratio, entropy, upper_bound


    def measure_novelty_overall(self):
        #self.args.logger.info('Evaluating the novelty metrics on AL selected data (Overall across All Rounds)...')
        labeled_idxs, labeled_data = self.al_dataset.get_labeled_data(self.test_transform)
        coverage, ratio, entropy, upper_bound = self.measure_novelty(labeled_idxs)

        return coverage, ratio, entropy, upper_bound


    def measure_acc(self, query_idxs, ind_map):
        query_data = self.al_dataset.get_indexed_data(query_idxs, self.test_transform)
        preds = self.predict(query_data)
        gt_labels = self.get_gt_labels(query_data)
        gt_labels_mapped = map_labels(gt_labels, ind_map)
        query_acc = (preds == gt_labels_mapped).sum() / len(preds)

        preds_old, preds_new = preds[preds<self.args.num_labeled_classes], preds[preds>=self.args.num_labeled_classes]
        gt_labels_mapped_pred_old, gt_labels_mapped_pred_new = gt_labels_mapped[preds<self.args.num_labeled_classes], gt_labels_mapped[preds>=self.args.num_labeled_classes]
        preds_old_acc, preds_new_acc = (preds_old==gt_labels_mapped_pred_old).sum() / len(preds_old), (preds_new==gt_labels_mapped_pred_new).sum() / len(preds_new)

        preds_gt_old, preds_gt_new = preds[gt_labels<self.args.num_labeled_classes], preds[gt_labels>=self.args.num_labeled_classes]
        gt_labels_mapped_gt_old, gt_labels_mapped_gt_new = gt_labels_mapped[gt_labels<self.args.num_labeled_classes], gt_labels_mapped[gt_labels>=self.args.num_labeled_classes]
        gt_old_acc, gt_new_acc = (preds_gt_old==gt_labels_mapped_gt_old).sum() / len(preds_gt_old), (preds_gt_new==gt_labels_mapped_gt_new).sum() / len(preds_gt_new)

        self.args.logger.info('Query Acc {:.4f} | Pred Old Acc {:.4f} | Pred New Acc {:.4f} | Gt Old Acc {:.4f} | Gt New Acc {:.4f}'.
                              format(query_acc, preds_old_acc, preds_new_acc, gt_old_acc, gt_new_acc))

    def train(self, current_round):
        labeled_idxs, labeled_data = self.al_dataset.get_labeled_data(self.train_transform)
        labeled_idxs, labeled_data_ind_mapping = self.al_dataset.get_labeled_data(self.test_transform)   # NOTE!!! test_transform
        al_labeled_loader = DataLoader(labeled_data, num_workers=self.args.num_workers, batch_size=self.args.al_batch_size, shuffle=True)   # al_batch_size
        al_labeled_loader_ind_mapping = DataLoader(labeled_data_ind_mapping, num_workers=self.args.num_workers, batch_size=256, shuffle=False)   # 256, test batch_size

        #best_test_acc_all, best_test_acc_lab, best_test_acc_ubl = self.al_net.train(self.original_train_loader, al_labeled_loader, self.original_test_loader, self.original_unlabelled_train_loader, self.original_train_loader_mixup, current_round)
        best_test_acc_all, best_test_acc_lab, best_test_acc_ubl = self.al_net.train(self.original_train_loader, al_labeled_loader, self.original_test_loader,
                                                                                    self.original_unlabelled_train_loader,
                                                                                    self.original_train_labeled_loader_ind_mapping, al_labeled_loader_ind_mapping,
                                                                                    current_round)

        return best_test_acc_all, best_test_acc_lab, best_test_acc_ubl


    def test(self):
        self.args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc, ind_map_test = self.al_net.test(self.original_unlabelled_train_loader, 'Train ACC Unlabelled')
        self.args.logger.info('Testing on disjoint test set...')
        all_acc, old_acc, new_acc, ind_map_test = self.al_net.test(self.original_test_loader, 'Test ACC')

        return all_acc, old_acc, new_acc, ind_map_test


    def predict(self, data):
        preds = self.al_net.predict(data)
        return preds


    def predict_prob(self, data):
        probs = self.al_net.predict_prob(data)
        return probs


    def get_gt_labels(self, data):
        gt_labels = self.al_net.get_gt_labels(data)
        return gt_labels


    def get_gt_labels_pro(self, data):
        gt_labels = self.al_net.get_gt_labels_pro(data)
        return gt_labels


    # def predict_prob_dropout(self, data, n_drop=10):
    #     probs = self.al_net.predict_prob_dropout(data, n_drop=n_drop)
    #     return probs


    # def predict_prob_dropout_split(self, data, n_drop=10):
    #     probs = self.al_net.predict_prob_dropout_split(data, n_drop=n_drop)
    #     return probs


    def get_embeddings(self, data):
        embeddings = self.al_net.get_embeddings(data)
        return embeddings


    def get_grad_embeddings(self, data):
        embeddings = self.al_net.get_grad_embeddings(data)
        return embeddings


    def get_logits(self, data):
        logits = self.al_net.get_logits(data)
        return logits




    # query test 4: new_max_entropy, new_min_entropy, old_max_entropy, old_min_entropy
    ####################################################################################################################
    def query_test_acc(self, n, current_round):
        num_old_classes = self.args.num_labeled_classes
        num_novel_classes = self.args.num_unlabeled_classes
        num_per_class = int(n/num_novel_classes)
        num_per_class_old = int(n/num_old_classes)
        unlabeled_idxs, unlabeled_data = self.al_dataset.get_unlabeled_data(self.test_transform)
        probs = self.predict_prob(unlabeled_data)
        log_probs = torch.log(probs)
        uncertainties = (probs*log_probs).sum(1)
        preds = probs.max(1)[1]

        # remapping with sorted indexes by entropy
        sorted_idxs = uncertainties.sort()[1]
        probs = probs[sorted_idxs]
        uncertainties = uncertainties[sorted_idxs]
        preds = preds[sorted_idxs]
        unlabeled_idxs = unlabeled_idxs[sorted_idxs]

        # select max entropy samples from novel clusters
        ################################################################################################################
        new_max_entropy_final_idxs_list = []
        for i in range(num_novel_classes):
            novel_idx_i = unlabeled_idxs[preds==(i+num_old_classes)]
            min_select_i = min(len(novel_idx_i), num_per_class)
            new_max_entropy_final_idxs_list.append(novel_idx_i[:min_select_i])

        new_max_entropy_final_idxs = np.concatenate(new_max_entropy_final_idxs_list)
        # append more large entropy data if selected less (some novel clusters a small)
        if len(new_max_entropy_final_idxs) < n:
            novel_idxs = unlabeled_idxs[preds>=num_old_classes]
            diff_idxs = np.setdiff1d(novel_idxs, new_max_entropy_final_idxs, True)
            diff_num = n - len(new_max_entropy_final_idxs)
            new_max_entropy_final_idxs = np.concatenate([new_max_entropy_final_idxs, diff_idxs[:diff_num]])

        # select min entropy samples from novel clusters
        ################################################################################################################
        new_min_entropy_final_idxs_list = []
        for i in range(num_novel_classes):
            novel_idx_i = unlabeled_idxs[preds==(i+num_old_classes)]
            min_select_i = min(len(novel_idx_i), num_per_class)
            new_min_entropy_final_idxs_list.append(novel_idx_i[-min_select_i:])

        new_min_entropy_final_idxs = np.concatenate(new_min_entropy_final_idxs_list)
        # append more large entropy data if selected less (some novel clusters a small)
        if len(new_min_entropy_final_idxs) < n:
            novel_idxs = unlabeled_idxs[preds>=num_old_classes]
            diff_idxs = np.setdiff1d(novel_idxs, new_min_entropy_final_idxs, True)
            diff_num = n - len(new_min_entropy_final_idxs)
            new_min_entropy_final_idxs = np.concatenate([new_min_entropy_final_idxs, diff_idxs[-diff_num:]])



        # select max entropy samples from old clusters
        ################################################################################################################
        old_max_entropy_final_idxs_list = []
        for i in range(num_old_classes):
            novel_idx_i = unlabeled_idxs[preds==i]
            min_select_i = min(len(novel_idx_i), num_per_class_old)
            old_max_entropy_final_idxs_list.append(novel_idx_i[:min_select_i])

        old_max_entropy_final_idxs = np.concatenate(old_max_entropy_final_idxs_list)
        # append more large entropy data if selected less (some novel clusters a small)
        if len(old_max_entropy_final_idxs) < n:
            novel_idxs = unlabeled_idxs[preds<num_old_classes]
            diff_idxs = np.setdiff1d(novel_idxs, old_max_entropy_final_idxs, True)
            diff_num = n - len(old_max_entropy_final_idxs)
            old_max_entropy_final_idxs = np.concatenate([old_max_entropy_final_idxs, diff_idxs[:diff_num]])

        # select min entropy samples from old clusters
        ################################################################################################################
        old_min_entropy_final_idxs_list = []
        for i in range(num_old_classes):
            novel_idx_i = unlabeled_idxs[preds==i]
            min_select_i = min(len(novel_idx_i), num_per_class_old)
            old_min_entropy_final_idxs_list.append(novel_idx_i[-min_select_i:])

        old_min_entropy_final_idxs = np.concatenate(old_min_entropy_final_idxs_list)
        # append more large entropy data if selected less (some novel clusters a small)
        if len(old_min_entropy_final_idxs) < n:
            novel_idxs = unlabeled_idxs[preds<num_old_classes]
            diff_idxs = np.setdiff1d(novel_idxs, old_min_entropy_final_idxs, True)
            diff_num = n - len(old_min_entropy_final_idxs)
            old_min_entropy_final_idxs = np.concatenate([old_min_entropy_final_idxs, diff_idxs[-diff_num:]])


        return new_max_entropy_final_idxs, new_min_entropy_final_idxs, old_max_entropy_final_idxs, old_min_entropy_final_idxs
