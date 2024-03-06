import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
from utils_al.train_recipes_ema import train_al_ema, test_al

class AL_Net:
    '''
    Active Learning Network Data_Structure
    net: DINOHead_feature, return: x, x_proj, logits
    '''
    def __init__(self, net, net_ema, args, device):
        self.net = net.to(device)
        self.net_ema = None
        if net_ema is not None:
            self.net_ema = net_ema.to(device)
        self.args = args
        self.device = device


    def train(self, train_loader, al_labeled_loader, test_loader, unlabelled_train_loader,
              train_labeled_loader_ind_mapping, al_labeled_loader_ind_mapping, current_round):
        best_test_acc_all, best_test_acc_lab, best_test_acc_ubl = train_al_ema(self.net, self.net_ema, train_loader, al_labeled_loader, test_loader, unlabelled_train_loader,
                                                                                       train_labeled_loader_ind_mapping, al_labeled_loader_ind_mapping, self.args, current_round)
        return best_test_acc_all, best_test_acc_lab, best_test_acc_ubl


    def test(self, test_loader, save_name):
        all_acc, old_acc, new_acc, ind_map_test = test_al(self.net, test_loader, None, save_name, self.args)
        self.args.logger.info('[{}] Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(save_name, all_acc, old_acc, new_acc))
        return all_acc, old_acc, new_acc, ind_map_test


    def predict(self, data):
        self.net.eval()
        preds = torch.zeros(len(data), dtype=torch.int64)
        loader = DataLoader(data, num_workers=self.args.num_workers, batch_size=256, shuffle=False)
        with torch.no_grad():
            for x, y, idxs in tqdm(loader):
                x, y = x.to(self.device), y.to(self.device)
                _, _, out = self.net(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def predict_max_logits(self, data):
        self.net.eval()
        #max_logits = torch.zeros(len(data), dtype=torch.int64)
        max_logits = torch.zeros(len(data))
        loader = DataLoader(data, num_workers=self.args.num_workers, batch_size=256, shuffle=False)
        with torch.no_grad():
            for x, y, idxs in tqdm(loader):
                x, y = x.to(self.device), y.to(self.device)
                _, _, out = self.net(x)
                max_logit = out.max(1)[0] / self.args.logits_temp   # NOTE!!!
                max_logits[idxs] = max_logit.cpu()
        return max_logits


    def predict_prob(self, data):
        self.net.eval()
        probs = torch.zeros([len(data), self.args.num_labeled_classes + self.args.num_unlabeled_classes])
        loader = DataLoader(data, num_workers=self.args.num_workers, batch_size=256, shuffle=False)
        with torch.no_grad():
            for x, y, idxs in tqdm(loader):
                x, y = x.to(self.device), y.to(self.device)
                _, _, out = self.net(x)
                out = out / self.args.logits_temp   # NOTE!!!
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs


    def get_gt_labels(self, data):
        self.net.eval()
        labels = torch.zeros(len(data), dtype=torch.int64)
        loader = DataLoader(data, num_workers=self.args.num_workers, batch_size=256, shuffle=False)
        with torch.no_grad():
            for x, y, idxs in tqdm(loader):
                x, y = x.to(self.device), y.to(self.device)
                # _, _, out = self.net(x)
                # pred = out.max(1)[1]
                labels[idxs] = y.cpu()
        return labels


    def get_gt_labels_pro(self, data):
        self.net.eval()
        labels = torch.zeros(len(data), dtype=torch.int64)
        loader = DataLoader(data, num_workers=self.args.num_workers, batch_size=256, shuffle=False)
        current_idx = 0
        with torch.no_grad():
            for x, y, _ in tqdm(loader):
                x, y = x.to(self.device), y.to(self.device)
                # _, _, out = self.net(x)
                # pred = out.max(1)[1]
                labels[current_idx:current_idx+len(y)] = y.cpu()
                current_idx += len(y)
        return labels


    def predict_prob_dropout(self, data, n_drop=10):
        self.net.eval()
        probs = torch.zeros([len(data), self.args.num_labeled_classes + self.args.num_unlabeled_classes])
        loader = DataLoader(data, num_workers=self.args.num_workers, batch_size=256, shuffle=False)
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    _, _, out = self.net(x)
                    out = out / self.args.logits_temp   # NOTE!!!
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs


    def predict_prob_dropout_split(self, data, n_drop=10):
        self.net.eval()
        probs = torch.zeros([len(data), self.args.num_labeled_classes + self.args.num_unlabeled_classes])
        loader = DataLoader(data, num_workers=self.args.num_workers, batch_size=256, shuffle=False)
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    _, _, out = self.net(x)
                    out = out / self.args.logits_temp   # NOTE!!!
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs


    def get_embeddings(self, data):
        self.net.eval()
        embeddings = torch.zeros([len(data), self.args.feat_dim])
        loader = DataLoader(data, num_workers=self.args.num_workers, batch_size=256, shuffle=False)
        current_idx = 0
        with torch.no_grad():
            for x, y, _ in tqdm(loader):
                x, y = x.to(self.device), y.to(self.device)
                feature, _, out = self.net(x)
                embeddings[current_idx:current_idx+len(y)] = feature.cpu()
                current_idx += len(y)
        return embeddings


    def get_grad_embeddings(self, data):
        self.net.eval()
        #embDim = self.net.get_embedding_dim()
        #nLab = self.params['num_class']
        total_classes = self.args.num_labeled_classes + self.args.num_unlabeled_classes
        embeddings = np.zeros([len(data), self.args.feat_dim * total_classes])

        loader = DataLoader(data, num_workers=self.args.num_workers, batch_size=256, shuffle=False)
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                #cout, out = self.net(x)
                feature, _, out = self.net(x)
                out = out / self.args.logits_temp   # NOTE!!!
                feature = feature.data.cpu().numpy()
                batchProbs = F.softmax(out, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(total_classes):
                        if c == maxInds[j]:
                            embeddings[idxs[j]][self.args.feat_dim * c : self.args.feat_dim * (c+1)] = deepcopy(feature[j]) * (1 - batchProbs[j][c]) * -1.0
                        else:
                            embeddings[idxs[j]][self.args.feat_dim * c : self.args.feat_dim * (c+1)] = deepcopy(feature[j]) * (-1 * batchProbs[j][c]) * -1.0

        return embeddings


    # get logits / activations
    def get_logits(self, data):
        self.net.eval()
        total_classes = self.args.num_labeled_classes + self.args.num_unlabeled_classes
        logits = torch.zeros([len(data), total_classes])
        loader = DataLoader(data, num_workers=self.args.num_workers, batch_size=256, shuffle=False)
        with torch.no_grad():
            for x, y, idxs in tqdm(loader):
                x, y = x.to(self.device), y.to(self.device)
                _, _, out = self.net(x)
                #logit = out / self.args.logits_temp   # NOTE!!!
                logit = out
                logits[idxs] = logit.cpu()
        return logits
