import numpy as np
import torch
from .strategy import Strategy
from scipy import stats
from sklearn.metrics import pairwise_distances
import pdb

'''
This implementation is originated from https://github.com/JordanAsh/badge.
Please cite the original paper if you use this method.
@inproceedings{ash2019deep,
  author    = {Jordan T. Ash and
               Chicheng Zhang and
               Akshay Krishnamurthy and
               John Langford and
               Alekh Agarwal},
  title     = {Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds},
  booktitle = {8th International Conference on Learning Representations, {ICLR} 2020,
               Addis Ababa, Ethiopia, April 26-30, 2020},
  publisher = {OpenReview.net},
  year      = {2020}
}
'''

class BadgeSampling(Strategy):
    def __init__(self, al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args):
        super(BadgeSampling, self).__init__(al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args)

    def query(self, n, current_round):
        unlabeled_idxs, unlabeled_data = self.al_dataset.get_unlabeled_data(self.test_transform)
        gradEmbedding = self.get_grad_embeddings(unlabeled_data)
        chosen = init_centers(gradEmbedding, n)
        return unlabeled_idxs[chosen]


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll
