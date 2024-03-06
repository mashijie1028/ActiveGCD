import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

class KMeansSampling(Strategy):
    def __init__(self, al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args):
        super(KMeansSampling, self).__init__(al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args)

    def query(self, n, current_round):
        unlabeled_idxs, unlabeled_data = self.al_dataset.get_unlabeled_data(self.test_transform)
        embeddings = self.get_embeddings(unlabeled_data)
        embeddings = embeddings.numpy()
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embeddings)

        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers)**2
        dis = dis.sum(axis=1)
        q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])

        return unlabeled_idxs[q_idxs]
