import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch


def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices


def subsample_dataset(dataset, idxs):

    imgs_ = []
    for i in idxs:
        imgs_.append(dataset.imgs[i])
    dataset.imgs = imgs_

    samples_ = []
    for i in idxs:
        samples_.append(dataset.samples[i])
    dataset.samples = samples_

    # dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    # dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]

    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


class KCenterGreedy(Strategy):
    def __init__(self, al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args):
        super(KCenterGreedy, self).__init__(al_dataset, original_train_loader, original_test_loader, original_unlabelled_train_loader,
                 original_train_loader_mixup, original_train_labeled_loader_ind_mapping, al_net, train_transform, test_transform, args)

    def query(self, n, current_round):
        #labeled_idxs, train_data = self.al_dataset.get_train_data()
        # original train data
        original_labeled_train_data = self.original_train_labeled_loader_ind_mapping.dataset   # NOTE!!! original
        if self.args.dataset_name == 'imagenet_100':
            #original_labeled_train_data[:2000]
            subsample_indices = subsample_instances(original_labeled_train_data, prop_indices_to_subsample=0.1)
            original_labeled_train_data = subsample_dataset(original_labeled_train_data, subsample_indices)
        num_original_labeled = len(original_labeled_train_data)
        labeled_idxs_original = np.ones(num_original_labeled, dtype=bool)
        embeddings_original = self.get_embeddings(original_labeled_train_data)

        # al candidate data
        labeled_idxs_al, al_train_data = self.al_dataset.get_train_data(self.test_transform)
        embeddings_al = self.get_embeddings(al_train_data)

        # overall embeddings and labeled_idxs
        embeddings = torch.cat([embeddings_original, embeddings_al], dim=0)
        embeddings = embeddings.numpy()
        if self.args.dataset_name == 'imagenet_100':
            print('Performing PCA on the features...')
            pca = PCA(n_components=50)
            embeddings = pca.fit_transform(embeddings)

        labeled_idxs = np.concatenate((labeled_idxs_original, labeled_idxs_al))

        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.al_dataset.n_pool+num_original_labeled)[~labeled_idxs][q_idx_]   # NOTE!!! + num_original_labeled
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)

        #return np.arange(self.al_dataset.n_pool)[(self.al_dataset.labeled_idxs ^ labeled_idxs)]
        converted_labeled_idxs = np.concatenate((labeled_idxs_original, self.al_dataset.labeled_idxs))
        converted_final_idxs = np.arange(self.al_dataset.n_pool+num_original_labeled)[(converted_labeled_idxs ^ labeled_idxs)]
        return converted_final_idxs-num_original_labeled   # NOTE!!! -num_original_labeled


