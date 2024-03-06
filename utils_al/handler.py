import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import os
from tqdm import tqdm


# CIFAR
########################################################################################################################
########################################################################################################################
class DataHandler(Dataset):
    def __init__(self, X, Y, data_transform):
        self.X = X
        self.Y = Y
        self.data_transform = data_transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)   # NOTE!!!
        x = self.data_transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


# ImageNet
########################################################################################################################
########################################################################################################################
IMAGENET_ROOT = os.path.join('/lustre/datasharing/sjma/ImageNet/ILSVRC2012/imagenet/', 'train')
class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


class ImageNetDataHandler(ImageNetBase):
    def __init__(self, selected_idxs, imgs, samples, targets, uq_idxs, data_transform, target_transform, root=IMAGENET_ROOT):
        super(ImageNetDataHandler, self).__init__(root, data_transform)
        '''
        selected_idxs: np.array or NONE
        '''
        if selected_idxs is not None:
            imgs_ = []
            samples_ = []
            #targets_ = []
            for i in tqdm(selected_idxs):
                imgs_.append(imgs[i])
                samples_.append(samples[i])
                #targets_.append(targets[i])
            self.imgs = imgs_
            self.samples = samples_
            #self.targets = targets_
            self.targets = np.array(targets)[selected_idxs].tolist()
            self.uq_idxs = uq_idxs[selected_idxs]   # NOTE!!!
        else:
            self.imgs = imgs
            self.samples = samples
            self.targets = targets
            self.uq_idxs = uq_idxs


        #self.data_transform = data_transform
        self.target_transform = target_transform
        #self.uq_idxs = uq_idxs

    def __getitem__(self, index):
        x, y, _ = super().__getitem__(index)
        return x, y, index

    def __len__(self):
        return len(self.targets)


# CUB
########################################################################################################################
########################################################################################################################
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
import pandas as pd
CUB_ROOT = '/data4/sjma/dataset/CUB/'


class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx]


class CUBDataHandler(CustomCub2011):
    def __init__(self, selected_idxs, data, uq_idxs, data_transform, target_transform):

        super(CUBDataHandler, self).__init__(root=CUB_ROOT, train=True, transform=data_transform, download=False)
        '''
        selected_idxs: np.array or NONE
        '''

        if selected_idxs is not None:
            mask = np.zeros(len(data)).astype('bool')   # NOTE! len(data) NOT len(self)
            mask[selected_idxs] = True
            # tmp = len(data)   # 5395
            # tmp2 = len(self)   # 5994
            self.data = data[mask]
            self.uq_idxs = uq_idxs[mask]
        else:
            self.data = data
            self.uq_idxs = uq_idxs
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y, _ = super().__getitem__(index)
        return x, y, index

    def __len__(self):
        return len(self.data)



# FGVC-Aircraft
########################################################################################################################
########################################################################################################################
def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file):

    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


AIRCRAFT_ROOT = '/data4/sjma/dataset/FGVC-Aircraft/fgvc-aircraft-2013b/'
class FGVCAircraft(Dataset):

    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft>`_ Dataset.

    Args:
        root (string): Root directory path to dataset.
        class_type (string, optional): The level of FGVC-Aircraft fine-grain classification
            to label data with (i.e., ``variant``, ``family``, or ``manufacturer``).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root, class_type='variant', split='train', transform=None,
                 target_transform=None, loader=default_loader, download=False):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))
        self.root = os.path.expanduser(root)
        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file)
        samples = make_dataset(self.root, image_ids, targets)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = True if split == 'train' else False

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.uq_idxs[index]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'data', 'images')) and \
            os.path.exists(self.classes_file)

    def download(self):
        """Download the FGVC-Aircraft data if it doesn't exist already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s ... (may take a few minutes)' % self.url)
        parent_dir = os.path.abspath(os.path.join(self.root, os.pardir))
        tar_name = self.url.rpartition('/')[-1]
        tar_path = os.path.join(parent_dir, tar_name)
        data = urllib.request.urlopen(self.url)

        # download .tar.gz file
        with open(tar_path, 'wb') as f:
            f.write(data.read())

        # extract .tar.gz to PARENT_DIR/fgvc-aircraft-2013b
        data_folder = tar_path.strip('.tar.gz')
        print('Extracting %s to %s ... (may take a few minutes)' % (tar_path, data_folder))
        tar = tarfile.open(tar_path)
        tar.extractall(parent_dir)

        # if necessary, rename data folder to self.root
        if not os.path.samefile(data_folder, self.root):
            print('Renaming %s to %s ...' % (data_folder, self.root))
            os.rename(data_folder, self.root)

        # delete .tar.gz file
        print('Deleting %s ...' % tar_path)
        os.remove(tar_path)

        print('Done!')


class AircraftDataHandler(FGVCAircraft):
    def __init__(self, selected_idxs, samples, uq_idxs, data_transform, target_transform):

        super(AircraftDataHandler, self).__init__(root=AIRCRAFT_ROOT, transform=data_transform, split='trainval')
        '''
        selected_idxs: np.array or NONE
        '''
        if selected_idxs is not None:
            mask = np.zeros(len(samples)).astype('bool')
            mask[selected_idxs] = True
            self.samples = [(p, t) for i, (p, t) in enumerate(samples) if i in selected_idxs]
            self.uq_idxs = uq_idxs[mask]
        else:
            self.samples = samples
            self.uq_idxs = uq_idxs
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y, _ = super().__getitem__(index)
        return x, y, index

    def __len__(self):
        return len(self.samples)



# Standford Cars
########################################################################################################################
########################################################################################################################
from scipy import io as mat_io


CAR_ROOT = "/data4/sjma/dataset/Stanford-Cars/"
class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, train=True, limit=0, data_dir=CAR_ROOT, transform=None):

        metas = os.path.join(data_dir, 'devkit/cars_train_annos.mat') if train else os.path.join(data_dir, 'devkit/cars_test_annos_withlabels.mat')
        data_dir = os.path.join(data_dir, 'cars_train/') if train else os.path.join(data_dir, 'cars_test/')

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data.append(data_dir + img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data)


class CarsDataHandler(CarsDataset):
    def __init__(self, selected_idxs, data, target, uq_idxs, data_transform, target_transform):

        super(CarsDataHandler, self).__init__(data_dir=CAR_ROOT, transform=data_transform, train=True)
        '''
        selected_idxs: np.array or NONE
        '''
        if selected_idxs is not None:
            self.data = np.array(data)[selected_idxs].tolist()
            self.target = np.array(target)[selected_idxs].tolist()
            self.uq_idxs = uq_idxs[selected_idxs]
        else:
            self.data = data
            self.target = target
            self.uq_idxs = uq_idxs
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y, _ = super().__getitem__(index)
        return x, y, index

    def __len__(self):
        return len(self.data)



# HerbariumDataset19
########################################################################################################################
########################################################################################################################
HERB19_ROOT = '/data4/sjma/dataset/Herbarium19-Small/'

class HerbariumDataset19(torchvision.datasets.ImageFolder):

    def __init__(self, *args, **kwargs):

        # Process metadata json for training images into a DataFrame
        super().__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, idx):

        img, label = super().__getitem__(idx)
        uq_idx = self.uq_idxs[idx]

        return img, label, uq_idx



class Herbarium19DataHandler(HerbariumDataset19):
    def __init__(self, selected_idxs, samples, targets, uq_idxs, data_transform, target_transform):
        super(Herbarium19DataHandler, self).__init__(transform=data_transform, root=os.path.join(HERB19_ROOT, 'small-train'))
        '''
        selected_idxs: np.array or NONE
        '''
        if selected_idxs is not None:
            mask = np.zeros(len(samples)).astype('bool')
            mask[selected_idxs] = True
            self.samples = np.array(samples)[mask].tolist()
            self.targets = np.array(targets)[mask].tolist()
            self.uq_idxs = uq_idxs[mask]

            self.samples = [[x[0], int(x[1])] for x in self.samples]
            self.targets = [int(x) for x in self.targets]
        else:
            # self.samples = samples
            # self.targets = targets
            # self.uq_idxs = uq_idxs

            # self.samples = [[x[0], int(x[1])] for x in self.samples]   # NOTE!!!
            # self.targets = [int(x) for x in self.targets]   # NOTE!!!

            mask = np.ones(len(samples)).astype('bool')
            self.samples = np.array(samples)[mask].tolist()
            self.targets = np.array(targets)[mask].tolist()
            self.uq_idxs = uq_idxs[mask]

            self.samples = [[x[0], int(x[1])] for x in self.samples]
            self.targets = [int(x) for x in self.targets]


        #self.data_transform = data_transform
        self.target_transform = target_transform
        #self.uq_idxs = uq_idxs

    def __getitem__(self, index):
        x, y, _ = super().__getitem__(index)
        return x, y, index

    def __len__(self):
        return len(self.targets)