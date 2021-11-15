# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:14:19 2019

@author: chenlin
adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/phototour.py
make it running with ipi dataset
"""
    
import os
import numpy as np
from PIL import Image

import torch
#from torchvision.vision import VisionDataset

#from .utils import download_url
from tqdm import tqdm
import random
from copy import deepcopy
import torch.utils.data as data

class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, 'transform') and self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transforms: ")
        if hasattr(self, 'target_transform') and self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""

class PhotoTour_IPI(VisionDataset):
    """`Learning Local Image Descriptors Data <http://phototour.cs.washington.edu/patches/default.htm>`_ Dataset.
    Args:
        root (string): Root directory where images are.
        name (string): Name of the dataset to load.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
#    urls = {
#        'notredame_harris': [
#            'http://matthewalunbrown.com/patchdata/notredame_harris.zip',
#            'notredame_harris.zip',
#            '69f8c90f78e171349abdf0307afefe4d'
#        ],
#        'yosemite_harris': [
#            'http://matthewalunbrown.com/patchdata/yosemite_harris.zip',
#            'yosemite_harris.zip',
#            'a73253d1c6fbd3ba2613c45065c00d46'
#        ],
#        'liberty_harris': [
#            'http://matthewalunbrown.com/patchdata/liberty_harris.zip',
#            'liberty_harris.zip',
#            'c731fcfb3abb4091110d0ae8c7ba182c'
#        ],
#        'notredame': [
#            'http://icvl.ee.ic.ac.uk/vbalnt/notredame.zip',
#            'notredame.zip',
#            '509eda8535847b8c0a90bbb210c83484'
#        ],
#        'yosemite': [
#            'http://icvl.ee.ic.ac.uk/vbalnt/yosemite.zip',
#            'yosemite.zip',
#            '533b2e8eb7ede31be40abc317b2fd4f0'
#        ],
#        'liberty': [
#            'http://icvl.ee.ic.ac.uk/vbalnt/liberty.zip',
#            'liberty.zip',
#            'fdd9152f138ea5ef2091746689176414'
#        ],
#    }
    urls = {
        'ipi_dortmund5': [
            'http://matthewalunbrown.com/patchdata/notredame_harris.zip',
            'ipi_dortmund5.zip',
            '69f8c90f78e171349abdf0307afefe4d'
        ],
    }
#    mean = {'notredame': 0.4854, 'yosemite': 0.4844, 'liberty': 0.4437,
#            'notredame_harris': 0.4854, 'yosemite_harris': 0.4844, 'liberty_harris': 0.4437}
#    std = {'notredame': 0.1864, 'yosemite': 0.1818, 'liberty': 0.2019,
#           'notredame_harris': 0.1864, 'yosemite_harris': 0.1818, 'liberty_harris': 0.2019}
#    lens = {'notredame': 468159, 'yosemite': 633587, 'liberty': 450092,
#            'liberty_harris': 379587, 'yosemite_harris': 450912, 'notredame_harris': 325295}
    #check whether this is right mean and std for ipi_dortmund_5 datasets
    mean = {'ipi_dortmund5': 0.4854}
    std = {'ipi_dortmund5': 0.1864}
    lens = {'ipi_dortmund5': 222721}
    image_ext = 'bmp'
    info_file = 'info.txt'
    matches_files = 'm50_250000_250000_0.txt'

    def __init__(self, root, name, train=True, transform=None, download=False):
        super(PhotoTour_IPI, self).__init__(root)
        self.transform = transform
        self.name = name
        self.data_dir = os.path.join(self.root, name)
        self.data_down = os.path.join(self.root, '{}.zip'.format(name))
        self.data_file = os.path.join(self.root, '{}.pt'.format(name))

        self.train = train
        self.mean = self.mean[name]
        self.std = self.std[name]

        if download:
            self.download()

        if not self._check_datafile_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # load the serialized data
        self.data, self.labels, self.matches = torch.load(self.data_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (data1, data2, matches)
        """
        if self.train:
            data = self.data[index]
            if self.transform is not None:
                data = self.transform(data)
            return data
        m = self.matches[index]
        data1, data2 = self.data[m[0]], self.data[m[1]]
        if self.transform is not None:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
        return data1, data2, m[2]

    def __len__(self):
        if self.train:
            return self.lens[self.name]
        return len(self.matches)

    def _check_datafile_exists(self):
        return os.path.exists(self.data_file)

    def _check_downloaded(self):
        return os.path.exists(self.data_dir)

    def download(self):
        if self._check_datafile_exists():
            print('# Found cached data {}'.format(self.data_file))
            return

        if not self._check_downloaded():
            # download files
            url = self.urls[self.name][0]
            filename = self.urls[self.name][1]
            md5 = self.urls[self.name][2]
            fpath = os.path.join(self.root, filename)

#            download_url(url, self.root, filename, md5)

            print('# Extracting data {}\n'.format(self.data_down))

            import zipfile
            with zipfile.ZipFile(fpath, 'r') as z:
                z.extractall(self.data_dir)

            os.unlink(fpath)

        # process and save as torch files
        print('# Caching data {}'.format(self.data_file))

        dataset = (
            read_image_file(self.data_dir, self.image_ext, self.lens[self.name]),
            read_info_file(self.data_dir, self.info_file),
            read_matches_files(self.data_dir, self.matches_files)
        )

        with open(self.data_file, 'wb') as f:
            torch.save(dataset, f)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


def read_image_file(data_dir, image_ext, n):
    """Return a Tensor containing the patches
    """

    def PIL2array(_img):
        """Convert PIL image type to numpy 2D array
        """
        return np.array(_img.getdata(), dtype=np.uint8).reshape(64, 64)

    def find_files(_data_dir, _image_ext):
        """Return a list with the file names of the images containing the patches
        """
        files = []
        # find those files with the specified extension
        for file_dir in os.listdir(_data_dir):
            if file_dir.endswith(_image_ext):
                files.append(os.path.join(_data_dir, file_dir))
        return sorted(files)  # sort files in ascend order to keep relations

    patches = []
    list_files = find_files(data_dir, image_ext)

    for fpath in list_files:
        img = Image.open(fpath)
        for y in range(0, 1024, 64):
            for x in range(0, 1024, 64):
                patch = img.crop((x, y, x + 64, y + 64))
                patches.append(PIL2array(patch))
    return torch.ByteTensor(np.array(patches[:n]))


def read_info_file(data_dir, info_file):
    """Return a Tensor containing the list of labels
       Read the file and keep only the ID of the 3D point.
    """
    labels = []
    with open(os.path.join(data_dir, info_file), 'r') as f:
        labels = [int(line.split()[0]) for line in f]
    return torch.LongTensor(labels)


def read_matches_files(data_dir, matches_file):
    """Return a Tensor containing the ground truth matches
       Read the file and keep only 3D point ID.
       Matches are represented with a 1, non matches with a 0.
    """
    matches = []
    with open(os.path.join(data_dir, matches_file), 'r') as f:
        for line in f:
            line_split = line.split()
            matches.append([int(line_split[0]), int(line_split[3]),
                            int(line_split[1] == line_split[4])])
    return torch.LongTensor(matches)


class TripletPhotoTour_IPI(PhotoTour_IPI):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    urls = {
        'ipi_dortmund5': [
            'http://matthewalunbrown.com/patchdata/notredame_harris.zip',
            'ipi_dortmund5.zip',
            '69f8c90f78e171349abdf0307afefe4d'
        ],
    }
    mean = {'ipi_dortmund5': 0.4854}
    std = {'ipi_dortmund5': 0.1864}
    lens = {'ipi_dortmund5': 222721}
    def __init__(self, train=True, transform=None, batch_size = None, n_triplets = 5000, load_random_triplets = False,  *arg, **kw):
        super(TripletPhotoTour_IPI, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = 1000
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.pairs = self.generate_pairs(self.labels, self.n_triplets)
    def generate_pairs(self,labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds
        triplets = []
        indices = create_indices(labels.numpy())
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()
        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= self.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes - 1)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes - 1)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes - 1)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes - 1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            n3 = np.random.randint(0, len(indices[c2]) - 1)
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))
    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.pairs[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]
        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        # transform images if required
        if True:#args.fliprot:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0,2,1)
                img_p = img_p.permute(0,2,1)
                if self.out_triplets:
                    img_n = img_n.permute(0,2,1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))
                if self.out_triplets:
                    img_n = torch.from_numpy(deepcopy(img_n.numpy()[:,:,::-1]))
        if self.out_triplets:
            return (img_a, img_p, img_n)
        else:
            return (img_a, img_p)

    def __len__(self):
        if self.train:
            return self.pairs.size(0)
        else:
            return self.matches.size(0)