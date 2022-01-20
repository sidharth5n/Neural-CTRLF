import os
import numpy as np
import json
import torch
import torch.utils.data as data
import random

class Dataset(data.Dataset):
    def __init__(self, opt, split):
        self.dataset = opt.dataset
        self.dataset_path = opt.dataset_path
        self.split = split
        json_file = os.path.join(self.dataset_path, 
                                 opt.dataset + ('_augmented' if opt.augment else '') + 
                                 (f'_fold_{opt.fold}' if opt.fold else '') + '.json')
        # Load the json file which contains additional information about the dataset
        with open(json_file, 'r') as f:
            infos = json.load(f)
        self.vocab_size = len(infos['wtoi'])
        self.mean = infos['mean']
        self.files = os.listdir(os.path.join(self.dataset_path, self.dataset, 'npz', self.split))
        self.indices = [*range(len(self))]
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        """
        img              : numpy.ndarray of shape (1, H, W)
                           Normalized gray scale image
        boxes            : numpy.ndarray of shape (P, 4)
                           GT bbox coordinates in (xc, yc, w, h) format
        embeddings       : numpy.ndarray of shape (P, E)
                           GT embeddding of bbox label
        labels           : numpy.ndarray of shape (P, )
                           GT bbox labels encoded with wtoi
        region_proposals : numpy.ndarray of shape (Q, 4)
                           Region proposals extracted using DTP
        infos            : numpy.ndarray of shape (4, )
                           Original height, width and current height, width of the image
        """
        idx = self.indices[i]
        arr = np.load(os.path.join(self.dataset_path, self.dataset, 'npz', self.split, self.files[idx]))
        img = np.repeat(arr['image'], 3, axis = 0)
        embeddings = arr['embeddings']
        boxes = arr['boxes']
        labels = arr['labels']
        region_proposals = arr['region_proposals']
        infos = arr['infos']        
        return img, boxes, embeddings, labels, region_proposals, infos

class SubsetSampler(data.sampler.Sampler):

    def __init__(self, end, start = 0):
        """
        Parameters
        ----------
        end   : int
                Last index
        start : int, optional
                Starting index. Default is 0.
        """
        self.start = start
        self.end = end

    def __iter__(self):
        start = self.start
        self.start = 0
        return (i for i in range(start, self.end))

    def __len__(self):
        return self.end - self.start

class DataLoader:

    def __init__(self, args, split, params = None, length = 0, num_workers = 0):
        self.split = split
        self.shuffle = True if split == 'train' else False
        self.dataset = Dataset(args, split)
        self.iterator = 0

        if params is not None:
            self.load_state_dict(params)

        num_samples = length if length > 0 else len(self)

        sampler = SubsetSampler(num_samples, self.iterator)

        self.loader = data.DataLoader(dataset = self.dataset, batch_size = 1,
                                      sampler = sampler, num_workers = num_workers)

    def __iter__(self):
        for batch_data in self.loader:
            self.iterator += 1
            if self.iterator >= len(self):
                self.iterator = 0
                if self.shuffle:
                    random.shuffle(self.loader.dataset.indices)
            yield batch_data

    def __len__(self):
        return len(self.dataset)

    def load_state_dict(self, params):
        if 'split' in params:
            assert self.split == params['split']
        self.dataset.indices = params.get('indices', self.dataset.indices)
        self.iterator = params.get('iterator', self.iterator)

    def state_dict(self):
        return {'indices' : self.loader.dataset.indices, 'iterator' : self.iterator, 'split' : self.split}
    
    def get_vocab_size(self):
        return self.dataset.get_vocab_size()


if __name__ == '__main__':
    import argparse
    from opts import str2bool
    from models.PreActResNet import PreActResNet34
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'washington')
    parser.add_argument('--dataset_path', default = 'data')
    parser.add_argument('--augment', type = str2bool, default = False)
    parser.add_argument('--fold', default = -1)
    args = parser.parse_args()
    if args.fold < 0:
        args.fold = None
    dl = DataLoader(args, split = 'train')
    model = PreActResNet34()
    for data in dl:
        for x in data:
            print(type(x), x.shape, x.dtype)
        _=model(data[0])

        break