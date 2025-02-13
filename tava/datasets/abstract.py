# Copyright (c) Meta Platforms, Inc. and affiliates.
import math

import torch


class CachedIterDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        training: bool = False,
        cache_n_repeat: int = 0,
    ):
        self.training = training
        self.cache_n_repeat = cache_n_repeat

        self._cache = None
        self._n_repeat = 0

    def build_pose_meta_info(self):
        """Load all the pose data."""
        raise NotImplementedError

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        raise NotImplementedError

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if (
            worker_info is None
        ):  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.__len__()
        else:  # in a worker process
            # split workload
            per_worker = int(
                math.ceil(self.__len__() / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.__len__())
            #print('worker_info:', worker_info)
            #print(f'worker_id:{worker_id}, iter_start:{iter_start}, iter_end:{iter_end}')
        if self.training and self.cache_n_repeat==-1:
            from tqdm import tqdm
            self._cache = []
            print(f'Workder_id {worker_id}, prepare cache')
            for i in tqdm(range(iter_start, iter_end)):
                self._cache.append(self.fetch_data(i))
            while True:
                for index in torch.randperm(iter_end - iter_start):
                    yield self.__getitem__(index)
        else:
            if self.training:
                while True:
                    #print(worker_id, iter_start + torch.randperm(iter_end - iter_start))
                    for index in iter_start + torch.randperm(iter_end - iter_start):
                        yield self.__getitem__(index)
            else:
                for index in range(iter_start, iter_end):
                    yield self.__getitem__(index)

    def __getitem__(self, index):
        if self.training and self.cache_n_repeat==-1:
            data = self._cache[index]
            # if self._cache is None:
            #     data = self.fetch_data(index)
            #     self._cache = data  
            # else:
            #     data = self._cache              
            # if index not in self._cache:
            #     data = self.fetch_data(index)
            #     self._cache[index] = data
            # else:
            #     data = self._cache[index]
        elif (
            self.training
            and (self._cache is not None)
            and (self._n_repeat < self.cache_n_repeat)):
            data = self._cache
            self._n_repeat += 1
        else:
            data = self.fetch_data(index)
            self._cache = data
            self._n_repeat = 1
        return self.preprocess(data)
