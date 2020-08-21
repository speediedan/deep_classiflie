import shutil
import os

from transformers.data.processors.utils import InputExample
from torch.utils.data import Dataset, Sampler
from typing import Optional, Iterator, List, Dict, Tuple
import torch
import numpy as np


class TempTextDataset(Dataset):
    def __init__(self, *feat_lists: List) -> None:
        self.examples = list(zip(*feat_lists))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, item: int) -> Tuple:
        return self.examples[item]

    def __iter__(self) -> Iterator:
        return iter(self.examples)


class WeightedRandomDatasetSampler(Sampler):
    # noinspection PyMissingConstructor
    def __init__(self, datasets: List[Dataset], weights: List[float], num_samples: int) -> None:
        self.datasets = datasets
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples

    def __iter__(self) -> Iterator:
        samples = [self._sample_inner_ds(d_id) for d_id in
                   torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()]
        return iter(samples)

    def _sample_inner_ds(self, d_id: int) -> Tuple:
        sample_idx = np.random.choice(len(self.datasets[d_id]), 1)
        sample_idx = int(sample_idx)
        ds = self.datasets[d_id]
        return ds[sample_idx]

    def __len__(self) -> int:
        return self.num_samples


class UnivariateDistReplicator(Sampler):
    # N.B., by default, this class will "weakly converge" distributions on the desired dimension. That is, sub-class
    # samples will not be repeated so to the extent sub-class iterators are exhausted, the output class distribution
    # will remain somewhat (but less) divergent from the target distribution
    # noinspection PyMissingConstructor
    def __init__(self, datasets: List[Dataset], weights: List[float], num_samples: int) -> None:
        self.datasets = datasets
        self.ds_iters = [iter(ds) for ds in self.datasets]
        self._skipped_samples = [0] * len(datasets)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples

    def __iter__(self) -> Iterator:
        samples = [self._sample_inner_ds(d_id) for d_id in
                   torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()]
        samples = filter(lambda x: x is not None, samples)
        return iter(samples)

    @property
    def skipped_samples(self) -> List[int]:
        return self._skipped_samples

    def _sample_inner_ds(self, d_id: int) -> Optional[Tuple]:
        try:
            sample = next(self.ds_iters[d_id])
        except StopIteration:
            self._skipped_samples[d_id] += 1
            return None
        return sample

    def __len__(self) -> int:
        return self.num_samples - sum(self._skipped_samples)


def ds_minority_oversample_gen(sampling_weights: List, gens: List[Iterator], cls_limits: List) -> Sampler:
    max_lim = calc_max_lim(cls_limits, sampling_weights)
    cls_datasets = []
    for i, ds in enumerate(gens):
        stexts, stypes, labels = [], [], []
        for (stext, stype, label) in ds:
            stexts.append(stext)
            stypes.append(stype)
            labels.append(label)
        cls_datasets.append(TempTextDataset(stexts, stypes, labels))
    ds_iter = WeightedRandomDatasetSampler(cls_datasets, sampling_weights, int(max_lim))
    return ds_iter


def validate_normalized(test_arr: List[float]) -> bool:
    v_sum = 0
    if isinstance(test_arr, Dict):
        for k, v in test_arr.items():
            v_sum += v
    else:
        v_sum = sum(test_arr)
    is_valid = False if v_sum != 1 else True
    return is_valid


def parse_sql_to_example(ds_gen: Iterator, k: str) -> Tuple[int, List, List]:
    """ For now, this parsing code needs to be customized based upon the nature of the generating sql statement.
    It could be enhanced w/ column-level feature mappings to avoid some parsing, but the investment isn't worth
    the additional configuration at the moment
    """
    xformer_examples, ctxt_examples = [], []
    recs = 0
    for row in ds_gen:
        xformer_examples.append(gen_example(k, (row[0], row[2])))
        ctxt_examples.append(row[1])
        recs += 1
    return recs, xformer_examples, ctxt_examples


def class_weight_gen(sqlgens: List[Iterator]) -> Iterator:
    while True:
        for i, it in enumerate(sqlgens):
            try:
                yield next(it)
            except StopIteration:
                if len(sqlgens) > 1:
                    del sqlgens[i]
                else:
                    return


def calc_max_lim(cls_limits: List, sampling_weights: List) -> int:
    max_lim = 0
    for l, w in zip(cls_limits, sampling_weights):
        samples = (1 / w) * l
        max_lim = max(max_lim, samples)
    return max_lim


def gen_example(dstype: str, row: Tuple) -> InputExample:
    rec = 0
    guid = f"{dstype}-{rec}"
    return InputExample(guid=guid, text_a=row[0], label=row[1])


def link_swap(file_tup: List[Tuple]) -> None:
    for (s, a) in file_tup:
        shutil.move(s, a)
        if os.path.isfile(s):
            os.remove(s)
        os.symlink(a, s)
