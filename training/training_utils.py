from typing import MutableMapping, Optional, Callable, List, Dict, Tuple, Union
import random
import numpy as np
import logging
from collections import defaultdict
from dataclasses import dataclass
import pathlib
import sys

import psutil
import torch
from sklearn.metrics import matthews_corrcoef as mcc, confusion_matrix
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Sampler, DataLoader
from ruamel.yaml import YAML


import utils.constants as constants
from utils.core_utils import device_config
from training.model_mem_reporter import MemReporter
from utils.core_utils import log_config

logger = logging.getLogger(constants.APP_NAME)

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError as error:
    logger.debug(f"{error.__class__.__name__}: No apex module found, fp16 will not be available.")


def std_model_thaw_layer(model: torch.nn.Module, inc_depth: int = 1) -> Tuple[torch.nn.Module, List]:
    """ Default layer thawing function that will work for model layers that include one weight and one bias
    variable per layer. Explicit layer thawing schedule should be used for thawing of more complex layers."""
    thawed_p_names = []
    thawed_cnt = 0
    target_vars = set(['weight', 'bias'])
    changed_vars = set()
    thaw_limit = 2 * inc_depth
    # noinspection PyArgumentList
    for i, (n, p) in reversed(list(enumerate(model.named_parameters()))):
        if thawed_cnt < thaw_limit:
            if not p.requires_grad:
                p.requires_grad = True
                thawed_p_names.append(n)
                thawed_cnt += 1
        else:
            break
    logger.info(f"Thawed the following module parameters: {[n for n in thawed_p_names]}")
    for n in thawed_p_names:
        changed_vars.add(n.rpartition('.')[2])
    if target_vars - changed_vars:
        raise ValueError(f"Expected layer to have following variables {[s for s in target_vars]} "
                         f"but {[v for v in (target_vars - changed_vars)]} remained unchanged")
    return model, thawed_p_names


def exec_thaw_phase(model: torch.nn.Module, thaw_pl: List) -> Tuple[torch.nn.Module, List]:
    """ Explicit layer thawing schedule implementation."""
    thawed_p_names = []
    thawed_cnt = 0
    # noinspection PyArgumentList
    for i, (n, p) in enumerate(model.named_parameters()):
        if not p.requires_grad and n in thaw_pl:
            p.requires_grad = True
            thawed_p_names.append(n)
            thawed_cnt += 1
    logger.info(f"Thawed the following module parameters: {[n for n in thawed_p_names]}")
    return model, thawed_p_names


def dump_default_thawing_schedule(model: torch.nn.Module, dump_loc: str) -> None:
    logger.info(f"Proceeding with dumping default thawing schedule for {model.__class__.__name__}")
    layer_lists = []
    cur_layer = []
    model_params = list(model.named_parameters())[::-1]
    for i, (n, p) in enumerate(model_params):
        if i % 2 == 0:
            # noinspection PyListCreation
            cur_layer = []
            cur_layer.append(n)
        else:
            cur_layer.append(n)
            layer_lists.append(cur_layer)
    if len(model_params) % 2 == 1:
        layer_lists.append([model_params[-1][0]])
    layer_config = {}
    for i, l in enumerate(layer_lists):
        layer_config[f"thaw_phase_{i}"] = l
    yaml = YAML()
    yaml.dump(layer_config, pathlib.Path(f"{dump_loc}/{model.__class__.__name__}_thaw_schedule.yaml"))
    logger.info(f"Thawing schedule dumped to {dump_loc}. Exiting...")


def metric_scoring_func(metric_name: str, metric_val: float, metric_ref: float) -> bool:
    if metric_name == 'val_loss':
        return True if metric_val < metric_ref else False
    elif metric_name in ['acc', 'mcc']:
        return True if metric_val > metric_ref else False
    else:
        logger.error(f"Metric name {metric_name} not supported.")
        raise ValueError(f"Metric name {metric_name} not supported.")


@dataclass
class TrainingSession:
    config: MutableMapping
    model_name: str
    num_train_recs: int
    train_batch_size: int
    global_best_ckpt: Tuple[float, str] = None
    train_sampler: Sampler = None
    train_dataloader: DataLoader = None
    scheduler: CosineAnnealingWarmRestarts = None
    train_steps_total: int = 0
    training_epochs_remaining: int = 0
    last_completed_epoch: int = 0
    capture_swa_snaps: bool = False
    captured_swa_snaps: int = 0

    def __post_init__(self):
        self.device = device_config(self.config)
        self.p_id = psutil.Process()
        self.training_metrics = defaultdict(list)
        self.best_checkpoints: List[Tuple[float, str, int]] = []
        self.training_summaries = []
        self.histogram_vars = {}
        self.keep_n = self.config.trainer.keep_best_n_checkpoints
        self.tbwriter = SummaryWriter(self.config.experiment.dirs.instance_log_dir)
        self.next_epoch = self.config.trainer.next_epoch or 0
        self.num_train_steps = self.num_train_recs / self.train_batch_size
        self.monitor_metric = self.config.trainer.earlystopping.monitor_metric
        self.global_step = int(self.next_epoch * self.num_train_steps)
        if self.config.trainer.fine_tune_scheduler.thaw_schedule:
            max_depth = self.config.trainer.fine_tune_scheduler.max_depth or None
            self.fine_tune_scheduler = FineTuningScheduler(max_depth,
                                                           self.config.trainer.fine_tune_scheduler.thaw_schedule)
        elif self.config.trainer.fine_tune_scheduler.max_depth:
            self.fine_tune_scheduler = FineTuningScheduler(self.config.trainer.fine_tune_scheduler.max_depth)
        else:
            self.fine_tune_scheduler = None
        if self.config.trainer.earlystopping:
            self.stop_early_test = EarlyStopping(self.config.trainer.earlystopping.monitor_metric,
                                                 self.config.trainer.earlystopping.patience)
        if self.config.experiment.debug.log_model_mem_reports:
            self.model_mem_rpt = MemReporter(logger_name=constants.APP_NAME)
        log_config(self.config)
        logger.info(f"Training session initialized.")


class EarlyStopping(object):
    """Simple earlystopping class. Scoring lambda not required for vast majority of cases, so using this method"""

    def __init__(self, metric_name: str, patience: int, scoring_func: Callable = metric_scoring_func,
                 prev_best: float = None) -> None:
        self.metric_name = metric_name
        self.patience = patience
        self._counter = 0
        self.best_val = prev_best
        self.scoring_func = scoring_func

    def __call__(self, metric_val):
        if self.best_val is None:
            self.best_val = metric_val
        elif not (self.scoring_func(self.metric_name, metric_val, self.best_val)):
            self._counter += 1
            logger.debug(f"EarlyStopping count {self._counter} of {self.patience} threshold registered")
            if self._counter >= self.patience:
                logger.info(f"Stopping training due to metric name {self.metric_name} "
                            f"not improving for {self.patience} consecutive iterations")
                return True
        else:
            self.best_val = metric_val
            self._counter = 0
        return False

    def reset_cnt(self):
        self._counter = 0

    @property
    def tests_remaining(self):
        return self.patience - self._counter


class FineTuningScheduler(object):
    """Class used to manage progressive thawing of layers used in finetuning base model
    layers typically leveraged in transfer learning."""

    def __init__(self, max_depth: int = None, thaw_schedule: str = None,
                 ft_func: Callable[[torch.nn.Module, Union[int, List]], Tuple[torch.nn.Module, List]] = None) -> None:
        if not (max_depth or thaw_schedule):
            logger.error(f"Invalid FineTuningScheduler configuration specified. "
                         f"You must specify either an explicit thaw_schedule or max_depth. Exiting.")
            sys.exit(1)
        if thaw_schedule:
            self.ft_func = exec_thaw_phase
            thaw_schedule = load_yaml_schedule(thaw_schedule)
            if not max_depth:
                max_depth = len(thaw_schedule)-1
        elif ft_func:
            self.ft_func = ft_func
        else:
            self.ft_func = std_model_thaw_layer
        self.thaw_schedule = thaw_schedule
        self.max_depth = max_depth
        self._curr_thawed_params = []
        self._curr_depth = 0

    def __call__(self, tmp_model: torch.nn.Module, init_ft: bool = False) -> Tuple[torch.nn.Module, List]:
        # if initializing the fine-tuning scheduler using a thaw_schedule, ensure specified phase 0 is configured
        if self.thaw_schedule and init_ft:
            next_tl = self.thaw_schedule[f'thaw_phase_0']
            tmp_model, ft_params = self.ft_func(tmp_model, next_tl)
        else:
            if self.thaw_schedule:
                next_tl = self.thaw_schedule[f'thaw_phase_{self._curr_depth + 1}']
                tmp_model, ft_params = self.ft_func(tmp_model, next_tl)
            else:
                tmp_model, ft_params = self.ft_func(tmp_model)
            self._curr_depth += 1
        self._curr_thawed_params.extend(ft_params)
        logger.debug(f"Current parameters thawed by the fine-tune scheduler: {self._curr_thawed_params}. "
                     f"Current depth is {self.curr_depth}.")
        return tmp_model, self._curr_thawed_params

    def restart_ft(self, tmp_model: torch.nn.Module, restart_depth: int) -> Tuple[torch.nn.Module, List]:
        if self.thaw_schedule:
            next_tl = []
            for i, _ in enumerate(self.thaw_schedule.items()):
                if i <= restart_depth:
                    next_tl.extend(self.thaw_schedule[f'thaw_phase_{i}'])
            # noinspection PyArgumentList
            tmp_model, ft_params = self.ft_func(tmp_model, thaw_pl=next_tl)
        else:
            tmp_model, ft_params = self.ft_func(tmp_model, inc_depth=restart_depth)
        self._curr_thawed_params.extend(ft_params)
        self._curr_depth = restart_depth
        return tmp_model, self._curr_thawed_params

    @property
    def depth_remaining(self) -> int:
        return self.max_depth - self._curr_depth

    @property
    def curr_depth(self) -> int:
        return self._curr_depth


def load_yaml_schedule(schedule_yaml_file: str) -> Dict:
    yaml = YAML()
    try:
        with open(schedule_yaml_file, 'r') as df:
            schedule_dict = yaml.load(df)
    except FileNotFoundError as fnf:
        logger.error(f"Could not find specified thaw scheduling file '{schedule_yaml_file}': {fnf}."
                     f" Please reconfigure and try again.")
        sys.exit(1)
    return schedule_dict


def metric_rank(metric_name: str, score_list: List, curr_metric: Tuple[float],
                score_func: Callable = metric_scoring_func) -> int:
    # a binary search implementation that takes in our metric scoring function
    # (based off of bisect_left, but with custom comparator)
    lo = 0
    high = len(score_list)
    while lo < high:
        mid = (lo + high) >> 1  # bitwise right shift by 1 faster than // 2
        if not score_func(metric_name, curr_metric, score_list[mid]):
            lo = mid + 1
        else:
            high = mid
    return lo


def save_ckpt(experiment_config: MutableMapping, model: torch.nn.Module, model_state_dict: Dict,
              optimizer_state_dict: Dict, ckpt_dict_file: str, amp_state_dict: bool = None,
              cust_dict: Dict = None) -> str:
    """ Save current model and optimizer states for training resumption or evaluation
    Arguments:
        experiment_config: configuration dict of the experiment,
        model: model object,
        model_state_dict: state_dict for the model defined here,
        optimizer_state_dict: current training optimizer state,
        ckpt_dict_file: absolute file path to which we save checkpoint dict
        amp_state_dict: fp16 state_dict if present, containing relevant loss scalers
        cust_dict: a dictionary for passing custom state external to both model and optimizer via checkpoint
    Returns:
        A custom checkpoint dictionary file path. This file includes fp16 state, recursive fine-tuning state,
        optimizer state, model state and the configuration of the experiment
    """
    checkpoint_dict = {
        'experiment_config': experiment_config,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'amp_state_dict': amp_state_dict,
        'cust_dict': cust_dict,
    }
    torch.save(checkpoint_dict, ckpt_dict_file)
    logger.info(f"Training checkpoint for model class {model.__class__.__name__} serialized/saved to {ckpt_dict_file}")
    return ckpt_dict_file


def load_ckpt(model: torch.nn.Module, checkpoint_file_path: str,
              mode: str = 'eval', optimizer: Optional[Optimizer] = None,
              mp: bool = False) -> Union[Tuple[torch.nn.Module, Optimizer, Dict], Tuple[torch.nn.Module, Dict]]:
    # Load a pytorch model and optimizer for training resumption or evaluation
    checkpoint_dict = torch.load(checkpoint_file_path)
    model_state_dict = checkpoint_dict['model_state_dict']
    if mode == 'train':
        model.train()
    else:
        model.eval()
    model.load_state_dict(model_state_dict)
    cust_dict = checkpoint_dict.get('cust_dict')
    if optimizer:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        # N.B. w/ recursive ft, we're discarding state of the optimizer and starting next level of ft from "scratch",
        # so restoring the amp_state_dict is not done in that (most common) context
        if mp:
            try:
                from apex import amp
                amp.load_state_dict(checkpoint_dict['amp_state_dict'])
            except ImportError as err:
                logger.debug(f"{err.__class__.__name__}: No apex module found, fp16 will not be available.")
        return model, optimizer, cust_dict
    else:
        return model, cust_dict


def load_old_ckpt(model: torch.nn.Module, checkpoint_file_path: str, strip_prefix: str = None, mode: str = 'train') \
        -> torch.nn.Module:
    # Load a pytorch model and optimizer for training resumption or evaluation
    model_state_dict = torch.load(checkpoint_file_path)
    model, _ = custom_state_dict_load(model, model_state_dict, strip_prefix)
    if mode == 'train':
        model.train()
    else:
        model.eval()
    return model


def custom_state_dict_load(model: torch.nn.Module, state_dict: Dict, base_prefix: str) -> Tuple[torch.nn.Module, Dict]:
    """Adapted from
    https://github.com/huggingface/transformers/blob/ef74b0f07a190f19c69abc0732ea955e8dd7330f/src/transformers/modeling_utils.py#L474
    """

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = ''
    model_to_load = model
    if not hasattr(model, base_prefix) and any(
            s.startswith(base_prefix) for s in state_dict.keys()):
        start_prefix = base_prefix + '.'
    if hasattr(model, base_prefix) and not any(
            s.startswith(base_prefix) for s in state_dict.keys()):
        model_to_load = getattr(model, base_prefix)

    load(model_to_load, prefix=start_prefix)
    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
            model.__class__.__name__, "\n\t".join(error_msgs)))
    loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
    return model, loading_info


def convert_old_pytorch_keys(state_dict: Dict) -> Dict:
    """Adapted from:
    https://github.com/huggingface/transformers/blob/ef74b0f07a190f19c69abc0732ea955e8dd7330f/src/transformers/modeling_utils.py#L495
    """
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    return state_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_fp16_model_opt(model: torch.nn.Module, opt_level: str, optimizer: Optional[Optimizer] = None)\
        -> Union[Tuple[torch.nn.Module, Optional[Optimizer]], Tuple[torch.nn.Module]]:
    amp.register_float_function(torch, 'sigmoid')
    if optimizer:
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        return model, optimizer
    else:
        model = amp.initialize(model, opt_level=opt_level)
        return model


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    # noinspection PyUnresolvedReferences
    acc_score = (preds == labels).mean()
    mcc_score = mcc(labels, preds)
    tot_samp = preds.shape[0]
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    tn, fp, fn, tp = (tn / tot_samp).round(2), (fp / tot_samp).round(2), (fn / tot_samp).round(2), (
            tp / tot_samp).round(2)
    return {"acc": acc_score, "mcc": mcc_score, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def smoothed_label_bce(pred_labels: torch.Tensor, true_labels: torch.Tensor, smoothing: float = 0.0,
                       reduction: str = "mean") -> torch.Tensor:
    """ Smoothes labels for use w/ BCE loss
    More commonly used w/ categorical crossentropy loss and multiclass classification,
    label smoothing still confers some benefit in the binary label circumstance

    Args:
        pred_labels: predicted labels
        true_labels: smoothed ground truth labels
        smoothing: if smoothing == 0.0, behaves like normal labels
        reduction: One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses

    Returns:
        the calculated loss

    Raises:
        ValueError: If an invalid reduction method is passed
    """
    assert true_labels.dim() == 1, 'this label smoothing function requires a 1-dim label Tensor to smooth for BCE input'
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    raw_signal_losses = F.binary_cross_entropy(pred_labels, true_labels, reduction="none")
    raw_noise_losses = F.binary_cross_entropy(pred_labels, torch.where(true_labels > 0, torch.zeros_like(true_labels),
                                                                       torch.ones_like(true_labels)), reduction="none")
    cum_losses = (confidence * raw_signal_losses + smoothing * raw_noise_losses)
    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def log_tbwriter_metrics(training_session: TrainingSession, train_loss: float, prev_loss: float) -> TrainingSession:
    # noinspection PyTypeChecker
    for i, group_lr in enumerate(training_session.scheduler.get_lr()):
        training_session.tbwriter.add_scalar(f'LR/param_group{i}', group_lr, training_session.global_step)
    training_session.tbwriter.add_scalar('Loss/train',
                                         (train_loss - prev_loss) / training_session.num_train_steps,
                                         training_session.global_step)
    training_session.tbwriter.add_scalar(f'MemUsage/rss-{constants.MEM_MAG}',
                                         training_session.p_id.memory_info().rss / constants.MEM_FACTOR,
                                         training_session.global_step)
    training_session.tbwriter.add_scalar(f'MemUsage/vms-{constants.MEM_MAG}',
                                         training_session.p_id.memory_info().vms / constants.MEM_FACTOR,
                                         training_session.global_step)
    if torch.cuda.is_available():
        training_session.tbwriter.add_scalar(f'MemUsage/cuda_allocated-{constants.MEM_MAG}',
                                             torch.cuda.memory_allocated() / constants.MEM_FACTOR,
                                             training_session.global_step)
        training_session.tbwriter.add_scalar(f'MemUsage/cuda_cached-{constants.MEM_MAG}',
                                             torch.cuda.memory_cached() / constants.MEM_FACTOR,
                                             training_session.global_step)
    return training_session
