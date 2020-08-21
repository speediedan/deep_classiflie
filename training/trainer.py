import logging
from collections import defaultdict
import sys
import os
from typing import MutableMapping, List, Dict, Tuple

import numpy as np
import torch
import tqdm
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torchcontrib.optim.swa import SWA

import utils.constants as constants
from models.deep_classiflie_module import DeepClassiflie
from dataprep.dataprep import DatasetCollection
from training.training_utils import metric_scoring_func, metric_rank, save_ckpt, load_ckpt, compute_metrics, set_seed, \
    init_fp16_model_opt, TrainingSession, log_tbwriter_metrics, dump_default_thawing_schedule
from analysis.inference import Inference

logger = logging.getLogger(constants.APP_NAME)

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError as error:
    logger.debug(f"{error.__class__.__name__}: No apex module found, fp16 will not be available.")


class Trainer(object):
    def __init__(self, data: DatasetCollection, config: MutableMapping) -> None:
        self.model = DeepClassiflie(config)
        if config.trainer.dump_model_thaw_sched_only:
            dump_default_thawing_schedule(self.model, f"{config.experiment.dc_base}/thaw_schedules")
            sys.exit(0)
        self.data = data
        self.training_session = TrainingSession(config, self.model.__class__.__name__,
                                                self.data.dataset_conf['num_train_recs'],
                                                self.data.dataset_conf['train_batch_size'])
        if self.training_session.config.trainer.histogram_vars:
            self.training_session.histogram_vars = {n:p for (n, p) in self.model.named_parameters() if any(
                n == v for v in self.training_session.config.trainer.histogram_vars)}
        self.optimizer = self.init_optimizer()
        self.tokenizer = self.data.dataset_conf['albert_tokenizer']
        self.datasets = {'train': self.data.dataset_conf['train_ds'], 'val': self.data.dataset_conf['val_ds'],
                         'test': self.data.dataset_conf['test_ds']}

    def init_optimizer(self, mode: str = 'normal', ft_params: List = None) -> AdamW:
        self.model.to(self.training_session.device)
        no_decay = ['bias', 'LayerNorm.weight']
        if mode == 'ft':
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters()
                            if (not any(nd in n for nd in no_decay))
                            and (not any(ftp in n for ftp in ft_params))],
                 'weight_decay': self.training_session.config.trainer.optimizer_params.weight_decay},
                {'params': [p for n, p in self.model.named_parameters()
                            if (any(nd in n for nd in no_decay))
                            and (not any(ftp in n for ftp in ft_params))],
                 'weight_decay': 0.0},
                {'params': [p for n, p in self.model.named_parameters()
                            if (not any(nd in n for nd in no_decay))
                            and (any(ftp in n for ftp in ft_params))],
                 'weight_decay': self.training_session.config.trainer.optimizer_params.weight_decay,
                 'lr': self.training_session.config.trainer.fine_tune_scheduler.base_max_lr},
                {'params': [p for n, p in self.model.named_parameters()
                            if (any(nd in n for nd in no_decay))
                            and (any(ftp in n for ftp in ft_params))],
                 'weight_decay': 0.0,
                 'lr': self.training_session.config.trainer.fine_tune_scheduler.base_max_lr},
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.training_session.config.trainer.optimizer_params.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
        return AdamW(optimizer_grouped_parameters,
                     lr=self.training_session.config.trainer.optimizer_params.learning_rate,
                     eps=self.training_session.config.trainer.optimizer_params.adam_epsilon,
                     amsgrad=self.training_session.config.trainer.optimizer_params.amsgrad)

    def train(self, restart_checkpoint: str = None) -> None:
        self.init_train(restart_checkpoint)
        train_loss, prev_loss, val_loss = 0.0, 0.0, 0.0
        train_iterator = tqdm.trange(int(self.training_session.training_epochs_remaining), desc="Epoch")
        stop_early = False
        # Train the model
        for epoch in train_iterator:
            if not stop_early:
                intra_epoch_iterator = tqdm.tqdm(self.training_session.train_dataloader, desc="Iteration")
                for step, batch in enumerate(intra_epoch_iterator):
                    train_loss = self.train_loop(batch, train_loss)
                    # Update learning rate schedule
                    self.training_session.scheduler.step(epoch + (step / len(intra_epoch_iterator)))
                self.train_capture_metrics(train_loss, prev_loss, train_iterator, epoch)
                prev_loss = train_loss
                if self.training_session.config.trainer.earlystopping:
                    test_metric = self.training_session.training_metrics[self.training_session.monitor_metric][-1]
                    stop_early = self.training_session.stop_early_test(test_metric)
            else:
                # stopped early, if performing recursive fine-tuning, reload best checkpoint, modify params and continue
                stop_early = self.recursive_ft_iter()

        self.finish_training()

    def init_train(self, restart_checkpoint: str = None) -> None:
        ft_conf = False
        if restart_checkpoint and self.training_session.fine_tune_scheduler:
            self.recursive_ft_restart(restart_checkpoint)
            ft_conf = True
        elif restart_checkpoint:
            logger.info(f"Restarting training starting with epoch {self.training_session.config.trainer.next_epoch}, "
                        f"(global step {self.training_session.global_step})")
            self.model, self.optimizer, _ = load_ckpt(self.model, restart_checkpoint, 'train', self.optimizer,
                                                      mp=self.training_session.config.trainer.fp16)
        elif self.training_session.fine_tune_scheduler:
            # not restarting and using the fine_tune_scheduler, so must initialize it
            self.ft_config(init_ft=True)
            ft_conf = True
        self.init_train_config(ft_conf)

    def init_train_config(self, ft_conf):
        self.training_session.train_sampler = RandomSampler(self.datasets['train'])
        self.training_session.train_dataloader = DataLoader(self.datasets['train'],
                                                            sampler=self.training_session.train_sampler,
                                                            batch_size=self.training_session.train_batch_size)
        self.model.to(self.training_session.device)
        self.training_session.training_epochs_remaining = \
            self.training_session.config.trainer.epochs - self.training_session.next_epoch
        self.training_session.train_steps_total = \
            len(self.training_session.train_dataloader) * self.training_session.training_epochs_remaining
        self.init_training_msgs()
        self.init_train_optional_cfg(ft_conf)
        self.model.zero_grad()
        set_seed(self.training_session.config.trainer.seed)  # Added here for reproductibility

    def init_train_optional_cfg(self, ft_conf):
        if not ft_conf:
            if self.training_session.config.trainer.fp16:
                self.model, self.optimizer = \
                    init_fp16_model_opt(self.model, self.training_session.config.trainer.fp16_opt_level, self.optimizer)
            self.config_scheduler()
        if self.training_session.config.trainer.add_summary:
            # N.B. in order to get add_graph working with some complex models one may need to set check_trace=False
            # to default in the core pytorch trace function defined in torch.jit.__init__.py
            self.add_summary()
        if (not self.training_session.fine_tune_scheduler) and \
                self.training_session.config.trainer.optimizer_params.swa_mode == "last":
            self.training_session.capture_swa_snaps = True
            self.optimizer = SWA(self.optimizer)

    def recursive_ft_restart(self, restart_checkpoint):
        logger.info(f"Restarting training starting with epoch {self.training_session.config.trainer.next_epoch}, "
                    f"(global step {self.training_session.global_step})")
        self.model, cust_dict = load_ckpt(self.model, restart_checkpoint,
                                          'train', mp=self.training_session.config.trainer.fp16)
        curr_depth = cust_dict['fts_state']['curr_depth']
        if cust_dict['fts_state']['curr_depth'] and cust_dict['fts_state']['curr_depth'] > 0:
            self.model, ft_params = self.training_session.fine_tune_scheduler.restart_ft(self.model, curr_depth)
            logger.info(f"Restored fine tuning scheduler state current depth to"
                        f" {cust_dict['fts_state']['curr_depth']}")
            self.ft_config(ckpt=restart_checkpoint, ft_params=ft_params)
        else:
            logger.info(f"No previous fine tuning scheduler state found in provided checkpoint, "
                        f"or saved depth was 0. Proceeding with inital depth=0")
            self.ft_config(ckpt=restart_checkpoint, init_ft=True)

    def init_training_msgs(self) -> None:
        logger.info("***** Running training *****")
        logger.info(f"Num examples = {len(self.datasets['train'])}")
        logger.info(f"Num training epochs remaining = {self.training_session.training_epochs_remaining}")
        logger.info(f"Train batch size = {self.training_session.train_batch_size}"),
        logger.info(f"Total optimization steps = {self.training_session.train_steps_total}")

    def train_loop(self, batch: Tuple[torch.Tensor], train_loss: float) -> float:
        self.model.train()  # set module to training mode
        batch = tuple(t.to(self.training_session.device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'ctxt_type': batch[3],
                  'labels': batch[4]}
        outputs = self.model(**inputs)
        loss = outputs[0]
        train_loss += loss.item()
        if self.training_session.config.trainer.fp16:
            if self.training_session.capture_swa_snaps:
                with amp.scale_loss(loss, self.optimizer.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer),
                                           self.training_session.config.trainer.optimizer_params.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.training_session.config.trainer.optimizer_params.max_grad_norm)
        self.optimizer.step()
        self.model.zero_grad()
        self.training_session.global_step += 1
        return train_loss

    def train_capture_metrics(self, train_loss: float, prev_loss: float, train_iterator: tqdm, epoch: int) -> None:
        if self.training_session.capture_swa_snaps and \
                (((len(train_iterator) - epoch) <= self.training_session.config.trainer.optimizer_params.last_swa_snaps)
                 or (self.training_session.stop_early_test.tests_remaining <=
                     self.training_session.config.trainer.optimizer_params.last_swa_snaps)):
            self.optimizer.update_swa()
            self.training_session.captured_swa_snaps += 1
        # N.B. checkpoint frequency using epoch as opposed to global_step units
        if self.training_session.config.trainer.checkpoint_freq > 0 \
                and epoch % self.training_session.config.trainer.checkpoint_freq == 0:
            # Log metrics
            self.init_evaluation()
            self.training_session = log_tbwriter_metrics(self.training_session, train_loss, prev_loss)
            self.save_best(self.training_session.training_metrics[self.training_session.monitor_metric][-1])

    def recursive_ft_iter(self) -> bool:
        stop_early = True
        self.global_ckpt_update()
        if self.training_session.fine_tune_scheduler:
            if self.training_session.fine_tune_scheduler.depth_remaining > 0:
                self.prep_next_train_level()
                self.training_session.stop_early_test.reset_cnt()
                stop_early = False
        return stop_early

    def finish_training(self) -> None:
        logger.info(f"trained steps (global) = {self.training_session.global_step}")
        if self.training_session.config.trainer.optimizer_params.swa_mode in ["best", "last"]:
            swa_checkpointfile = self.gen_swa_ckpt()
            logger.info(f"Captured {self.training_session.captured_swa_snaps} swa snaps and have built swa ckpt"
                        f" {swa_checkpointfile} "
                        f"for subsequent testing.")
        logger.info(f"Evaluating trained model with best checkpoint "
                    f"{self.training_session.global_best_ckpt[1]}, having best"
                    f" {self.training_session.monitor_metric} = {self.training_session.global_best_ckpt[0]}")
        self.init_test(self.training_session.global_best_ckpt[1])
        self.training_session.tbwriter.close()

    def global_ckpt_update(self) -> None:
        if not self.training_session.global_best_ckpt:
            self.training_session.global_best_ckpt = (self.training_session.best_checkpoints[0][0],
                                                      self.training_session.best_checkpoints[0][1])
        else:
            if metric_scoring_func(self.training_session.monitor_metric, self.training_session.best_checkpoints[0][0],
                                   self.training_session.global_best_ckpt[0]):
                self.training_session.global_best_ckpt = (self.training_session.best_checkpoints[0][0],
                                                          self.training_session.best_checkpoints[0][1])

    def prep_next_train_level(self) -> None:
        # append training summaries to training history after each training
        # level completes depth starting with initial, base-frozen, training level 0
        self.training_session.training_summaries.append((self.training_session.training_metrics,
                                                         self.training_session.best_checkpoints))
        self.global_ckpt_update()
        if not self.training_session.config.trainer.fine_tune_scheduler.keep_ckpts_global:
            self.training_session.best_checkpoints = []
        self.training_session.training_metrics = defaultdict(list)
        if self.training_session.config.experiment.debug.log_model_mem_reports:
            self.training_session.model_mem_rpt.log_report("BEFORE CKPT LOAD")
        # discarding state of the optimizer and starting next level of ft from "scratch"
        self.model, _ = load_ckpt(self.model, self.training_session.global_best_ckpt[1], 'train',
                                  mp=self.training_session.config.trainer.fp16)
        if self.training_session.config.experiment.debug.log_model_mem_reports:
            self.training_session.model_mem_rpt.log_report("AFTER CKPT LOAD")
        self.ft_config()

    def ft_config(self, ckpt: str = None, ft_params: List = None, init_ft: bool = False) -> None:
        if init_ft:
            self.model, ft_params = self.training_session.fine_tune_scheduler(self.model, init_ft)
        elif not ft_params:  # we've already configured model parameters if ft_params provided
            self.model, ft_params = self.training_session.fine_tune_scheduler(self.model)
        if self.training_session.config.experiment.debug.log_model_mem_reports:
            self.training_session.model_mem_rpt.log_report("AFTER MODEL PARAM CHANGE")
        self.optimizer = self.init_optimizer(mode='ft', ft_params=ft_params)
        if not ckpt and not init_ft:
            ckpt = self.training_session.global_best_ckpt[1]
        if ckpt:
            logger.info(f"Fine tuning scheduler depth is now {self.training_session.fine_tune_scheduler.curr_depth}")
            logger.info(f"Recursive fine tuning continuing using ckpt {ckpt}")
        if self.training_session.config.trainer.fp16:
            self.model, self.optimizer = init_fp16_model_opt(self.model,
                                                             self.training_session.config.trainer.fp16_opt_level,
                                                             self.optimizer)
        self.config_scheduler()
        if self.training_session.fine_tune_scheduler.depth_remaining == 0:
            if self.training_session.config.trainer.optimizer_params.swa_mode == "last":
                self.training_session.capture_swa_snaps = True
                self.optimizer = SWA(self.optimizer)

    def gen_swa_ckpt(self) -> str:
        if self.training_session.config.trainer.optimizer_params.swa_mode == "last":
            # noinspection PyUnresolvedReferences
            self.optimizer.swap_swa_sgd()  # swap model weights using swa snapshots at end of training if using swa
            swa_checkpointfile = self.save_progress(swa_ckpts=self.training_session.captured_swa_snaps)
        else:
            swa_checkpointfile = self.swa_ckpt_build()
        return swa_checkpointfile

    def config_scheduler(self) -> None:
        self.training_session.scheduler = \
            CosineAnnealingWarmRestarts(self.optimizer,
                                        self.training_session.config.trainer.optimizer_params.init_lr_cycle_period,
                                        self.training_session.config.trainer.optimizer_params.lr_cycle_mult_fact,
                                        self.training_session.config.trainer.optimizer_params.min_lr)

    def add_summary(self) -> None:
        sample_batch = next(iter(self.training_session.train_dataloader))
        sample_batch = tuple(t.to(self.training_session.device) for t in sample_batch)
        logger.debug(f"raw input_ids: {sample_batch[0]},"
                     f"raw attention_mask: {sample_batch[1]},"
                     f"raw token_type_ids: {sample_batch[2]}, "
                     f"raw ctxt_type: {sample_batch[3]}, "
                     f"raw label: {sample_batch[4]}")
        if self.training_session.config.trainer.add_summary:
            # don't supply attention_mask since add_graph doesn't appear to support dictionary of inputs at this point
            self.training_session.tbwriter.add_graph(self.model, [sample_batch[0], sample_batch[4], sample_batch[1],
                                                                  sample_batch[2], sample_batch[3]])

    def init_test(self, checkpoint_path: str) -> None:
        self.model, _ = load_ckpt(self.model, checkpoint_path, 'eval')
        self.model.to(self.training_session.device)
        if not self.training_session.config.experiment.predict_only:
            self.init_evaluation(dset='test', ckpt=checkpoint_path)
        self.init_predict(checkpoint_path)

    def save_best(self, curr_metric: float) -> None:
        score_rank = metric_rank(self.training_session.monitor_metric, self.training_session.best_checkpoints,
                                 (curr_metric,)) \
            if len(self.training_session.best_checkpoints) > 0 else 0
        if score_rank < self.training_session.config.trainer.keep_best_n_checkpoints:
            checkpoint_file = self.save_progress(curr_metric)
            self.training_session.best_checkpoints.insert(score_rank, (curr_metric, checkpoint_file,
                                                                       self.training_session.last_completed_epoch))
            if len(self.training_session.best_checkpoints) > \
                    self.training_session.keep_n:
                for i, (_, ckpt, _) in enumerate(self.training_session.best_checkpoints[self.training_session.keep_n:]):
                    if os.path.exists(ckpt):
                        logger.info(f"Checkpoint {ckpt} no longer in top {self.training_session.keep_n} "
                                    f"best epochs. Deleting checkpoint...")
                        os.remove(ckpt)
                        self.training_session.best_checkpoints.pop(self.training_session.keep_n + i)
        else:
            logger.info(
                f"Current epoch's monitored metric ({self.training_session.monitor_metric} = "
                f"{curr_metric}) not"
                f" in top {self.training_session.keep_n} best epochs. Not saving checkpoint.")

    def save_progress(self, curr_metric: float = 0.0, swa_ckpts: int = 0, save_tokenizer: bool = False) -> str:
        if swa_ckpts > 0:
            swa_msg = f"swa_last_{swa_ckpts}_ckpts-" if \
                self.training_session.config.trainer.optimizer_params.swa_mode == "last" \
                else f"swa_best_{swa_ckpts}_ckpts-"
        else:
            swa_msg = ""
        self.training_session.last_completed_epoch = \
            int(self.training_session.global_step // self.training_session.num_train_steps) - 1
        checkpoint_file = f"{self.training_session.config.experiment.dirs.checkpoint_dir}/" \
                          f"checkpoint-{curr_metric:.4f}-{swa_msg}{self.training_session.last_completed_epoch}-" \
                          f"{self.training_session.global_step}{constants.CKPT_EXT}"
        cust_dict = None
        if self.training_session.fine_tune_scheduler:
            cust_dict = {'fts_state': {'curr_depth': self.training_session.fine_tune_scheduler.curr_depth}}
        amp_state_dict = amp.state_dict() if self.training_session.config.trainer.fp16 else None
        checkpoint_dict_file = save_ckpt(self.training_session.config, self.model, self.model.state_dict(),
                                         self.optimizer.state_dict(), checkpoint_file, amp_state_dict, cust_dict)
        if save_tokenizer:
            # usually no need to save tokenizer config since we'll be using default config,
            # but will save on last checkpoint sometimes just in case (previous tokenizer saves will be overwritten)
            self.tokenizer.save_pretrained(self.training_session.config.experiment.dirs.checkpoint_dir)
        return checkpoint_dict_file

    def init_evaluation(self, dset: str = 'val', ckpt: str = "") -> None:
        if dset == 'val':
            eval_dataset = self.datasets['val']
            eval_batch_size = self.data.dataset_conf['val_batch_size']
        else:
            eval_dataset = self.datasets['test']
            eval_batch_size = self.data.dataset_conf['test_batch_size']
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
        ckpt = f"on checkpoint {ckpt}" if not ckpt == "" else ""
        logger.info(f"***** Running evaluation {ckpt} *****")
        logger.info(f"Num examples = {len(eval_dataset)}")
        logger.info(f"Batch size = {eval_batch_size}")
        self.evaluate(eval_dataloader, dset)

    def evaluate(self, eval_dataloader: DataLoader, dset: str) -> None:
        eval_loss, num_eval_steps = 0.0, 0
        preds, out_label_ids = None, None
        for batch in tqdm.tqdm(eval_dataloader, desc=f"Evaluating on {dset} set"):
            self.model.eval()  # set model to evaluation mode
            batch = tuple(t.to(self.training_session.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'ctxt_type': batch[3],
                          'labels': batch[4]}
                outputs = self.model(**inputs)
                tmp_eval_loss, probs = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            num_eval_steps += 1
            if preds is None:
                preds = probs.numpy()  # probs tensor already detached
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, probs.numpy(), axis=0)  # probs tensor already detached
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / num_eval_steps
        pred_labels = np.reshape(preds, (-1)).round()
        epoch_metrics = compute_metrics(pred_labels, out_label_ids)
        self.log_evaluation(dset, eval_loss, epoch_metrics)

    def log_evaluation(self, dset: str, eval_loss: float, epoch_metrics: Dict) -> None:
        if dset == 'val':
            self.training_session.tbwriter.add_scalar('Loss/val', eval_loss, self.training_session.global_step)
            logger.info(f"Validation set results at global step {self.training_session.global_step}:")
            for key in epoch_metrics.keys():
                logger.info(f"{key} = {epoch_metrics[key]:.2f}")
                self.training_session.tbwriter.add_scalar(f"validation_metrics/{key}", epoch_metrics[key],
                                                          self.training_session.global_step)
                self.training_session.training_metrics[key].append(epoch_metrics[key])
            self.training_session.training_metrics['val_loss'].append(eval_loss)
            self.training_session.training_metrics['epoch'].append(self.training_session.last_completed_epoch)
            for n, p in self.training_session.histogram_vars.items():
                self.training_session.tbwriter.add_histogram(n, p.data, self.training_session.global_step)
        else:
            hparam_dict = {'epochs': self.training_session.config.trainer.epochs}
            hparam_config_nodes = ['fine_tune_scheduler', 'earlystopping', 'optimizer_params']
            for pnodes in [self.training_session.config.trainer[n] for n in hparam_config_nodes]:
                for subk, subv in pnodes.items():
                    hparam_dict[subk] = subv
            logger.info(f"Test set results:")
            for key in epoch_metrics.keys():
                logger.info(f"{key} = {epoch_metrics[key]:.2f}")
            hp_epoch_metrics = {}
            for k in epoch_metrics.keys():
                hp_epoch_metrics[f'test_metrics/{k}'] = epoch_metrics[k]
            self.training_session.tbwriter.add_hparams(hparam_dict, hp_epoch_metrics)

    def init_predict(self, ckpt: str) -> None:
        if not self.training_session.config.inference.pred_inputs:
            eval_dataset = self.datasets['test']
            eval_sampler = RandomSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)
            self.model.eval()  # set model to evaluation mode
            eval_tuple = tuple((eval_dataset, eval_sampler, eval_dataloader))
            Inference(self.training_session.config).init_predict(ckpt=ckpt, model=self.model, tokenizer=self.tokenizer,
                                                                 eval_tuple=eval_tuple)
        else:
            Inference(self.training_session.config).init_predict(ckpt=ckpt)

    def swa_ckpt_build(self, mode: str = "best", ckpt_list: [str] = None) -> str:
        self.optimizer = SWA(self.optimizer)
        if mode == "best":
            ckpts = [c for (_, c, _) in self.training_session.best_checkpoints]
        else:
            ckpts = ckpt_list
            self.model, self.optimizer = init_fp16_model_opt(self.model,
                                                             self.training_session.config.trainer.fp16_opt_level,
                                                             self.optimizer)
        for ckpt in ckpts:
            self.model, _ = load_ckpt(self.model, ckpt, 'train')
            self.optimizer.update_swa()
            self.training_session.captured_swa_snaps += 1
            self.optimizer.swap_swa_sgd()  # swap model weights using swa snapshots at end of training if using swa
        swa_checkpointfile = self.save_progress(swa_ckpts=self.training_session.captured_swa_snaps)
        return swa_checkpointfile
