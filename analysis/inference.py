import logging
from typing import MutableMapping, Optional, Union, List, Dict, Tuple

from transformers import AlbertTokenizer, PreTrainedTokenizer
import torch
import tqdm

import utils.constants as constants
from models.deep_classiflie_module import DeepClassiflie
from analysis.inference_utils import tokens_to_sentence, gen_embed_mappings, prep_mapping_tups, \
    prep_base_mapping_tups, pred_inputs_from_config, pred_inputs_from_test, prep_rpt_tups, prep_pred_exp_tups
from utils.core_utils import log_config
from training.training_utils import load_ckpt
from analysis.inference_utils import InferenceSession
from analysis.interpretation import InterpretTransformer

logger = logging.getLogger(constants.APP_NAME)

try:
    from apex import amp
except ImportError as error:
    logger.debug(f"{error.__class__.__name__}: No apex module found, fp16 will not be available.")


class Inference(object):
    def __init__(self, config: MutableMapping, mapping_set: List[Tuple] = None,
                 analysis_set: List[Union[Tuple, List]] = None,
                 pred_exp_set: List[Tuple] = None,
                 rpt_type: str = None, base_mode: bool = True) -> None:
        self.inf_session = InferenceSession(config, mapping_set, analysis_set, pred_exp_set, rpt_type, base_mode)

    def init_predict(self, model: torch.nn.Module = None, ckpt: str = None, tokenizer: PreTrainedTokenizer = None,
                     eval_tuple: Tuple = None) -> Union[Tuple[List[Tuple], Optional[Dict]], Dict, List[Tuple]]:
        ckpt = self.init_predict_model(model, ckpt)
        self.init_predict_tokenizer(tokenizer, ckpt)
        self.config_interpretation()
        pred_inputs = self.prep_pred_inputs(eval_tuple)
        if self.inf_session.analysis_set:
            return self.gen_model_rpt(pred_inputs)
        elif self.inf_session.pred_exp_set:
            return self.pred_exp_viz(pred_inputs)
        elif self.inf_session.mapping_set:
            return gen_embed_mappings(self.inf_session, pred_inputs)
        elif self.inf_session.config.inference.interpret_preds and self.inf_session.config.experiment.tweetbot.enabled:
            unpublished_reports = self.predict_viz(pred_inputs)
            return unpublished_reports
        elif self.inf_session.config.inference.interpret_preds:
            _ = self.predict_viz(pred_inputs)
        else:
            self.predict(pred_inputs)

    def init_predict_model(self, model: torch.nn.Module, ckpt: str) -> str:
        if not model:
            self.inf_session.model = DeepClassiflie(self.inf_session.config)
            ckpt = ckpt or self.inf_session.config.experiment.inference_ckpt
            if ckpt:
                self.inf_session.model, _ = load_ckpt(self.inf_session.model, ckpt, 'eval')
            elif self.inf_session.base_mode:
                logger.info(f'Proceeding with uninitialized base model to generate dist-based duplicate filter')
            else:
                raise ValueError("A model object or a checkpoint to load one from must be provided to init_predict()")
            self.inf_session.model.to(self.inf_session.device)
        else:
            self.inf_session.model = model
        self.inf_session.model.eval()  # set model to evaluation mode
        if self.inf_session.config.experiment.predict_only and self.inf_session.config.experiment.pred_inputs:
            log_config(self.inf_session.config, self.inf_session.model.__class__.__name__)
        return ckpt

    def init_predict_tokenizer(self, tokenizer: PreTrainedTokenizer, ckpt: str) -> None:
        self.inf_session.tokenizer = tokenizer if tokenizer else \
            AlbertTokenizer.from_pretrained(self.inf_session.config.model.sp_model,
                                            max_len=self.inf_session.config.data_source.max_seq_length, truncation=True)
        self.inf_session.special_token_mask = [self.inf_session.tokenizer.unk_token_id,
                                               self.inf_session.tokenizer.sep_token_id,
                                               self.inf_session.tokenizer.pad_token_id,
                                               self.inf_session.tokenizer.cls_token_id]
        logger.info(f'Predictions from model weights: {ckpt}')

    def config_interpretation(self) -> None:
        if self.inf_session.config.inference.interpret_preds or self.inf_session.analysis_set or \
                self.inf_session.mapping_set or self.inf_session.pred_exp_set:
            self.inf_session.model.set_interpret_mode()
            pred_report_path = f"{self.inf_session.config.experiment.dirs.inference_output_dir}"
            self.inf_session.pred_interpreter = InterpretTransformer(self.inf_session.config,
                                                                     self.inf_session.model, self.inf_session.tokenizer,
                                                                     self.inf_session.device, pred_report_path)

    def prep_pred_inputs(self, eval_tuple: Tuple) -> List[Dict]:
        if not (eval_tuple or self.inf_session.config.inference.pred_inputs or self.inf_session.analysis_set
                or self.inf_session.mapping_set or self.inf_session.pred_exp_set):
            raise ValueError("init_predict must be provided inputs via either test set samples,"
                             "a prediction set or a dataset to score/analyze")
        elif self.inf_session.analysis_set:
            pred_inputs = prep_rpt_tups(self.inf_session)
        elif self.inf_session.pred_exp_set:
            pred_inputs = prep_pred_exp_tups(self.inf_session)
        elif self.inf_session.mapping_set and self.inf_session.base_mode:
            pred_inputs = prep_base_mapping_tups(self.inf_session)
        elif self.inf_session.mapping_set:
            pred_inputs = prep_mapping_tups(self.inf_session)
        elif eval_tuple:
            num_samples = self.inf_session.config.inference.sample_predictions
            pred_inputs = pred_inputs_from_test(self.inf_session, eval_tuple, num_samples)
        else:
            pred_inputs = pred_inputs_from_config(self.inf_session)
        return pred_inputs

    def predict(self, pred_inputs: List[Dict]) -> None:
        for sample in pred_inputs:
            with torch.no_grad():
                inputs = {'input_ids': sample['input_ids'],
                          'attention_mask': sample['attention_mask'],
                          'token_type_ids': sample['token_type_ids'],
                          'ctxt_type': sample['ctxt_type'],
                          'labels': sample['labels'],
                          'position_ids': sample['position_ids']}
                _, probs = (self.inf_session.model(**inputs))[:2]
                token_list = sample['input_ids'].squeeze(0).tolist()
                token_list = list(filter(lambda l: l not in self.inf_session.special_token_mask, token_list))
                prob = round(probs.squeeze(0).item(), 4)
                label = sample['labels'].item() if sample['labels'] in [0, 1] else None
                parsed_sent = tokens_to_sentence(self.inf_session, token_list)
                logger.info(
                    f"Predictions on first {self.inf_session.config.inference.sample_predictions} samples in test set:")
                logger.info(
                    f"PREDICTION: {prob} ({round(prob)}), actual label: {round(label)}"
                    f" INPUT: {parsed_sent} ")

    def predict_viz(self, pred_inputs: List[Dict]) -> List[Tuple]:
        for sample in tqdm.tqdm(pred_inputs, desc=f'Interpreting {len(pred_inputs)} '
                                                  f'predictions and generating per-prediction reports'):
            input_embedding, inputs, probs, token_list, prob = self.pass_interpretable_inputs(sample)
            stmt_pooled_embed = self.inf_session.model.stmt_pooled_embed.squeeze()
            tokens = [self.inf_session.tokenizer._convert_id_to_token(t) for t in token_list]
            token_list = list(filter(lambda l: l not in self.inf_session.special_token_mask, token_list))
            label = sample['labels'].item() if sample['labels'] in [0, 1] else None
            self.inf_session.pred_interpreter.set_reference_embedding(sample['input_ids'])
            debug_baseline = True if self.inf_session.config.inference.debug_baselines else False
            input_meta = (sample['parentid'], sample['childid']) if 'parentid' and 'childid' in sample.keys() else None
            parsed_sent = tokens_to_sentence(self.inf_session, token_list)
            sample_tup = tuple((sample['labels'], sample['attention_mask'],
                                sample['token_type_ids'], sample['position_ids'], None, sample['ctxt_type']))
            core_pred_tup = (tokens, prob, round(prob), label)
            attr_dict = {'internal_batch_size': self.inf_session.config.inference.interpret_batch_size,
                         'num_interpret_steps': self.inf_session.config.inference.num_interpret_steps,
                         'parsed_sent': parsed_sent, 'stmt_embed': stmt_pooled_embed, 'debug_baseline': debug_baseline,
                         'input_meta': input_meta}
            self.inf_session.pred_interpreter.calc_attributions(core_pred_tup, sample_tup, **attr_dict)
        unpublished_reports = self.inf_session.pred_interpreter.gen_viz_report()
        return unpublished_reports

    def pred_exp_viz(self, pred_inputs: List[Dict]) -> List[Tuple]:
        for sample in tqdm.tqdm(pred_inputs, desc=f'Generating statement attributions for {len(pred_inputs)} samples '
                                                  f'from min/max accuracy buckets'):
            input_embedding, inputs, probs, token_list, prob = self.pass_interpretable_inputs(sample)
            stmt_pooled_embed = self.inf_session.model.stmt_pooled_embed.squeeze()
            tokens = [self.inf_session.tokenizer._convert_id_to_token(t) for t in token_list]
            token_list = list(filter(lambda l: l not in self.inf_session.special_token_mask, token_list))
            label = sample['labels'].item() if sample['labels'] in [0, 1] else None
            self.inf_session.pred_interpreter.set_reference_embedding(sample['input_ids'])
            debug_baseline = True if self.inf_session.config.inference.debug_baselines else False
            input_meta = (sample['parentid'], sample['childid']) if 'parentid' and 'childid' in sample.keys() else None
            parsed_sent = tokens_to_sentence(self.inf_session, token_list)
            sample_tup = tuple((sample['labels'], sample['attention_mask'],
                                sample['token_type_ids'], sample['position_ids'], None, sample['ctxt_type']))
            core_pred_tup = (tokens, prob, round(prob), label)
            attr_dict = {'internal_batch_size': self.inf_session.config.inference.interpret_batch_size,
                         'num_interpret_steps': self.inf_session.config.inference.num_interpret_steps,
                         'parsed_sent': parsed_sent, 'stmt_embed': stmt_pooled_embed, 'debug_baseline': debug_baseline,
                         'input_meta': input_meta}
            self.inf_session.pred_interpreter.calc_attributions(core_pred_tup, sample_tup, gen_plot=False, **attr_dict)
        pred_exp_tups = self.inf_session.pred_interpreter.gen_viz_report(pred_exp_mode=True)
        return pred_exp_tups

    def gen_model_rpt(self, pred_inputs: List[Dict]) -> Tuple[List[Tuple], Dict]:
        statement_id = 0
        rpt_tups, sids, stexts, embeds, labels, preds = [], [], [], [], [], []
        for sample in tqdm.tqdm(pred_inputs, desc=f"Generating report using {len(pred_inputs)} samples"):
            input_embedding, inputs, probs, token_list, prob = self.pass_interpretable_inputs(sample)
            token_list = list(filter(lambda l: l not in self.inf_session.special_token_mask, token_list))
            # all records should have a label ("True" unless explicitly labeled false by wapo) unless
            # using "gt, ground truth" version of scoring (model_rpt_all_tweet_data_gt)
            label = sample['labels'].item() if sample['labels'] in [0, 1] else None
            parsed_sent = tokens_to_sentence(self.inf_session, token_list)
            # include only training data in the statement embedding
            if self.inf_session.config.data_source.train_start_date <= sample['sdate'] \
                    <= self.inf_session.config.data_source.train_end_date:
                sids.append(statement_id)
                stexts.append(parsed_sent)
                embeds.append(self.inf_session.model.stmt_pooled_embed.squeeze())
                labels.append(label)
                preds.append(prob)
            rpt_tups.append((constants.APP_INSTANCE, self.inf_session.config.data_source.dsid,
                             self.inf_session.rpt_type, statement_id, parsed_sent, sample['ctxt_type'].item(),
                             sample['sdate'], label, round(prob), prob))
            statement_id += 1
        stmt_embed_dict = {'sids': sids, 'stext': stexts, 'embeds': embeds, 'labels': labels, 'preds': preds}
        return rpt_tups, stmt_embed_dict

    def pass_interpretable_inputs(self, sample: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                               List[int], float]:
        self.inf_session.pred_interpreter.set_input_embedding(sample['input_ids'])
        inputs = {'input_ids': self.inf_session.pred_interpreter.interpret_session.input_embedding,
                  'attention_mask': sample['attention_mask'],
                  'token_type_ids': sample['token_type_ids'],
                  'ctxt_type': sample['ctxt_type'],
                  'labels': sample['labels'],
                  'position_ids': sample['position_ids']}
        probs = self.inf_session.model(**inputs)
        token_list = sample['input_ids'].squeeze(0).tolist()
        prob = round(probs.squeeze(0).item(), 4)
        # noinspection PyTypeChecker
        return self.inf_session.pred_interpreter.interpret_session.input_embedding, inputs, probs, token_list, prob
