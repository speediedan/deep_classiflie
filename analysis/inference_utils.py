import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, MutableMapping, Union

import torch
from transformers import AlbertTokenizer
import tqdm

import utils.constants as constants
from analysis.interpretation import InterpretTransformer
from utils.core_utils import device_config

logger = logging.getLogger(constants.APP_NAME)


@dataclass
class InferenceSession:
    config: MutableMapping
    mapping_set: List[Tuple] = None
    analysis_set: List[Union[Tuple, List]] = None
    pred_exp_set: List[Tuple] = None
    rpt_type: str = None
    base_mode: bool = True
    # don't initialize model or tokenizer unless necessary, i.e. used outside training context
    model: torch.nn.Module = None
    tokenizer: AlbertTokenizer = None
    pred_interpreter: InterpretTransformer = None
    special_token_mask: List[int] = None

    def __post_init__(self) -> None:
        self.device = device_config(self.config)
        self.pred_inputs = self.config.experiment.pred_inputs or None
        self.regex_dict = {'punc_fix': re.compile(r"(\s(?=[,’':.\-?]))|((?<=[$:’\-'])\s)"),
                           'num_fix': re.compile(r"(?<=\d[.,])\s"),
                           'quotes_smart': re.compile(r"[“”]\s([^\"]*)\s[“”]"),
                           'quotes_reg': re.compile(r"\"\s([^\"]*)\s\"")
                           }


def tokens_to_sentence(inf_session: InferenceSession, token_list: List) -> str:
    tokens = [inf_session.tokenizer._convert_id_to_token(t) for t in token_list]
    raw_sent = inf_session.tokenizer.convert_tokens_to_string(tokens)
    raw_sent = inf_session.regex_dict['punc_fix'].sub('', raw_sent)
    raw_sent = inf_session.regex_dict['num_fix'].sub('', raw_sent)
    raw_sent = inf_session.regex_dict['quotes_smart'].sub(r'"\1"', raw_sent)
    parsed_sent = inf_session.regex_dict['quotes_reg'].sub(r'"\1"', raw_sent)
    return parsed_sent


def gen_embed_mappings(inf_session: InferenceSession, pred_inputs: List[Dict]) -> Dict:
    sids, stexts, embeds = [], [], []
    for sample in tqdm.tqdm(pred_inputs, desc=f"Generating mapping of {len(pred_inputs)} statement embeddings "):
        inf_session.pred_interpreter.set_input_embedding(sample['input_ids'])
        if not inf_session.base_mode:
            inputs = {'input_ids': inf_session.pred_interpreter.interpret_session.input_embedding,
                      'attention_mask': sample['attention_mask'],
                      'token_type_ids': sample['token_type_ids'],
                      'ctxt_type': sample['ctxt_type'],
                      'labels': sample['labels'],
                      'position_ids': sample['position_ids']}
            _ = inf_session.model(**inputs)
            stmt_pooled_embed = inf_session.model.stmt_pooled_embed.squeeze()
            sids.append(sample['id'])
            embeds.append(stmt_pooled_embed)
        else:
            sids.append(sample['id'])
            embeds.append(torch.mean(inf_session.pred_interpreter.interpret_session.input_embedding,[1]).squeeze())
    sid_embed_mapping = {'sids': sids, 'embeds': embeds}
    return sid_embed_mapping


def pred_inputs_from_test(inf_session: InferenceSession, eval_tuple: Tuple = None, num_samples: int = None) \
        -> List[Dict]:
    if eval_tuple and num_samples:
        eval_dataset, eval_sampler, eval_dataloader = eval_tuple
        pred_inputs = []
        for i, sample in enumerate(eval_dataloader):
            if i < num_samples:
                sample = tuple(t.to(inf_session.device) for t in sample)
                seq_length = sample[0].size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=inf_session.device)
                position_ids = position_ids.unsqueeze(0).expand_as(sample[0])
                pred_inputs.append({'input_ids': sample[0],
                                    'attention_mask': sample[1],
                                    'token_type_ids': sample[2],
                                    'ctxt_type': sample[3],
                                    'labels': sample[4],
                                    'position_ids': position_ids})
            else:
                break
    else:
        raise ValueError("an evaluation tuple (dataset, sampler, dataloader) "
                         "and num_samples required to gen pred_inputs")
    return pred_inputs


def prep_rpt_tups(inf_session: InferenceSession) -> List[Dict]:
    pred_inputs = []
    class_labels = [k for k in inf_session.config.data_source.class_labels]
    label_map = {label: i for i, label in enumerate(class_labels)}
    for i, (sdate, ex, ctxt_type, label) in tqdm.tqdm(enumerate(inf_session.analysis_set),
                                                      desc=f"preparing {len(inf_session.analysis_set)} "
                                                           f"samples for inference"):
        label = label_map[label]
        input_ids, attention_mask, token_type_ids, position_ids = prep_model_inputs(inf_session, ex)
        pred_input = {'input_ids': torch.tensor(input_ids).to(inf_session.device).unsqueeze(0),
                      'attention_mask': torch.tensor(attention_mask).to(inf_session.device).unsqueeze(0),
                      'token_type_ids': torch.tensor(token_type_ids).to(inf_session.device).unsqueeze(0),
                      'labels': torch.tensor(label).to(inf_session.device).unsqueeze(0).unsqueeze(
                          0) if label in [0, 1] else None,
                      'position_ids': torch.tensor(position_ids).to(inf_session.device).unsqueeze(0),
                      'ctxt_type': torch.tensor(ctxt_type).to(inf_session.device).unsqueeze(0),
                      'sdate': sdate}
        pred_inputs.append(pred_input)
    return pred_inputs


def prep_pred_exp_tups(inf_session: InferenceSession) -> List[Dict]:
    pred_inputs = []
    class_labels = [k for k in inf_session.config.data_source.class_labels]
    label_map = {label: i for i, label in enumerate(class_labels)}
    for i, (ex, ctxt_type, label) in tqdm.tqdm(enumerate(inf_session.pred_exp_set),
                                                      desc=f"preparing {len(inf_session.pred_exp_set)} "
                                                           f"samples for inference"):
        label = label_map[label]
        input_ids, attention_mask, token_type_ids, position_ids = prep_model_inputs(inf_session, ex)
        pred_input = {'input_ids': torch.tensor(input_ids).to(inf_session.device).unsqueeze(0),
                      'attention_mask': torch.tensor(attention_mask).to(inf_session.device).unsqueeze(0),
                      'token_type_ids': torch.tensor(token_type_ids).to(inf_session.device).unsqueeze(0),
                      'labels': torch.tensor(label).to(inf_session.device).unsqueeze(0).unsqueeze(
                          0) if label in [0, 1] else None,
                      'position_ids': torch.tensor(position_ids).to(inf_session.device).unsqueeze(0),
                      'ctxt_type': torch.tensor(ctxt_type).to(inf_session.device).unsqueeze(0)}
        pred_inputs.append(pred_input)
    return pred_inputs


def pred_inputs_from_config(inf_session: InferenceSession) -> List[Dict]:
    pred_inputs = []
    class_labels = [k for k in inf_session.config.data_source.class_labels]
    label_map = {label: i for i, label in enumerate(class_labels)}
    for i, (parentid, childid, ex, ctxt_type, label) in enumerate(inf_session.config.inference.pred_inputs):
        label = None if label == "" else label_map[label]
        input_ids, attention_mask, token_type_ids, position_ids = prep_model_inputs(inf_session, ex)
        pred_input = {'input_ids': torch.tensor(input_ids).to(inf_session.device).unsqueeze(0),
                      'attention_mask': torch.tensor(attention_mask).to(inf_session.device).unsqueeze(0),
                      'token_type_ids': torch.tensor(token_type_ids).to(inf_session.device).unsqueeze(0),
                      'labels': torch.tensor(label).to(inf_session.device).unsqueeze(0) if label else None,
                      'position_ids': torch.tensor(position_ids).to(inf_session.device).unsqueeze(0),
                      'ctxt_type': torch.tensor(ctxt_type).to(inf_session.device).unsqueeze(0),
                      'parentid': parentid,
                      'childid': childid}
        pred_inputs.append(pred_input)
    return pred_inputs


def prep_base_mapping_tups(inf_session: InferenceSession) -> List[Dict]:
    pred_inputs = []
    for i, (tup_id, ex, _) in tqdm.tqdm(enumerate(inf_session.mapping_set),
                                                desc=f"preparing {len(inf_session.mapping_set)} samples for distance-based filtering"):
        input_ids = inf_session.tokenizer.encode_plus(ex, max_length=inf_session.config.data_source.max_seq_length,
                                                       add_special_tokens=False, truncation=True)['input_ids']
        pred_input = {'input_ids': torch.tensor(input_ids).to(inf_session.device).unsqueeze(0), 'id': tup_id}
        pred_inputs.append(pred_input)
    return pred_inputs


def prep_mapping_tups(inf_session: InferenceSession) -> List[Dict]:
    pred_inputs = []
    for i, (tup_id, ex, ctxt_type) in tqdm.tqdm(enumerate(inf_session.mapping_set),
                                                desc=f"preparing {len(inf_session.mapping_set)} samples for inference"):
        input_ids, attention_mask, token_type_ids, position_ids = prep_model_inputs(inf_session, ex)
        pred_input = {'input_ids': torch.tensor(input_ids).to(inf_session.device).unsqueeze(0),
                      'attention_mask': torch.tensor(attention_mask).to(inf_session.device).unsqueeze(0),
                      'token_type_ids': torch.tensor(token_type_ids).to(inf_session.device).unsqueeze(0),
                      'labels': None,
                      'position_ids': torch.tensor(position_ids).to(inf_session.device).unsqueeze(0),
                      'ctxt_type': torch.tensor(ctxt_type).to(inf_session.device).unsqueeze(0),
                      #TODO: change to tup_id
                      'id': tup_id}
        pred_inputs.append(pred_input)
    return pred_inputs


# noinspection PyTypeChecker,PyUnresolvedReferences
def prep_model_inputs(inf_session: InferenceSession, ex: str) -> Tuple[List[int], List[int], List[int], List[int]]:
    inputs = inf_session.tokenizer.encode_plus(ex, max_length=inf_session.config.data_source.max_seq_length,
                                               add_special_tokens=True, truncation=True)
    input_ids, token_type_ids = inputs['input_ids'], inputs['token_type_ids']
    attention_mask = [1] * len(input_ids)
    padding_length = inf_session.config.data_source.max_seq_length - len(input_ids)
    input_ids = input_ids + ([inf_session.tokenizer.pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    # N.B. this would need to change for XLnet as pad_segment_id is 4 instead of 0
    token_type_ids = token_type_ids + ([0] * padding_length)
    position_ids = ([pos for pos in range(inf_session.config.data_source.max_seq_length)])
    return input_ids, attention_mask, token_type_ids, position_ids
