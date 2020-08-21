import re
from dataclasses import dataclass
import os
import logging
import bisect
from typing import MutableMapping, Any, List, Dict, Tuple
from collections import OrderedDict

import numpy as np
import torch
from transformers import AlbertTokenizer
from captum.attr import configure_interpretable_embedding_layer

import utils.constants as constants

logger = logging.getLogger(constants.APP_NAME)

try:
    # noinspection PyUnresolvedReferences
    import cudf
    # noinspection PyUnresolvedReferences
    import cuml
except ImportError:
    logger.warning("umap implementation currently requires a cuda-capable gpu and cudf/cuml, "
                   "disabling interpretation...")


@dataclass
class InterpretationSession:
    config: MutableMapping
    model: torch.nn.Module
    tokenizer: AlbertTokenizer
    device: torch.device
    pred_report_path: str
    input_embedding: torch.Tensor = None
    reference_embedding: torch.Tensor = None
    baseline_text: List[str] = None

    def __post_init__(self) -> None:
        self.special_token_mask = [self.tokenizer.unk_token_id, self.tokenizer.sep_token_id,
                                   self.tokenizer.pad_token_id, self.tokenizer.cls_token_id]
        self.candidate_token_mask, self.candidate_token_map = gen_candidate_mask(self.tokenizer)
        self.interpretable_embedding = configure_interpretable_embedding_layer(self.model,
                                                                               'albert.embeddings.word_embeddings')
        self.model_perf_tups = {'tweets': load_cache(f'{self.config.experiment.dirs.model_cache_dir}/'
                                                     f'{constants.TWEET_MODEL_PERF_CACHE_NAME}'),
                                'nontweets': load_cache(f'{self.config.experiment.dirs.model_cache_dir}/'
                                                        f'{constants.NONTWEET_MODEL_PERF_CACHE_NAME}'),
                                'global': load_cache(f'{self.config.experiment.dirs.model_cache_dir}/'
                                                     f'{constants.GLOBAL_MODEL_PERF_CACHE_NAME}')}
        self.stmt_embed_dict = load_cache(f'{self.config.experiment.dirs.model_cache_dir}/'
                                          f'{constants.STMT_EMBED_CACHE_NAME}')
        if self.stmt_embed_dict:
            self.max_pred_idx = torch.tensor(np.argmax(np.asarray(self.stmt_embed_dict['preds'])))
            self.min_pred_idx = torch.tensor(np.argmin(np.asarray(self.stmt_embed_dict['preds'])))
        self.vis_data_records, self.ext_vis_data_records, self.ss_image_paths = [], [], []


def calc_accuracy_data(model_perf_tups: Dict, raw_confidence: float, is_stmt: bool) -> Tuple:
    cache_type = 'nontweets' if is_stmt else 'tweets'
    cache_idx = bisect.bisect_right(model_perf_tups[cache_type], (raw_confidence,))
    tmp_tup = model_perf_tups[cache_type][-1] if cache_idx >= len(model_perf_tups[cache_type]) else \
        model_perf_tups[cache_type][cache_idx]
    accuracy_tup = (tmp_tup[1:])
    return accuracy_tup


def gen_candidate_mask(tokenizer: AlbertTokenizer) -> Tuple[List, OrderedDict]:
    all_sp_vocab = OrderedDict({k: tokenizer._convert_id_to_token(k) for k
                                in range(0, tokenizer.vocab_size - 1)})
    cand_pos_filter = re.compile(r"^(▁|\w){4,}")
    cand_neg_filter = re.compile(r"[\d]+")
    candidate_token_mask = sorted([k for (k, v) in all_sp_vocab.items() if cand_pos_filter.search(v)
                                   and not cand_neg_filter.search(v)])
    candidate_token_map = OrderedDict()
    idx = 0
    for (k, v) in all_sp_vocab.items():
        if k in candidate_token_mask:
            candidate_token_map[idx] = k
            idx += 1
    return candidate_token_mask, candidate_token_map


def save_vocab(vocabd: Dict, filename: str) -> None:
    with open(filename, 'w') as file:
        for k, v in vocabd.items():
            file.write(f"{k}: {v} \n")


def load_cache(path: str) -> Any:
    loaded_obj = None
    try:
        with open(path, "rb") as fp:
            loaded_obj = torch.load(fp)
    except FileNotFoundError:
        logger.debug(f'No cached obj file found at {path}, proceeding w/o the cache may result in an error '
                     f'for some functions')
    return loaded_obj


def purge_intermediate_files(unpub_rpts: List[Tuple]) -> None:
    for (_, _, _, _, _, ss_image, pred_html) in unpub_rpts:
        os.remove(ss_image)
        os.remove(pred_html)


def prep_viz_inputs(interpret_session: InterpretationSession, attributions: torch.Tensor, text: List[int],
                    pred_ind: int, parsed_sent: str, pred: float,
                    stmt_embed: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    stmt_pred_matrix = torch.tensor(interpret_session.stmt_embed_dict['preds']).to(device=interpret_session.device)
    stmt_embed_matrix = torch.stack(interpret_session.stmt_embed_dict['embeds']).to(device=interpret_session.device)
    attributions = attributions.sum(dim=2).squeeze(0)
    token_level_attributions = attributions.clone()
    # this should prevent non-"interpretability friendly" tokens from being chosen for detailed visualization,
    # 0 class should always negative attributions, 1 class should always have attributions > 0
    for i, t in enumerate(text):
        if interpret_session.tokenizer._convert_token_to_id(text[i]) not in interpret_session.candidate_token_mask:
            token_level_attributions[i] = 0.0
    max_attr_func = torch.argmax if pred_ind == 1 else torch.argmin
    max_token_attr_idx = max_attr_func(token_level_attributions, dim=0)
    viz_inputs = {'target_text': parsed_sent, 'target_pred_ind': pred_ind, 'stmt_pred': pred, 'stmt_embed': stmt_embed,
                  'stmt_pred_matrix': stmt_pred_matrix, 'stmt_embed_matrix': stmt_embed_matrix}
    return viz_inputs, max_token_attr_idx, token_level_attributions, attributions


def prep_trumap_inputs(interpret_session: InterpretationSession, sid_idx: torch.Tensor, topk_sim_full_embeds: List,
                       target_text: str, target_pred_ind: int) -> Tuple[List[Tuple], Tuple, Tuple]:
    full_embed_df = cudf.DataFrame([tuple(k) for k in topk_sim_full_embeds])
    umap_xformed_embeds = cuml.UMAP(n_neighbors=5, n_components=3, n_epochs=500, min_dist=0.1).fit_transform(
        full_embed_df).as_matrix()
    umap_normed_embeds = (umap_xformed_embeds / np.linalg.norm(umap_xformed_embeds)).tolist()
    max_truth_umap, max_falsehood_umap, target_sent_umap = [umap_normed_embeds.pop() for _ in range(3)]
    trumap_spectrum = []
    for idx, umapval in zip(sid_idx.tolist(), umap_normed_embeds):
        trumap_spectrum.append((interpret_session.stmt_embed_dict['stext'][idx],
                                interpret_session.stmt_embed_dict['labels'][idx], tuple(umapval)))
    target_token_tup = (target_text, target_pred_ind, tuple(target_sent_umap))
    umap_bounds_tup = (tuple(max_falsehood_umap), tuple(max_truth_umap))
    return trumap_spectrum, target_token_tup, umap_bounds_tup


def calc_ext_records(token_level_attributions: torch.Tensor, max_token_attr_idx: torch.Tensor,
                     attributions: torch.Tensor, text: List[str], ext_tup: Tuple) -> Tuple:
    max_token_level_attr = token_level_attributions[max_token_attr_idx].item()
    max_token_level_attr_norm = (token_level_attributions[max_token_attr_idx] / torch.norm(attributions)).item()
    total_attribution = attributions.detach().cpu().numpy().sum().item()
    max_attr_token = text[max_token_attr_idx.item()].replace('▁', '')
    ext_record_tup = (max_token_level_attr, max_token_level_attr_norm, total_attribution, max_attr_token, *ext_tup)
    return ext_record_tup


def assemble_viz_record(attributions: torch.Tensor, delta: torch.Tensor, pred: float, pred_ind: int, label: int,
                        text: List[str]) -> Dict:
    normed_attributions = attributions / torch.norm(attributions)
    normed_attributions = normed_attributions.detach().cpu().numpy()
    delta = delta.detach().cpu().numpy()
    viz_dict = {'word_attributions': normed_attributions, 'pred_prob': pred, 'pred_class': pred_ind,
                'true_class': label, 'attr_class': pred_ind,
                'attr_score': normed_attributions.sum().item(),
                'raw_input': text, 'convergence_score': delta}
    return viz_dict
