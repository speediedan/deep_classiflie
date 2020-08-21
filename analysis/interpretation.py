import datetime
import logging
from typing import List, Tuple, MutableMapping, Union
import re

import torch
from captum.attr import IntegratedGradients
from transformers import AlbertTokenizer
from captum.attr._utils.visualization import VisualizationDataRecord

import utils.constants as constants
from analysis.captum_cust_viz import gen_per_pred_reports, pred_exp_attr
from analysis.trumap_spectrum import TrumapSpectrum
from analysis.interpretation_utils import calc_accuracy_data, prep_viz_inputs, calc_ext_records,\
    assemble_viz_record, purge_intermediate_files, prep_trumap_inputs
from analysis.interpretation_utils import InterpretationSession


logger = logging.getLogger(constants.APP_NAME)
try:
    # noinspection PyUnresolvedReferences
    import cudf
    # noinspection PyUnresolvedReferences
    import cuml
except ImportError:
    logger.warning("umap implementation currently requires a cuda-capable gpu and cudf/cuml, "
                   "disabling interpretation...")


class InterpretTransformer(object):
    """
    Class using slightly modified captum functions to generate interpretation reports for various transformer
    model-based (initially ALBERT) predictions
    """
    def __init__(self, config: MutableMapping, model: torch.nn.Module, tokenizer: AlbertTokenizer,
                 device: torch.device, pred_report_path: str) -> None:
        self.interpret_session = InterpretationSession(config, model, tokenizer, device, pred_report_path)

    def set_input_embedding(self, example: torch.Tensor) -> None:
        self.interpret_session.input_embedding = \
            self.interpret_session.interpretable_embedding.indices_to_embeddings(example)

    def set_reference_embedding(self, example_tokens: torch.Tensor) -> None:
        reference_tokens = example_tokens.clone().squeeze(0).tolist()
        pad_func = lambda x: self.interpret_session.tokenizer.pad_token_id if x not in [
            self.interpret_session.tokenizer.cls_token_id,
            self.interpret_session.tokenizer.sep_token_id] else x
        reference_tokens = list(map(pad_func, reference_tokens))
        self.interpret_session.baseline_text = [self.interpret_session.tokenizer._convert_id_to_token(t) for t in
                                                reference_tokens]
        reference_tokens = torch.tensor(reference_tokens).unsqueeze(0).to(self.interpret_session.device)
        self.interpret_session.reference_embedding = \
            self.interpret_session.interpretable_embedding.indices_to_embeddings(reference_tokens)

    def calc_attributions(self, core_pred_tup: Tuple, addnl_forward_args: Tuple, internal_batch_size: int = None,
                          num_interpret_steps: int = 20, parsed_sent: str = None, stmt_embed: torch.Tensor = None,
                          debug_baseline: bool = False, input_meta: Tuple = None, **kwargs) -> None:
        example, pred, pred_ind, label = core_pred_tup
        if debug_baseline:
            input_embedding = self.interpret_session.reference_embedding
            input_example = self.interpret_session.baseline_text
        else:
            input_embedding = self.interpret_session.input_embedding
            input_example = example
        # compute attributions and approximation delta using integrated gradients
        attribution_cfg = {'inputs': input_embedding, 'baselines': self.interpret_session.reference_embedding,
                           'n_steps': num_interpret_steps, 'return_convergence_delta': True,
                           'additional_forward_args': addnl_forward_args, 'internal_batch_size': internal_batch_size}
        attributions_ig, delta = IntegratedGradients(self.interpret_session.model).attribute(**attribution_cfg)
        visualizer_cfg = {'attributions': attributions_ig, 'text': input_example, 'parsed_sent': parsed_sent,
                          'pred': pred, 'pred_ind': pred_ind, 'label': label, 'delta': delta, 'stmt_embed': stmt_embed,
                          'input_meta': input_meta}
        self.add_attributions_to_visualizer(**visualizer_cfg, **kwargs)

    def add_attributions_to_visualizer(self, attributions: torch.Tensor, text: List[str], parsed_sent: str, pred: float,
                                       pred_ind: int, label: int, delta: torch.Tensor, stmt_embed: torch.Tensor = None,
                                       input_meta: Tuple = None, gen_plot: bool = True) -> None:
        if input_meta:
            (parentid, childid) = input_meta
        else:
            parentid, childid = None, None
        prep_viz_input = (self.interpret_session, attributions, text, pred_ind, parsed_sent, pred, stmt_embed)
        viz_inputs, max_token_attr_idx, token_level_attributions, attributions = prep_viz_inputs(*prep_viz_input)
        if gen_plot:
            plt_path, sim_spectrum = self.gen_trumap_viz(**viz_inputs)
            self.interpret_session.ss_image_paths.append(plt_path)
        else:
            plt_path, sim_spectrum = None, None
        raw_confidence = pred if pred >= 0.5 else 1 - pred
        nondigit_check = re.compile(r"\D")
        is_stmt = True if nondigit_check.search(str(parentid)) or not parentid else False
        accuracy_tup = calc_accuracy_data(self.interpret_session.model_perf_tups, raw_confidence, is_stmt)
        ext_tup = (sim_spectrum, accuracy_tup, parentid, childid, self.interpret_session.model_perf_tups['global'][0],
                   is_stmt)
        ext_record = calc_ext_records(token_level_attributions, max_token_attr_idx, attributions, text, ext_tup)
        self.interpret_session.ext_vis_data_records.append(ext_record)
        viz_record_inputs = assemble_viz_record(attributions, delta, pred, pred_ind, label, text)
        self.interpret_session.vis_data_records.append(VisualizationDataRecord(**viz_record_inputs))

    def gen_viz_report(self, pred_exp_mode: bool = False) -> Union[List[Tuple], Tuple[List[Tuple], Tuple]]:
        special_tokens = [self.interpret_session.tokenizer._convert_id_to_token(tok_id) for tok_id in
                          self.interpret_session.special_token_mask]
        stylesheet_path = f"{self.interpret_session.config.inference.asset_dir}/detailed_report.css"
        logo_path = f"{self.interpret_session.config.inference.asset_dir}/dc_logo.png"
        paths_tup = (
            self.interpret_session.pred_report_path, stylesheet_path, logo_path,
            self.interpret_session.config.experiment.dirs.inference_output_dir)
        per_pred_inputs = {'datarecords': self.interpret_session.vis_data_records,
                           'ext_recs': self.interpret_session.ext_vis_data_records,
                           'ss_images': self.interpret_session.ss_image_paths, 'paths_tup': paths_tup,
                           'token_mask': special_tokens, 'invert_colors': True,
                           'pub_thresholds': (self.interpret_session.config.inference.tweet_pub_conf_threshold,
                                              self.interpret_session.config.inference.nontweet_pub_conf_threshold)
                           }
        if not pred_exp_mode:
            unpublished_reports = gen_per_pred_reports(**per_pred_inputs)
            if self.interpret_session.config.inference.purge_intermediate_rpt_files:
                purge_intermediate_files(unpublished_reports)
            return unpublished_reports
        else:
            unpublished_reports, global_summ = pred_exp_attr(**per_pred_inputs)
            return unpublished_reports, global_summ

    def gen_trumap_viz(self, target_text: str, target_pred_ind: int, stmt_pred: float, stmt_embed: torch.Tensor,
                       stmt_pred_matrix: torch.Tensor, stmt_embed_matrix: torch.Tensor) -> Tuple[str, List[Tuple]]:
        embed_compare = stmt_embed.expand_as(stmt_embed_matrix)
        l2_norm_distance = torch.nn.PairwiseDistance(p=2)
        embed_sims = (l2_norm_distance(stmt_embed_matrix, embed_compare))
        embed_sims = embed_sims / torch.norm(embed_sims)
        stmt_pred = torch.tensor(stmt_pred).to(device=self.interpret_session.device)
        stmt_compare = stmt_pred.expand_as(stmt_pred_matrix)
        pred_sims = torch.abs_(stmt_pred_matrix - stmt_compare)
        _, sid_idx = torch.topk((embed_sims + pred_sims), k=self.interpret_session.config.inference.trumap_topk,
                                largest=False)
        topk_sim_full_embeds = torch.cat([stmt_embed_matrix[sid_idx], stmt_embed.unsqueeze(0),
                                          stmt_embed_matrix[self.interpret_session.max_pred_idx].unsqueeze(0),
                                          stmt_embed_matrix[self.interpret_session.min_pred_idx].unsqueeze(0)]).tolist()
        trumap_spectrum, target_token_tup, umap_bounds_tup = \
            prep_trumap_inputs(self.interpret_session, sid_idx, topk_sim_full_embeds, target_text, target_pred_ind)
        ss_image_path = self.plot_trumap_spectrum(trumap_spectrum, target_token_tup, umap_bounds_tup)
        return ss_image_path, trumap_spectrum

    def plot_trumap_spectrum(self, spectrum: List[Tuple], target_token_tup: Tuple, umap_bounds_tup: Tuple) -> str:
        labels, vals = [], []
        for i in spectrum:
            labels.append(i[1])
            vals.append(i[2])
        ts_image_path = \
            f'{self.interpret_session.config.experiment.dirs.inference_output_dir}/{datetime.datetime.now()}.png'
        word_dim_data = (vals, labels, target_token_tup, ts_image_path, umap_bounds_tup)
        TrumapSpectrum(word_dim_data)
        return ts_image_path
