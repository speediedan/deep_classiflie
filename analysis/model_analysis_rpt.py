import logging
from pathlib import Path
import datetime
from collections import OrderedDict
from typing import MutableMapping, Optional, List, Dict, Tuple

import torch

import utils.constants as constants
import constants as db_constants
from db_ingest import get_cnxp_handle
from analysis.inference import Inference
from db_utils import fetchallwrapper, batch_execute_many
from analysis.interpretation_utils import load_cache
from analysis.gen_pred_explorer import build_pred_exp_doc
from analysis.gen_perf_explorer import build_perf_exp_doc

logger = logging.getLogger(constants.APP_NAME)


class ModelAnalysisRpt(object):
    """
    Currently supported model analysis report types:
    model_rpt_gt:     report on all ground truth (gt) data chronologically bounded by the available labeling source
                      dates (currently WaPo, start=2017.1.19, end= changes often)
    """

    def __init__(self, config: MutableMapping) -> None:
        self.config = config
        self.cnxp = get_cnxp_handle()
        self.stmt_embed_cache = Path(f'{self.config.experiment.dirs.model_cache_dir}/{constants.STMT_EMBED_CACHE_NAME}')
        self.pred_exp_cache = Path(f'{self.config.experiment.dirs.model_cache_dir}/'
                                          f'{constants.PRED_EXP_CACHE_NAME}')
        self.perf_exp_cache = Path(f'{self.config.experiment.dirs.model_cache_dir}/'
                                          f'{constants.PERF_EXP_CACHE_NAME}')
        self.tweet_model_perf_cache = \
            Path(f'{self.config.experiment.dirs.model_cache_dir}/{constants.TWEET_MODEL_PERF_CACHE_NAME}')
        self.nontweet_model_perf_cache = Path(
            f'{self.config.experiment.dirs.model_cache_dir}/{constants.NONTWEET_MODEL_PERF_CACHE_NAME}')
        self.global_model_perf_cache = Path(
            f'{self.config.experiment.dirs.model_cache_dir}/{constants.GLOBAL_MODEL_PERF_CACHE_NAME}')
        self.report_view = None
        self.init_report_gen()

    def build_bokeh_explorers(self) -> None:
        if self.config.inference.rebuild_pred_explorer:
            if (not self.pred_exp_cache.exists()) or self.config.inference.rebuild_pred_exp_stmt_cache:
                pred_exp_dict, global_metric_summ = self.gen_pred_exp_ds()
                torch.save(tuple((pred_exp_dict, global_metric_summ)), self.pred_exp_cache)
            else:
                pred_exp_dict, global_metric_summ = load_cache(self.pred_exp_cache)
            build_pred_exp_doc(self.config, pred_exp_dict, global_metric_summ, debug_mode=False)
        if self.config.inference.rebuild_perf_explorer:
            if (not self.pred_exp_cache.exists()) or self.config.inference.rebuild_perf_exp_cache:
                perf_exp_dict = self.gen_perf_exp_ds()
                torch.save(perf_exp_dict, self.perf_exp_cache)
            else:
                perf_exp_dict= load_cache(self.perf_exp_cache)
            build_perf_exp_doc(self.config, perf_exp_dict, debug_mode=True)

    def init_report_gen(self) -> None:
        if self.config.inference.rebuild_pred_explorer or self.config.inference.rebuild_perf_explorer:
            self.build_bokeh_explorers()
        elif self.config.inference.update_perf_caches_only:
            self.maybe_build_cache()
        elif self.config.inference.model_report_type != "all":
            self.gen_report(self.config.inference.model_report_type)
        else:
            for rpt_type in constants.REPORT_TYPES:
                self.report_view = rpt_type
                self.gen_report(self.report_view)

    def gen_pred_exp_ds(self) -> Tuple[Dict, Tuple]:
        pred_exp_tups = fetchallwrapper(self.cnxp.get_connection(), self.config.inference.sql.pred_exp_sql)
        pred_exp_set = []
        pred_exp_ds = OrderedDict({'bucket_type': [], 'bucket_acc': [], 'conf_percentile': [], 'pos_pred_acc': [],
                                  'neg_pred_acc': [], 'pos_pred_ratio': [], 'neg_pred_ratio': [], 'statement_id': [],
                                  'statement_text': [], 'tp': [], 'tn': [], 'fp': [], 'fn': []})
        for (bucket_type, bucket_acc, conf_percentile, pos_pred_acc, neg_pred_acc, pos_pred_ratio, neg_pred_ratio,
             statement_id, statement_text, ctxt_type, tp, tn, fp, fn) in pred_exp_tups:
            label = 'False' if tp == 1 or fn == 1 else 'True'
            pred_exp_set.append((statement_text, ctxt_type, label))
            for k, v in zip(list(pred_exp_ds.keys()), [bucket_type, bucket_acc, conf_percentile, pos_pred_acc,
                                                      neg_pred_acc, pos_pred_ratio, neg_pred_ratio, statement_id,
                                                      statement_text, tp, tn, fp, fn]):
                pred_exp_ds[k].append(v)
        pred_exp_attr_tups, global_metric_summ = Inference(self.config, pred_exp_set=pred_exp_set).init_predict()
        pred_exp_ds['pred_exp_attr_tups'] = pred_exp_attr_tups
        return pred_exp_ds, global_metric_summ

    def gen_perf_exp_ds(self) -> Dict:
        perf_exp_dict = {}
        for cmatrix_rpt_type in [*db_constants.TEST_CMATRICES, *db_constants.TEST_CONF_CMATRICES]:
            perf_exp_dict[cmatrix_rpt_type] = fetchallwrapper(self.cnxp.get_connection(),
                                                              f"select * from {cmatrix_rpt_type}")
        return perf_exp_dict

    def gen_report(self, rpt_type: str) -> None:
        analysis_set = self.gen_analysis_set()
        ds_meta = fetchallwrapper(self.cnxp.get_connection(), self.config.inference.sql.ds_md_sql)[0]
        self.config.data_source.dsid = ds_meta[0]
        self.config.data_source.train_start_date = datetime.datetime.combine(ds_meta[1], datetime.time())
        self.config.data_source.train_end_date = datetime.datetime.combine(ds_meta[2], datetime.time())
        rpt_tups, stmt_embed_dict = Inference(self.config, analysis_set=analysis_set, rpt_type=rpt_type).init_predict()
        inserted_rowcnt, error_rows = batch_execute_many(self.cnxp.get_connection(),
                                                         self.config.inference.sql.save_model_sql, rpt_tups)
        logger.info(f"Generated {inserted_rowcnt} inference records for analysis of "
                    f"model version {constants.APP_INSTANCE}")
        self.maybe_build_cache(stmt_embed_dict)

    def gen_analysis_set(self) -> List[Tuple]:
        # current use case involves relatively small analysis set that fits in memory and should only be used once
        # so wasteful to persist. if later use cases necessitate, will pickle or persist for larger datasets
        report_sql = f"select * from {self.report_view}"
        # TODO: remove this unnecessary transformation? should be able to directly return report_sql tuple list now...
        analysis_set = ModelAnalysisRpt.prep_model_analysis_ds(fetchallwrapper(self.cnxp.get_connection(), report_sql))
        return analysis_set

    def maybe_build_cache(self, stmt_embed_dict: Optional[Dict] = None) -> None:
        if ((not self.stmt_embed_cache.exists()) or self.config.inference.rebuild_stmt_cache) \
                and not self.config.inference.update_perf_caches_only:
            torch.save(stmt_embed_dict, self.stmt_embed_cache)
            logger.info(f"Cached {len(stmt_embed_dict['sids'])} statement embeddings for analysis:"
                        f" {self.stmt_embed_cache}")
        if ((not self.tweet_model_perf_cache.exists()) or (not self.nontweet_model_perf_cache.exists())
                or self.config.inference.rebuild_perf_cache):
            tweet_model_perf_tups = fetchallwrapper(self.cnxp.get_connection(),
                                                    self.config.inference.sql.tweet_model_perf_cache_sql)
            torch.save(tweet_model_perf_tups, self.tweet_model_perf_cache)
            logger.info(f"Cached model test accuracy into {len(tweet_model_perf_tups)} buckets for tweet model "
                        f"reporting: {self.tweet_model_perf_cache}")
            nontweet_model_perf_tups = fetchallwrapper(self.cnxp.get_connection(),
                                                       self.config.inference.sql.nontweet_model_perf_cache_sql)
            torch.save(nontweet_model_perf_tups, self.nontweet_model_perf_cache)
            logger.info(f"Cached model test accuracy into {len(nontweet_model_perf_tups)} buckets for nontweet model "
                        f"reporting: {self.nontweet_model_perf_cache}")
            self.refresh_global_cache()  # refresh global cache after updating

    def refresh_global_cache(self):
        global_model_perf_tups = fetchallwrapper(self.cnxp.get_connection(),
                                                 self.config.inference.sql.global_model_perf_cache_sql)
        torch.save(global_model_perf_tups, self.global_model_perf_cache)
        logger.info(f"(Re)Built global model accuracy cache {self.global_model_perf_cache}")

    # TODO: remove this unnecessary transformation?
    @staticmethod
    def prep_model_analysis_ds(example_tups: List[Tuple]) -> List[Tuple]:
        analysis_set = []
        for (sdate, text, ctxt_type, label) in example_tups:
            analysis_set.append((sdate, text, ctxt_type, label))
        return analysis_set
