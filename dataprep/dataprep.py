import datetime
import logging
import sys
from pathlib import Path
from typing import MutableMapping, Iterator, List, Dict, Tuple

import numpy as np
import torch
import tqdm
from scipy.special import softmax
from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler, TensorDataset
from transformers import AlbertTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features

import dataprep.dataprep_utils
import utils.constants as constants
import utils.core_utils
from analysis.inference import Inference
from analysis.interpretation_utils import load_cache

logger = logging.getLogger(constants.APP_NAME)

try:
    # noinspection PyUnresolvedReferences
    from db_ingest import refresh_db, get_cnxp_handle
    # noinspection PyUnresolvedReferences
    import db_utils
except ImportError as error:
    logger.debug(f"{error.__class__.__name__}: {constants.DEF_DB_PRJ_NAME} modules not found, "
                 f"db functionality will not be available")


class DatasetCollection(object):
    def __init__(self, config: MutableMapping) -> None:
        self.config = config
        self.dataset_conf_id, self.target_ds_structure, self.ds_meta = None, None, None
        if self.config.experiment.db_functionality_enabled:
            self.cnxp = get_cnxp_handle()
            if not self.config.data_source.skip_db_refresh:
                refresh_db(self.config.data_source.db_conf, self.cnxp)
        self.device = utils.core_utils.device_config(self.config)
        self.dataset_conf = self.init_dataset_conf()
        self.base_falsehood_embed_cache = Path(f'{self.config.experiment.dirs.model_cache_dir}/'
                                               f'{constants.BASE_FALSEHOOD_EMBED_CACHE_NAME}')
        self.base_truth_embed_cache = Path(f'{self.config.experiment.dirs.model_cache_dir}/'
                                           f'{constants.BASE_TRUTH_EMBED_CACHE_NAME}')
        self.falsehood_embed_cache = Path(f'{self.config.experiment.dirs.model_cache_dir}/'
                                          f'{constants.FALSEHOOD_EMBED_CACHE_NAME}')
        self.truth_embed_cache = Path(f'{self.config.experiment.dirs.model_cache_dir}/'
                                      f'{constants.TRUTH_EMBED_CACHE_NAME}')
        self.file_suffix = self.config.data_source.file_suffix
        if self.config.data_source.update_ds_db_metadata_only and self.config.experiment.db_functionality_enabled:
            self.update_ds_db_metadata()
        else:
            # build dataset from DB if a cached file version doesn't already exist
            self.load_datasets()

    def init_dataset_conf(self) -> Dict:
        # define dataset configuration dict
        dataset_conf = {}
        (dataset_conf['num_train_recs'], dataset_conf['num_val_recs'], dataset_conf['num_test_recs']) = (0, 0, 0)
        # multiple ds structures can be configured, currently defaulting to a single one
        self.target_ds_structure = 'primary_ds_structure'
        dataset_conf[self.target_ds_structure] = {
            "train": self.config.data_source[self.target_ds_structure]['train_ratio'],
            "val": self.config.data_source[self.target_ds_structure]['val_ratio'],
            "test": self.config.data_source[self.target_ds_structure]['test_ratio']}
        if not dataprep.dataprep_utils.validate_normalized(dataset_conf[self.target_ds_structure]):
            logger.error(f"invalid values specified {dataset_conf[self.target_ds_structure]} (sum must == 1.0). "
                         f"Please reconfigure and restart")
            sys.exit(0)
        dataset_conf['class_weights'] = {}
        dataset_conf['train_batch_size'] = self.config.data_source.train_batch_size
        dataset_conf['val_batch_size'] = self.config.data_source.val_batch_size
        dataset_conf['test_batch_size'] = self.config.data_source.test_batch_size
        return dataset_conf

    def load_datasets(self) -> None:
        self.dataset_build_file_cache()
        for ds, _ in self.dataset_conf[self.target_ds_structure].items():
            self.dataset_conf_id = ds + '_ds'
            self.dataset_conf[self.dataset_conf_id] = self.build_dataset(ds)

    def dataset_build_file_cache(self) -> None:
        self.ds_meta = Path(f"{self.config.experiment.dirs.tmp_data_dir}/ds_meta{self.file_suffix}.json")
        self.dataset_conf['ds_datafiles'] = {}
        for k, v in self.dataset_conf[self.target_ds_structure].items():
            self.dataset_conf['ds_datafiles'][
                k] = f"{self.config.experiment.dirs.tmp_data_dir}/{k}{self.file_suffix}.pkl"
        if (not self.ds_meta.exists()) or self.config.data_source.rebuild_dataset:
            self.build_ds_from_db_flow()
        else:
            self.load_ds_from_cache()

    def load_ds_from_cache(self) -> None:
        ds_metadata = utils.core_utils.load_json(self.ds_meta)
        self.dataset_conf['num_train_recs'], self.dataset_conf['num_val_recs'], self.dataset_conf[
            'num_test_recs'] = \
            ds_metadata["train_recs"], ds_metadata["val_recs"], ds_metadata["test_recs"]
        for k in self.dataset_conf[self.target_ds_structure].keys():
            self.dataset_conf[f'{k}_start_date'] = ds_metadata[f'{k}_start_date'] \
                if f'{k}_start_date' in ds_metadata.keys() else None
            self.dataset_conf[f'{k}_end_date'] = ds_metadata[f'{k}_end_date'] \
                if f'{k}_end_date' in ds_metadata.keys() else None
        self.dataset_conf['dsid'] = ds_metadata['dsid'] if 'dsid' in ds_metadata.keys() else None
        self.dataset_conf['albert_tokenizer'] = \
            AlbertTokenizer.from_pretrained(self.config.model.sp_model, max_len=self.config.data_source.max_seq_length,
                                            truncation=True)

    def build_ds_from_db_flow(self) -> None:
        if not self.config.experiment.db_functionality_enabled:
            logger.error(
                f"{constants.DB_WARNING_START} Since the specified cached dataset ({self.file_suffix[1:]}) "
                f"has not been found or cannot be rebuilt, instance aborting. "
                f"Please see repo readme for further details.")
            sys.exit(0)
        self.db_to_pkl()
        ds_dict = {"train_recs": self.dataset_conf['num_train_recs'], "val_recs": self.dataset_conf['num_val_recs'],
                   "test_recs": self.dataset_conf['num_test_recs']}
        for k in self.dataset_conf[self.target_ds_structure].keys():
            ds_dict[f'{k}_start_date'] = self.dataset_conf[f'{k}_start_date']
            ds_dict[f'{k}_end_date'] = self.dataset_conf[f'{k}_end_date']
            ds_dict['dsid'] = constants.APP_INSTANCE
        utils.core_utils.save_json(ds_dict, self.ds_meta)
        self.update_ds_db_metadata()
        self.arc_ds()

    def arc_ds(self) -> None:
        file_swaps = []
        for k in self.dataset_conf[self.target_ds_structure].keys():
            src_file = self.dataset_conf['ds_datafiles'][k]
            arc_file = f"{self.config.experiment.dirs.arc_data_dir}/{k}{self.file_suffix}.pkl"
            file_swaps.append((src_file, arc_file))
        file_swaps.append((self.ds_meta, f"{self.config.experiment.dirs.arc_data_dir}/ds_meta{self.file_suffix}.json"))
        #  archive created dataset files and replace with sym links
        dataprep.dataprep_utils.link_swap(file_swaps)

    def update_ds_db_metadata(self) -> None:
        self.ds_meta = Path(f"{self.config.experiment.dirs.tmp_data_dir}/ds_meta{self.file_suffix}.json")
        try:
            ds_metadata = utils.core_utils.load_json(self.ds_meta)
            ds_metadata_tmp = []
            ds_metadata_tmp.extend((ds_metadata['dsid'], self.file_suffix[1:]))
            for k in self.dataset_conf[self.target_ds_structure].keys():
                ds_metadata_tmp.extend([datetime.datetime.strptime(ds_metadata[f'{k}_start_date'], '%Y-%m-%d').date(),
                                        datetime.datetime.strptime(ds_metadata[f'{k}_end_date'], '%Y-%m-%d').date()])
            ds_metadata_tup = tuple(ds_metadata_tmp)
            self.save_ds_metadata(ds_metadata_tup)
        except (KeyError, FileNotFoundError) as e:
            logger.warning(
                f'exception occured while updating metadata, metadata may be old or missing, skipping metadata '
                f'update for now. Error:{e}')

    def save_ds_metadata(self, ds_metadata_tup: Tuple) -> Tuple[int, List]:
        inserted_rowcnt, insert_errors = db_utils.single_execute(self.cnxp.get_connection(),
                                                                 self.config.data_source.sql.ds_metadata_sql,
                                                                 ds_metadata_tup)
        logger.debug(f"Metadata update complete, {inserted_rowcnt} record(s) affected.")
        return inserted_rowcnt, insert_errors

    def db_to_pkl(self) -> None:
        self.dataset_conf['albert_tokenizer'] = \
            AlbertTokenizer.from_pretrained(self.config.model.sp_model, max_len=self.config.data_source.max_seq_length,
                                            truncation=True)
        self.dataset_conf['class_labels'] = [k for k in self.config.data_source.class_labels]
        self.filter_converge_set_sql()
        self.dataset_conf['class_cards'] = self.db_current_card()
        self.dataset_conf['class_ratio'] = self.calc_class_ratios()
        self.calc_class_weights()
        logger.info(f"Building a balanced dataset from the following raw class data:")
        for label, card in zip(self.dataset_conf['class_labels'], self.dataset_conf['class_cards']):
            logger.info(f"Label {label}: {card} records")
        self.define_dataset_structure()

    def filter_converge_set_sql(self) -> None:
        if self.config.experiment.debug.use_debug_dataset:
            self.set_class_sql(self.config.data_source.sql.debug.class_sql.values(),
                               self.config.data_source.sql.debug.class_card_sql.values(),
                               self.config.data_source.sql.debug.class_bound_card_sql.values())
        else:
            self.filter_truths()
            self.converge_class_distribution()
            self.set_class_sql(self.config.data_source.sql.primary.class_sql.values(),
                               self.config.data_source.sql.primary.class_card_sql.values(),
                               self.config.data_source.sql.primary.class_bound_card_sql.values())

    def set_class_sql(self, cls_sql: List[str], cls_card_sql: List[str], cls_bound_sql: List[str]) -> None:
        self.dataset_conf['class_sql'] = [sql for sql in cls_sql]
        self.dataset_conf['class_card_sql'] = [sql for sql in cls_card_sql]
        self.dataset_conf['class_bound_sql'] = [sql for sql in cls_bound_sql]

    def hash_based_prune(self) -> None:
        db_utils.truncate_existing(self.cnxp.get_connection(), "all_truth_statements_tmp")
        _ = self.prune_ft()

    def model_dist_prune(self, mapping_inputs: List, base_mode: bool = True):
        if base_mode:
            cand_table, cand_sql = "base_false_truth_del_cands", self.config.data_source.sql.base_model_prune_sql
        else:
            cand_table, cand_sql = "false_truth_del_cands", self.config.data_source.sql.dc_model_based_prune_sql
        truth_embed_dict, falsehood_embed_dict = self.fetch_embeddings(mapping_inputs, base_mode=base_mode)
        truth_embed_matrix = torch.stack(truth_embed_dict['embeds'], dim=0).to(device=self.device)
        falsehood_id_sim_tups = self.gen_false_truth_cands(falsehood_embed_dict, truth_embed_dict, truth_embed_matrix)
        self.write_del_topk_sim_tups(falsehood_id_sim_tups, cand_table, cand_sql)
        return mapping_inputs

    def write_del_topk_sim_tups(self, falsehood_id_tups: List, cand_table_name: str, prune_sql: str):
        # writing topk sim tups to db for tmp analysis
        db_utils.truncate_existing(self.cnxp.get_connection(), cand_table_name)
        inserted_rowcnt, error_rows = self.save_model_analysis(falsehood_id_tups)
        logger.info(f"Generated {inserted_rowcnt} candidates for false truth analysis")
        deletion_cnt = self.prune_ft(prune_sql)
        logger.info(
            f"Deleted {deletion_cnt} 'truths' from truths table based on similarity with falsehoods enumerated in "
            f"{cand_table_name}")

    def filter_truths(self) -> None:
        """
        see diagrams at https://deepclassiflie.org/index.html#data-pipeline
        """
        base_model_modes = [True, False]
        if not self.config.experiment.inference_ckpt:
            logger.info(
                "No inference checkpoint provided. Performing hash and base-model based (as opposed to DeepClassiflie "
                "model-based) dataset pruning")
            base_model_modes = [True]
        self.hash_based_prune()
        mapping_inputs = self.fetch_mapping_inputs()
        for bmode in base_model_modes:
            self.model_dist_prune(mapping_inputs, base_mode=bmode)

    def gen_false_truth_cands(self, falsehood_embed_dict: Dict, truth_embed_dict: Dict,
                              truth_embed_matrix: torch.Tensor) -> List:
        falsehood_id_sim_tups = []
        for f_id, f_embed in zip(falsehood_embed_dict['sids'], falsehood_embed_dict['embeds']):
            embed_compare = f_embed.expand_as(truth_embed_matrix)
            l2_norm_distance = torch.nn.PairwiseDistance(p=2)
            # prune up to k similar "truth" statements based on similarity to a falsehood, within a threshold
            # manually defined in dist_based_filter_vw
            top_embed_sims, top_embed_sims_idxs = torch.topk(l2_norm_distance(truth_embed_matrix, embed_compare),
                                                             largest=False, k=self.config.data_source.model_filter_topk)
            for sim, idx in zip(top_embed_sims, top_embed_sims_idxs):
                falsehood_id_sim_tups.append((f_id, truth_embed_dict['sids'][idx], sim.item()))
        return falsehood_id_sim_tups

    def fetch_embeddings(self, mapping_inputs: List, base_mode: bool = True) -> Tuple[Dict, Dict]:
        if self.config.data_source.filter_w_embed_cache:
            if base_mode:
                truth_embed_dict, falsehood_embed_dict = load_cache(self.base_truth_embed_cache), \
                                                         load_cache(self.base_falsehood_embed_cache)
            else:
                truth_embed_dict, falsehood_embed_dict = load_cache(self.truth_embed_cache), \
                                                         load_cache(self.falsehood_embed_cache)
        else:
            if base_mode:
                truth_embed_dict, falsehood_embed_dict = self.build_embed_mappings(mapping_inputs)
                torch.save(truth_embed_dict, self.base_truth_embed_cache)
                torch.save(falsehood_embed_dict, self.base_falsehood_embed_cache)
            else:
                truth_embed_dict, falsehood_embed_dict = self.build_embed_mappings(mapping_inputs, base_mode=False)
                torch.save(truth_embed_dict, self.truth_embed_cache)
                torch.save(falsehood_embed_dict, self.falsehood_embed_cache)
        return truth_embed_dict, falsehood_embed_dict

    def save_model_analysis(self, report_tups: List[Tuple], base_mode: bool = True) -> Tuple[int, List]:
        cand_save_sql = self.config.data_source.sql.base_model_based_cands_sql if base_mode \
            else self.config.data_source.sql.dc_model_based_cands_sql
        # save analysis of candidate "false" truths to be deleted/deduped from truths statements source
        inserted_rowcnt, insert_errors = db_utils.batch_execute_many(self.cnxp.get_connection(), cand_save_sql,
                                                                     report_tups,
                                                                     self.config.data_source.db_commit_freq)
        return inserted_rowcnt, insert_errors

    def fetch_mapping_inputs(self):
        mapping_inputs = []
        truth_sql = self.config.data_source.sql.build_truths_embedding
        falsehood_sql = self.config.data_source.sql.build_falsehoods_embedding
        for idsql in [truth_sql, falsehood_sql]:
            mapping_inputs.append(db_utils.fetchallwrapper(self.cnxp.get_connection(), idsql))
        return mapping_inputs

    def build_embed_mappings(self, mapping_inputs: List, base_mode: bool = True) -> List[Dict]:
        embed_mappings = []
        for inputs in mapping_inputs:
            embed_mappings.append(Inference(self.config, mapping_set=inputs, base_mode=base_mode).init_predict())
        return embed_mappings

    def converge_class_distribution(self) -> None:
        subclass_datasets, subclass_weights, max_lim = self.build_subclass_datasets(*self.prep_base_sql())
        ds_iter = dataprep.dataprep_utils.UnivariateDistReplicator(subclass_datasets, subclass_weights, max_lim)
        # TODO: update skipped_samples messages log level
        if sum(ds_iter.skipped_samples) > 0:
            logger.warning(f"Some subclass samples were skipped due to data source exhaustion."
                           f"Target subclass density will differ from requested transformation as summarized below:")
            for i, skpd in enumerate(ds_iter.skipped_samples):
                if skpd > 0:
                    logger.warning(
                        f"skipped {skpd} samples of subclass #{i}. Target vs actual density: {ds_iter.weights[i]} vs "
                        f"{len(ds_iter.datasets[i]) / len(ds_iter)} ")
        # save transformed distribution to db for later analysis
        inserted_rowcnt = self.dist_to_db(ds_iter)
        logger.info(f"saved {inserted_rowcnt} rows of a transformed truth distribution to db")

    def prep_base_sql(self) -> Tuple[Iterator, str]:
        db_utils.truncate_existing(self.cnxp.get_connection(), "pt_converged_truths")
        db_gen = db_utils.db_ds_gen(self.cnxp, self.config.data_source.sql.converge_class_dist,
                                    self.config.data_source.db_fetch_size)
        subclass_sql_base = self.config.data_source.sql.converge_dist_subclasses
        return db_gen, subclass_sql_base

    def build_subclass_datasets(self, db_gen: Iterator, subclass_sql_base: str) -> Tuple[List[Dataset],
                                                                                         List[float], int]:
        max_lim = 0
        subclass_gens, subclass_weights = [], []
        for row in db_gen:
            subclass_sql = f"{subclass_sql_base}{row[0]}"
            # pass connection pool handle to db_ds_gen
            subclass_gens.append(db_utils.db_ds_gen(self.cnxp, subclass_sql,
                                                    self.config.data_source.db_fetch_size))
            subclass_weights.append(float(row[1]))
            if max_lim == 0:
                max_lim = row[2]
        subcls_datasets = []
        for i, ds in enumerate(subclass_gens):
            stexts, stypes, sdates = [], [], []
            for (stext, stype, sdate) in ds:
                stexts.append(stext)
                stypes.append(stype)
                sdates.append(sdate)
            subcls_datasets.append(dataprep.dataprep_utils.TempTextDataset(stexts, stypes, sdates))
        return subcls_datasets, subclass_weights, int(max_lim)

    def prune_ft(self, prune_sql: str = None) -> int:
        conn = self.cnxp.get_connection()
        cursor = conn.cursor(prepared=True)
        altered_rowcnt = 0
        if not prune_sql:
            prune_false_truths_sql = self.config.data_source.sql.hash_based_prune_sql
            for prune_sql in prune_false_truths_sql:
                cursor.execute(prune_sql)
                if cursor.rowcount >= 1:
                    altered_rowcnt += cursor.rowcount
        else:
            cursor.execute(prune_sql)
            if cursor.rowcount >= 1:
                altered_rowcnt += cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        return altered_rowcnt

    def dist_to_db(self, converged_ds_iter: Sampler) -> int:
        stmts = []
        for sample in converged_ds_iter:
            stmts.append((sample[0], sample[1], sample[2]))
        inserted_rowcnt, _ = db_utils.batch_execute_many(self.cnxp.get_connection(),
                                                         self.config.data_source.sql.converge_truths, stmts,
                                                         self.config.data_source.db_commit_freq)
        return inserted_rowcnt

    def db_current_card(self) -> List:
        class_cnts = []
        conn = self.cnxp.get_connection()
        logger.debug(f"DB connection obtained: {conn}")
        cursor = conn.cursor()
        for sql in self.dataset_conf['class_card_sql']:
            cursor.execute(sql)
            row = cursor.fetchone()
            class_cnts.append(row[0])
        cursor.close()
        conn.close()
        logger.debug(f"DB connection closed: {conn}")
        return class_cnts

    def calc_class_ratios(self) -> List:
        max_cnt = max(self.dataset_conf['class_cards'])
        class_ratios = [max_cnt / cnt for cnt in self.dataset_conf['class_cards']]
        return class_ratios

    def calc_class_weights(self) -> None:
        if self.config.data_source.use_class_weights:
            self.dataset_conf['class_weights'] = {}
            class_weights = []
            card_sum = sum(self.dataset_conf['class_cards'])
            for i, card in enumerate(self.dataset_conf['class_cards']):
                class_weights[i] = (1 / card) * card_sum / 2.0
            label_int = 0
            for label, weight in zip(self.dataset_conf['class_labels'], class_weights):
                self.dataset_conf['class_weights'][label_int] = weight
                label_int += 1
            for w, l in self.dataset_conf['class_weights']:
                logger.debug(f"weight for label {l} is {w}.")
        else:
            self.dataset_conf['oversampling_weights'] = softmax(np.array(self.dataset_conf['class_cards']))

    def define_dataset_structure(self) -> None:
        if self.config.experiment.debug.use_debug_dataset is True:
            bsql = self.config.data_source.sql.debug.dist_dt_bound_sql
            dt_lower_bound = datetime.datetime.strptime('2017-01-19', '%Y-%m-%d').date()
        else:
            bsql = self.config.data_source.sql.converge_dist_dt_bound_sql
            dt_lower_bound = datetime.datetime.strptime('1900-01-01', '%Y-%m-%d').date()
        cume_v = 0
        for k, v in self.dataset_conf[self.target_ds_structure].items():
            # setup db dataset generators per specified train/dev/test splits
            cume_v += v
            sql = f"{bsql} {cume_v}"
            dt_upper_bound = db_utils.fetch_one(self.cnxp.get_connection(), sql)[0]
            ds_gen = self.construct_gen(dt_lower_bound, dt_upper_bound)
            recs, xformer_examples, ctxt_features = dataprep.dataprep_utils.parse_sql_to_example(ds_gen, k)
            self.dataset_conf[f'num_{k}_recs'] = recs
            self.dataset_conf[f'{k}_start_date'] = self.dataset_conf['start_date']
            self.dataset_conf[f'{k}_end_date'] = self.dataset_conf['end_date']
            xformer_features = convert_examples_to_features(xformer_examples, self.dataset_conf['albert_tokenizer'],
                                                            label_list=self.dataset_conf['class_labels'],
                                                            max_length=self.config.data_source.max_seq_length,
                                                            output_mode="classification")
            logger.info(f"Saving features into cached file {self.dataset_conf['ds_datafiles'][k]}", )
            torch.save([xformer_features, ctxt_features], self.dataset_conf['ds_datafiles'][k])
            dt_lower_bound = dt_upper_bound

    def construct_gen(self, start_dt: datetime, end_dt: datetime) -> Iterator:
        sql_stmts = []
        sql_bound_pred = f"STR_TO_DATE('{start_dt.strftime('%Y-%m-%d')}','%Y-%m-%d') " \
                         f"and STR_TO_DATE('{end_dt.strftime('%Y-%m-%d')}','%Y-%m-%d')"
        for sql in self.dataset_conf['class_sql']:
            sql_stmts.append(f"{sql} {sql_bound_pred}")
        gens = [db_utils.db_ds_gen(self.cnxp, sql, self.config.data_source.db_fetch_size)
                for sql in sql_stmts]
        self.dataset_conf['start_date'], self.dataset_conf['end_date'] = start_dt, end_dt
        if self.config.data_source.class_balancing_strategy == "class_weights":
            gen = dataprep.dataprep_utils.class_weight_gen(gens)
        else:
            if not dataprep.dataprep_utils.validate_normalized(self.config.data_source.sampling_weights):
                logger.error(f"invalid values specified {self.config.data_source.sampling_weights} (sum must == 1.0). "
                             f"Please reconfigure and restart")
                sys.exit(0)
            cls_bound_cards = []
            for sql in self.dataset_conf['class_bound_sql']:
                sql = f"{sql} {sql_bound_pred}"
                cls_bound_cards.append(db_utils.fetch_one(self.cnxp.get_connection(), sql)[0])
            gen = dataprep.dataprep_utils.ds_minority_oversample_gen(self.config.data_source.sampling_weights, gens,
                                                                     cls_bound_cards)
        return gen

    def build_dataset(self, dstype: str) -> TensorDataset:
        if dstype == 'train':
            batch_size = self.dataset_conf['train_batch_size']
        elif dstype == 'val':
            batch_size = self.dataset_conf['val_batch_size']
        else:
            batch_size = self.dataset_conf['test_batch_size']
        df = self.dataset_conf['ds_datafiles'][dstype]
        logger.info(f"Loading features from cached file {df}")
        xformer_features, ctxt_features = torch.load(df)
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in xformer_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in xformer_features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in xformer_features], dtype=torch.long)
        all_ctxt_types = torch.tensor(ctxt_features, dtype=torch.float)
        all_labels = torch.tensor([f.label for f in xformer_features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_ctxt_types, all_labels)
        if self.config.experiment.debug.debug_enabled:
            DatasetCollection.debug_dataset_output(dataset, batch_size, dstype)
        return dataset

    @staticmethod
    def debug_dataset_output(dataset: TensorDataset, batch_size: int, dstype: str) -> None:
        batch_means = []
        debug_sampler = RandomSampler(dataset)
        debug_dataloader = DataLoader(dataset, sampler=debug_sampler, batch_size=batch_size)
        debug_iterator = tqdm.tqdm(debug_dataloader, desc="Dataset Samples")
        debug_batches = 3
        for step, batch in enumerate(debug_iterator):
            if step < debug_batches:
                batch = tuple(t for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'ctxt_type': batch[3],
                          'labels': batch[4]}
                logger.debug(f"Batch {step}: input sizes: "
                             f"{inputs['input_ids'].shape},"
                             f"{inputs['attention_mask'].shape},"
                             f"{inputs['token_type_ids'].shape}, "
                             f"{inputs['ctxt_type'].shape}, "
                             f"labels: {inputs['labels'].shape}")
                batch_means.append(inputs['labels'].numpy().mean())
                logger.debug(f"Batch {step} label mean: {inputs['labels'].numpy().mean():.2f}")
            else:
                break
        logger.debug(
            f"using {debug_batches} batches, sampled a {dstype} dataset mean of: {np.average(batch_means):.2f} ")
