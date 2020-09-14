import logging
from typing import MutableMapping, NoReturn, List, Tuple, Dict
import os
import time
import traceback
import sys
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import tweepy
from tweepy import TweepError
import requests

import utils.constants as constants
from db_ingest import refresh_db, get_cnxp_handle
from db_utils import fetchallwrapper, batch_execute_many, fetch_one, single_execute
from utils.core_utils import save_json, to_json, load_json
from analysis.inference import Inference
from analysis.interpretation_utils import load_cache, calc_accuracy_data

logger = logging.getLogger(constants.APP_NAME)


class DCInfSvc(object):
    def __init__(self, config: MutableMapping) -> None:
        self.config = config
        self.cnxp = get_cnxp_handle()
        self.dtmask = '%Y%m%d'
        self.mimask = '%Y%m%d%H%M%S'
        self.svc_auth = {}
        self.fetch_auth_creds()
        self.non_twitter_updatefreq = round(config.experiment.infsvc.non_twitter_update_freq_multiple) \
            if config.experiment.infsvc.non_twitter_update_freq_multiple >= 1 else None
        self.model_perf_tups = {'tweets': load_cache(f'{self.config.experiment.dirs.model_cache_dir}/'
                                                     f'{constants.TWEET_MODEL_PERF_CACHE_NAME}'),
                                'nontweets': load_cache(f'{self.config.experiment.dirs.model_cache_dir}/'
                                                        f'{constants.NONTWEET_MODEL_PERF_CACHE_NAME}'),
                                'global': load_cache(f'{self.config.experiment.dirs.model_cache_dir}/'
                                                     f'{constants.GLOBAL_MODEL_PERF_CACHE_NAME}')}
        self.pinned_preds_cache = Path(os.path.join(self.config.experiment.dirs.model_cache_dir,
                                                    constants.LOCAL_INFSVC_PUB_CACHE_NAME))
        self.infsvc_dbconf = (self.config.experiment.infsvc, self.config.experiment.dirs.instance_log_dir)
        self.batch_inference() if self.config.experiment.infsvc.batch_mode else self.poll_and_analyze()

    def fetch_auth_creds(self):
        self.svc_auth['twitter'] = self.authenticate_dcbot()
        for creds, sql in zip(['pinata', 'cloudflare'],
                              [self.config.experiment.infsvc.sql.get_pinata_creds_sql,
                               self.config.experiment.infsvc.sql.get_cloudflare_creds_sql]):
            self.svc_auth[creds] = fetch_one(self.cnxp.get_connection(), sql)

    def batch_inference(self) -> None:
        if not self.config.experiment.infsvc.skip_db_refresh:
            refresh_db(self.config.data_source.db_conf, self.cnxp, self.infsvc_dbconf, self.svc_auth['twitter'],
                       batch_infsvc=True)
        self.publish_flow()

    def poll_and_analyze(self) -> NoReturn:
        last_nt_update_cnt = 0
        update_nt = True
        while True:
            if self.non_twitter_updatefreq and update_nt:
                update_nt, last_nt_update_cnt = self.update_non_twit_sources()
            refresh_db(self.config.data_source.db_conf, self.cnxp, self.infsvc_dbconf, self.svc_auth['twitter'])
            self.publish_flow()
            time.sleep(self.config.experiment.infsvc.dcbot_poll_interval)
            if self.non_twitter_updatefreq:
                last_nt_update_cnt += 1
                if last_nt_update_cnt >= self.non_twitter_updatefreq:
                    update_nt = True

    def update_non_twit_sources(self):
        refresh_db(self.config.data_source.db_conf, self.cnxp, self.infsvc_dbconf,
                   api_handle=self.svc_auth['twitter'], nontwtr_update=True)
        update_nt = False
        last_nt_update_cnt = 0
        return update_nt, last_nt_update_cnt

    def publish_flow(self) -> None:
        # N.B. publishing all statements and tweets that meet length thresholds, driven by separate statements/tweets
        # tables since metadata is substantially different and not straightforward to cleanly combine
        target_tups = []
        for sql in [self.config.experiment.infsvc.sql.stmts_to_analyze_sql,
                    self.config.experiment.infsvc.sql.tweets_to_analyze_sql]:
            target_tups.extend(fetchallwrapper(self.cnxp.get_connection(), sql))
        if target_tups:
            inf_metadata = self.prep_new_threads(target_tups)
            self.publish_inference(Inference(self.config).init_predict(), inf_metadata)
        else:
            logger.info(f"No new claims found to analyze and publish")

    def authenticate_dcbot(self) -> tweepy.API:
        # store tokens in DB and retrieve them rather than set them in memory
        consumer_key, consumer_secret, access_token, access_token_secret = \
            fetch_one(self.cnxp.get_connection(), self.config.experiment.infsvc.sql.get_bot_creds_sql)
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True,
                         wait_on_rate_limit_notify=True)
        try:
            api.verify_credentials()
            print("Authentication OK")
        except TweepError:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(f"Error during authentication:"
                         f" {repr(traceback.format_exception(exc_type, exc_value, exc_traceback))}")
        return api

    def apply_pin_limit(self, cid_list: List) -> List:
        return cid_list[-self.config.experiment.infsvc.pinata_stmt_limit:] if len(cid_list) > self.config.experiment.infsvc.pinata_stmt_limit else cid_list

    def publish_inference(self, inf_probs: List, inf_metadata: Dict) -> None:
        perf_keys = ['model_version', 'bucket_acc', 'global_acc', 'global_auc', 'global_mcc', 'ppv', 'npv', 'ppr',
                     'npr', 'tp_ratio', 'tn_ratio', 'fp_ratio', 'fn_ratio', 'test_start_date', 'test_end_date']
        tweet_inf_outputs, stmt_inf_outputs = self.build_inf_outputs(inf_probs, inf_metadata, perf_keys)
        all_inf_outputs = stmt_inf_outputs + tweet_inf_outputs
        all_inf_outputs = self.apply_pin_limit(all_inf_outputs)
        if not(self.pinned_preds_cache.exists()):
            pin_flow_success = self.pin_flow(all_inf_outputs)
            if pin_flow_success:
                save_json(all_inf_outputs, self.pinned_preds_cache)
        else:
            existing_preds = load_json(self.pinned_preds_cache)
            # noinspection PyUnresolvedReferences
            new_preds = self.apply_pin_limit(existing_preds + all_inf_outputs)
            pin_flow_success = self.pin_flow(new_preds, rm_previous=True)
            if pin_flow_success:
                save_json(new_preds, self.pinned_preds_cache)
        if pin_flow_success:
            self.log_published_preds(stmt_inf_outputs, tweet_inf_outputs)

    def build_inf_outputs(self, inf_probs, inf_metadata, perf_keys) -> Tuple[List, List]:
        tweet_inf_outputs, stmt_inf_outputs = [], []
        for prob, pid, cid, ctype, t_url, t_date, claim_text in zip(inf_probs, *list(inf_metadata.values())):
            tmp_d = defaultdict()
            is_stmt = True if ctype == 0 else False
            if ctype == 1:
                output_l = tweet_inf_outputs
                tmp_d['thread_id'] = pid
            else:
                output_l = stmt_inf_outputs
                tmp_d['tid'] = pid
                tmp_d['sid'] = cid
            tmp_d['prediction'] = 1 if prob >= 0.5 else 0
            raw_confidence = round(prob, 4) if prob >= 0.5 else round(1-prob, 4)
            tmp_d['raw_pred'] = prob
            tmp_d['raw_confidence'] = raw_confidence
            accuracy_tup = calc_accuracy_data(self.model_perf_tups, raw_confidence, is_stmt)
            for t, k in zip(accuracy_tup, perf_keys):
                t = float(t) if isinstance(t, Decimal) else t
                tmp_d[k] = t
            for k, v in zip(['transcript_url', 't_date', 'claim_text'], [t_url, t_date, claim_text]):
                tmp_d[k] = v
            output_l.append(tmp_d)
        return tweet_inf_outputs, stmt_inf_outputs

    @staticmethod
    def init_infsvc_recs() -> Tuple[List, List]:
        common_inf_keys = ['prediction', 'raw_pred', 'raw_confidence', 'transcript_url', 't_date', 'claim_text']
        perf_keys = ['model_version', 'bucket_acc', 'global_acc', 'global_auc', 'global_mcc', 'ppv', 'npv', 'ppr',
                     'npr', 'tp_ratio', 'tn_ratio', 'fp_ratio', 'fn_ratio', 'test_start_date', 'test_end_date']
        common_inf_keys.extend(perf_keys)
        tweet_inf_output_keys = ['thread_id']
        stmt_inf_output_keys = ['tid', 'sid']
        for l in [tweet_inf_output_keys, stmt_inf_output_keys]:
            l.extend(common_inf_keys)
        return tweet_inf_output_keys, stmt_inf_output_keys

    def log_published_preds(self, stmt_inf_outputs: List, tweet_inf_outputs: List) -> None:
        stmt_tups, tweet_tups = DCInfSvc.prep_logging(stmt_inf_outputs, tweet_inf_outputs)
        published_stmt_cnt, pub_stmt_errors = batch_execute_many(self.cnxp.get_connection(),
                                                                       self.config.experiment.infsvc.sql.stmts_pub_sql,
                                                                       stmt_tups)
        published_tweet_cnt, pub_tweet_errors = batch_execute_many(self.cnxp.get_connection(),
                                                                   self.config.experiment.infsvc.sql.tweets_pub_sql,
                                                                   tweet_tups)
        logger.info(f"published {published_stmt_cnt} statement and {published_tweet_cnt} tweet inference records")

    @staticmethod
    def prep_logging(stmt_outputs: List, tweet_outputs: List) -> Tuple[List[Tuple], List[Tuple]]:
        iid = 1  # currently only one issuer id
        stmt_tups = []
        tweet_tups = []
        for d in stmt_outputs:
            stmt_tups.append((d['model_version'], iid, d['tid'], d['sid'], d['prediction'], d['raw_pred'],
                              d['raw_confidence']))
        for d in tweet_outputs:
            tweet_tups.append((d['model_version'], iid, d['thread_id'], d['prediction'], d['raw_pred'],
                               d['raw_confidence']))
        return stmt_tups, tweet_tups

    def post_w_backoff(self, endpoint: str, headers: Dict, cid_json: str) -> requests.Response:
        retries = 0
        max_retries = self.config.experiment.infsvc.max_retries
        while True:
            cid_json = cid_json.replace('\n', '')
            r = requests.post(endpoint, headers=headers, data=cid_json)
            if r.ok:
                return r
            else:
                if retries < max_retries:
                    sleep_time = self.config.experiment.infsvc.init_wait * math.pow(2, retries)
                    logger.warning(
                        f"Encountered error ({r.reason}) posting json to pinata, retrying again in {sleep_time} "
                        f"seconds")
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    logger.error(
                        f"Max retries ({max_retries}) reached and post to pinata is still failing with {r.reason}. "
                        f"Exiting inference service.")
                    raise KeyboardInterrupt

    def pin_cid(self, preds: List, headers: Dict) -> Tuple[Tuple, int, int]:
        cid_json = to_json(preds)
        r_tup = tuple((constants.PINATA_PINJSON_ENDPOINT, headers, cid_json))
        r = self.post_w_backoff(*r_tup)
        pinata_response = r.json()
        pinned_tup = tuple((1, pinata_response['IpfsHash'], pinata_response['PinSize']))
        pin_cnt, pin_error = single_execute(self.cnxp.get_connection(),
                                            self.config.experiment.infsvc.sql.save_pinned_cid_sql, pinned_tup)
        return pinned_tup, pin_cnt, pin_error

    @staticmethod
    def unpin_cid(target_cid: str, headers: Dict) -> bool:
        r = requests.delete(f'{constants.PINATA_UNPINJSON_ENDPOINT}/{target_cid}', headers=headers)
        if r.status_code == 200:
            logger.info(f'Unpinned previous cid: {target_cid}')
            return True
        else:
            logger.warning(f'Unexpected status code {r.status_code} while unpinning hash: {target_cid}')
            return False

    def pin_flow(self, preds: List, rm_previous: bool = False) -> bool:
        current_cid = None
        headers = {'pinata_api_key': self.svc_auth['pinata'][0],
                   'pinata_secret_api_key': self.svc_auth['pinata'][1], 'Content-Type': 'application/json'}
        if rm_previous:
            current_cid = fetch_one(self.cnxp.get_connection(),
                                    self.config.experiment.infsvc.sql.fetch_current_pinned_cid_sql)
        pinned_tup, pin_cnt, pin_error = self.pin_cid(preds, headers)
        if pin_cnt == 1 and pin_error == 0:
            logger.info(f'Pinned latest unlabeled model predictions {pinned_tup[1]} with size {pinned_tup[2]}')
            self.patch_dns(pinned_tup[1])
            if rm_previous and (current_cid[0] != pinned_tup[1]):
                return DCInfSvc.unpin_cid(current_cid[0], headers)
            return True
        else:
            logger.warning(f'Unexpected pinning results. Pinned {pin_cnt} items, with {pin_error} errors detected '
                           f'while logging pinned items. You may want to inspect pinata/ipfs items.')
            return False

    def patch_dns(self, new_hash: str) -> None:
        dns_rid = self.svc_auth['cloudflare'][3] if constants.DEV_MODE else self.svc_auth['cloudflare'][2]
        dns_path = f"{self.svc_auth['cloudflare'][1]}/dns_records/{dns_rid}"
        headers = {"Authorization": f"Bearer {self.svc_auth['cloudflare'][0]}", "Content-Type": "application/json"}
        data = {"type": "TXT", "name": f"_dnslink.{constants.DC_PREDICTIONS_SUBDOMAIN}",
                "content": f"dnslink=/ipfs/{new_hash}"}
        r = requests.patch(f'{constants.CLOUDFLARE_DC_DNS_ENDPOINT}/{dns_path}', headers=headers, data=to_json(data))
        r = r.json()
        logger.info(f'DNS patch succeeded, now serving predictions using the hash: {new_hash}') if r['success'] else \
            logger.warning(f'dns patch did not succeed, existing hash will be unpinned and may be gc\'d')

    def prep_new_threads(self, target_tups: List[Tuple]) -> Dict:
        pred_inputs = []
        inf_metadata = {'parentid': [], 'childid': [], 'ctxt_type': [], 'transcript_url': [], 't_date': [],
                        'claim_text': []}
        for (parentid, childid, claim_text, ctxt_type, t_date, transcript_url) in target_tups:
            pred_inputs.append([parentid, childid, claim_text, ctxt_type, ""])
            for k, v in zip(inf_metadata.keys(), [parentid, childid, ctxt_type, transcript_url, t_date, claim_text]):
                inf_metadata[k].append(v)
        self.config.inference.pred_inputs = pred_inputs
        return inf_metadata
