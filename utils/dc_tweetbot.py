import datetime
import logging
from typing import MutableMapping, NoReturn, Optional, Union, List, Tuple
import os
import shutil
import time
import traceback
import sys

import tweepy
from tweepy import TweepError
from mysql.connector import Error

import utils.constants as constants

from db_ingest import refresh_db, get_cnxp_handle
from db_utils import fetchallwrapper, batch_execute_many, fetch_one
from envconfig import create_dirs
from analysis.inference import Inference

logger = logging.getLogger(constants.APP_NAME)


class DCTweetBot(object):
    def __init__(self, config: MutableMapping) -> NoReturn:
        self.config = config
        self.cnxp = get_cnxp_handle()
        self.dtmask = '%Y%m%d'
        self.mimask = '%Y%m%d%H%M%S'
        self.non_twitter_updatefreq = round(config.experiment.tweetbot.non_twitter_update_freq_multiple) \
            if config.experiment.tweetbot.non_twitter_update_freq_multiple >= 1 else None
        self.api = self.authenticate_dcbot()
        self.tweetbot_dbconf = (self.config.experiment.tweetbot, self.config.experiment.dirs.instance_log_dir)
        self.poll_and_analyze()

    def poll_and_analyze(self) -> NoReturn:
        last_nt_update_cnt = 0
        update_nt = True
        while True:
            self.verify_date()
            self.purge_old_reports()
            if self.non_twitter_updatefreq and update_nt:
                update_nt, last_nt_update_cnt = self.update_non_twit_sources()
            refresh_db(self.config.data_source.db_conf, self.cnxp, self.tweetbot_dbconf, self.api)
            self.maybe_publish('tweets')
            time.sleep(self.config.experiment.tweetbot.dcbot_poll_interval)
            if self.non_twitter_updatefreq:
                last_nt_update_cnt += 1
                if last_nt_update_cnt >= self.non_twitter_updatefreq:
                    update_nt = True

    def update_non_twit_sources(self):
        refresh_db(self.config.data_source.db_conf, self.cnxp, self.tweetbot_dbconf,
                   api_handle=self.api, nontwtr_update=True)
        update_nt = False
        last_nt_update_cnt = 0
        self.maybe_publish('stmts')
        return update_nt, last_nt_update_cnt

    def purge_old_reports(self) -> None:
        curr_day = datetime.date.today()
        purge_base = f"{self.config.experiment.dirs.base_dir}/repos/{constants.APP_NAME}_history/"
        purge_date = (curr_day - datetime.timedelta(days=self.config.inference.rpt_hist_retention_days))
        for d in os.listdir(purge_base):
            try:
                d_date = datetime.datetime.strptime(d, self.dtmask).date()
                if d_date < purge_date:
                    shutil.rmtree(f"{purge_base}/{d}")
            except ValueError:
                logger.debug(f'detected a history file/dir with invalid date format ({d}), not purging that item...')

    def verify_date(self) -> None:
        curr_day = datetime.datetime.now().strftime(self.dtmask)
        self.config.experiment.dirs.rpt_arc_dir_t = f"{self.config.experiment.dirs.base_dir}" \
                                                    f"/repos/{constants.APP_NAME}_history/{curr_day}" \
                                                    f"/predicted_truths"
        self.config.experiment.dirs.rpt_arc_dir_f = f"{self.config.experiment.dirs.base_dir}" \
                                                    f"/repos/{constants.APP_NAME}_history/{curr_day}/" \
                                                    f"predicted_falsehoods"
        logger.debug(f"Current day's reports ({curr_day}) are being archived to "
                     f"{self.config.experiment.dirs.rpt_arc_dir_t} and {self.config.experiment.dirs.rpt_arc_dir_f}")
        new_dirs = [d for d in [self.config.experiment.dirs.rpt_arc_dir_t, self.config.experiment.dirs.rpt_arc_dir_f] if
                    not os.path.exists(d)]
        if new_dirs:
            create_dirs(new_dirs)
        else:
            logger.debug(f"No new directories created, current day's ({curr_day}) dirs "
                         f"{self.config.experiment.dirs.rpt_arc_dir_t} and "
                         f"{self.config.experiment.dirs.rpt_arc_dir_f} already exist")

    def maybe_publish(self, target_type: str) -> None:
        # N.B. publishing all statements and tweets that meet length thresholds, driven by four tables:
        # a published and "notpublished" table for both statements and tweets
        # since metadata is substantially different and not straightforward to cleanly combine)
        if target_type == 'stmts':
            target_tups = fetchallwrapper(self.cnxp.get_connection(),
                                          self.config.experiment.tweetbot.sql.stmts_to_analyze_sql)
            interval = self.config.experiment.tweetbot.dcbot_poll_interval * self.non_twitter_updatefreq
        else:
            target_tups = fetchallwrapper(self.cnxp.get_connection(),
                                          self.config.experiment.tweetbot.sql.tweets_to_analyze_sql)
            interval = self.config.experiment.tweetbot.dcbot_poll_interval
        if target_tups:
            self.prep_new_threads(target_tups)
            self.publish_reports(Inference(self.config).init_predict(), target_type)
        else:
            logger.info(f"No new {target_type} found to analyze and publish. Trying again in {interval} seconds")

    def authenticate_dcbot(self) -> tweepy.API:
        # store tokens in DB and retrieve them rather than set them in memory
        consumer_key, consumer_secret, access_token, access_token_secret = \
            fetch_one(self.cnxp.get_connection(), self.config.experiment.tweetbot.sql.get_bot_creds_sql)
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

    def publish_reports(self, unpub_reports_tup: Tuple, target_type: str) -> None:
        published_reports = []
        analyzed_reports = []
        for (report_image, id1, id2, conf_b, pred_class, _, _) in unpub_reports_tup:
            dc_tweettup = self.tweet_and_archive(conf_b, pred_class, report_image)
            # log analyzed or published report
            published_reports, analyzed_reports = self.prep_logging(conf_b, pred_class, target_type, published_reports,
                                                                    dc_tweettup, report_image, id1,
                                                                    id2, analyzed_reports)
        published_rowcnt, error_rows = self.log_report(published_reports, target_type, 'pub')
        logger.info(f"published {published_rowcnt} reports")
        analyzed_rowcnt, error_rows = self.log_report(analyzed_reports, target_type, 'nopub')
        logger.info(f"analyzed {analyzed_rowcnt} reports")

    def prep_logging(self, conf_b: bool, pred_class: int, target_type: str, published_reports: Optional[List[Tuple]],
                     dc_tweettup: Tuple, report_image: str, id1: Union[str, int], id2: int,
                     analyzed_reports: Optional[List[Tuple]]) -> Tuple[List[Tuple], List[Tuple]]:
        if self.config.experiment.tweetbot.publish and conf_b and pred_class == 1:
            if target_type == 'tweets':
                published_reports.append((dc_tweettup, report_image, id1))
            else:
                published_reports.append((dc_tweettup, report_image, id1, id2))
        else:
            if target_type == 'tweets':
                analyzed_reports.append((report_image, id1))
            else:
                analyzed_reports.append((report_image, id1, id2))
        return published_reports, analyzed_reports

    def tweet_and_archive(self, conf_b: bool, pred_class: int, report_image: str) -> Tuple:
        dc_tweettup = None
        try:
            # publish tweets if publish==true & falsehood confidence reaches threshold otherwise, analyze only
            if self.config.experiment.tweetbot.publish and conf_b and pred_class == 1:
                dc_tweettup = self.tweet_report(report_image)
            if pred_class == 0:
                shutil.copy2(report_image, self.config.experiment.dirs.rpt_arc_dir_t)
            elif pred_class == 1:
                shutil.copy2(report_image, self.config.experiment.dirs.rpt_arc_dir_f)
            else:
                logger.warning(f"invalid pred_class, skipping archival...")
            os.remove(report_image)
        except Error as e:  # a lot could go wrong here. for now, shamefully using a broad except and logging traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(f"Encountered following error publishing report::"
                         f" {repr(traceback.format_exception(exc_type, exc_value, exc_traceback))}")
            raise e  # TODO: remove this to log error and continue daemon?
        return dc_tweettup

    def log_report(self, reports: List[Tuple], target_type: str, pubtype: str) -> Tuple[int, int]:
        report_sql, rpts = self.pub_flow(target_type, reports) \
            if pubtype == 'pub' else self.analyze_flow(target_type, reports)
        return batch_execute_many(self.cnxp.get_connection(), report_sql, rpts)

    def tweet_report(self, report_image: str) -> Tuple:
        media = self.api.media_upload(report_image)
        post_result = self.api.update_status(status=constants.TWEET_STATUS_MSG, media_ids=[media.media_id])
        dc_tweetid = post_result.id
        dc_media_id = media.media_id
        dc_tweettup = (dc_tweetid, dc_media_id)
        logger.info(f"published report with tweet_id {dc_tweetid}")
        return dc_tweettup

    def prep_new_threads(self, target_tups: List[Tuple]) -> None:
        pred_inputs = []
        for (parentid, childid, text, ctxt_type) in target_tups:
            pred_inputs.append([parentid, childid, text, ctxt_type, ""])
        self.config.inference.pred_inputs = pred_inputs

    def pub_flow(self, target_type: str, reports: List[Tuple]) -> Tuple[str, List[Tuple]]:
        rpts = []
        if target_type == 'tweets':
            for (dc_tweettup, report_image, thread_id) in reports:
                dc_tweetid, dc_media_id = dc_tweettup
                arc_report_name = os.path.basename(report_image)
                report_row = (dc_tweetid, thread_id, arc_report_name, dc_media_id)
                rpts.append(report_row)
            statement_sql = self.config.experiment.tweetbot.sql.tweets_analyzed_pub_sql
        else:
            for (dc_tweettup, report_image, tid, sid) in reports:
                dc_tweetid, dc_media_id = dc_tweettup
                arc_report_name = os.path.basename(report_image)
                report_row = (dc_tweetid, tid, sid, arc_report_name, dc_media_id)
                rpts.append(report_row)
            statement_sql = self.config.experiment.tweetbot.sql.stmts_analyzed_pub_sql
        return statement_sql, rpts

    def analyze_flow(self, target_type: str, reports: List[Tuple]) -> Tuple[str, List[Tuple]]:
        rpts = []
        if target_type == 'tweets':
            for (report_image, thread_id) in reports:
                arc_report_name = os.path.basename(report_image)
                report_row = (thread_id, arc_report_name)
                rpts.append(report_row)
            statement_sql = self.config.experiment.tweetbot.sql.tweets_analyzed_nopub_sql
        else:
            for (report_image, tid, sid) in reports:
                arc_report_name = os.path.basename(report_image)
                report_row = (tid, sid, arc_report_name)
                rpts.append(report_row)
            statement_sql = self.config.experiment.tweetbot.sql.stmts_analyzed_nopub_sql
        return statement_sql, rpts
