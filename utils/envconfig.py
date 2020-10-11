import collections.abc
import json
import logging
import os
import re
import sys
from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter, Namespace
from typing import MutableMapping, Mapping, NoReturn, Optional, Union, List, Dict, Tuple

import numpy as np
from dotmap import DotMap
from ruamel.yaml import YAML

import utils.constants as constants

logger = logging.getLogger(constants.APP_NAME)
__version__ = 0.2


class CLIError(Exception):
    """Generic exception to raise and log different fatal errors."""

    def __init__(self, msg):
        super().__init__(type(self))
        self.msg = f"ERROR: {msg}"

    def __str__(self):
        return self.msg

    def __unicode__(self):
        return self.msg


class EnvConfig(object):
    """
    Parses the target YAML configuration and prepares the framework to execute the requested processes
    """

    def __init__(self) -> None:
        self._config = None
        self.exec_config()

    @property
    def config(self) -> MutableMapping:
        return self._config

    def exec_config(self) -> None:
        # capture the config path from the run arguments
        # then process the yaml configuration file
        args = get_args()
        self.process_config(args)
        # create the experiments dirs
        create_dirs([loc for loc in self._config.experiment.dirs.values()])
        self.logging_config()

    def process_config(self, args: Namespace) -> None:
        # TODO: Add warnings for conflicting parameter settings
        self._config = get_config_from_yaml(args.config)
        self.cfg_dirs()
        if args.pred_inputs:
            pred_input_dict = get_config_from_json(args.pred_inputs)
            self._config.inference.pred_inputs = pred_input_dict['pred_inputs']
        self.cfg_instance()
        self.cfg_datasource()
        self.cfg_training()

    def define_arc_dir(self) -> None:
        file_suffix = "_debug" if self._config.experiment.debug.use_debug_dataset else "_converged_filtered"
        self._config.data_source.file_suffix = file_suffix
        ds_meta = Path(f"{self._config.experiment.dirs.tmp_data_dir}/ds_meta{file_suffix}.json")
        if (not ds_meta.exists()) or self._config.data_source.rebuild_dataset:
            self._config.experiment.dirs.arc_data_dir = self._config.experiment.dirs.arc_data_dir or \
                                                        f"{self._config.experiment.dirs.raw_data_dir}/arc/" \
                                                        f"{constants.APP_INSTANCE}"

    def cfg_dirs(self) -> None:
        self._config.experiment.dirs.base_dir = self._config.experiment.dirs.base_dir or os.environ['HOME']
        self._config.experiment.dirs.experiments_base_dir = self._config.experiment.dirs.experiments_base_dir or \
                                                            f"{self._config.experiment.dirs.base_dir}/experiments"
        self._config.experiment.dirs.raw_data_dir = self._config.experiment.dirs.raw_data_dir or \
                                                    f"{self._config.experiment.dirs.base_dir}/datasets"
        self._config.experiment.dirs.tmp_data_dir = self._config.experiment.dirs.tmp_data_dir or \
                                                    f"{self._config.experiment.dirs.raw_data_dir}/temp/" \
                                                    f"{constants.APP_NAME}"
        self.define_arc_dir()
        cust_model_cache_dir = f"{self._config.experiment.dirs.raw_data_dir}/model_cache/{constants.APP_NAME}"
        self._config.experiment.dirs.model_cache_dir = self._config.experiment.dirs.model_cache_dir or \
                                                       cust_model_cache_dir
        self._config.experiment.dirs.dcbot_log_dir = self._config.experiment.dirs.dcbot_log_dir or \
                                                         f"{self._config.experiment.dirs.experiments_base_dir}/dcbot"
        self._config.experiment.dirs.rpt_arc_dir = self._config.experiment.dirs.rpt_arc_dir or \
                                                   f"{self._config.experiment.dirs.base_dir}/repos/" \
                                                   f"{constants.APP_NAME}_history"

    def cfg_instance(self) -> None:
        if self._config.experiment.inference_ckpt and not os.path.exists(self._config.experiment.inference_ckpt):
            raise Exception(f"model checkpoint file not found at: {self._config.experiment.inference_ckpt}")
        if self._config.experiment.tweetbot.enabled:
            self._config.experiment.dirs.instance_log_dir = os.path.join(self._config.experiment.dirs.dcbot_log_dir,
                                                                         constants.APP_INSTANCE)
        else:
            self._config.experiment.dirs.instance_log_dir = \
                os.path.join(self._config.experiment.dirs.experiments_base_dir, constants.APP_NAME, "logs/",
                             constants.APP_INSTANCE)
        if not self._config.experiment.dataprep_only:
            self._config.experiment.dirs.inference_output_dir = \
                os.path.join(self._config.experiment.dirs.instance_log_dir, "inference_output")
        self._config.inference.asset_dir = \
            f"{os.path.join(os.path.split(self._config.experiment.dc_base)[0], 'assets')}"
        self._config.trainer.thaw_schedules_dir = \
            f"{os.path.join(os.path.split(self._config.experiment.dc_base)[0], 'configs/thaw_schedules')}"
        # don't ckpt dirs unless if in one of the training modes... on a full moon, in december...
        if (not self._config.experiment.inference_ckpt or (
                self._config.experiment.predict_only and self._config.inference.pred_inputs)) \
                and not self._config.experiment.dataprep_only:
            self._config.experiment.dirs.checkpoint_dir = self._config.experiment.dirs.checkpoint_dir or os.path.join(
                self._config.experiment.dirs.experiments_base_dir, constants.APP_NAME,
                "checkpoints/", constants.APP_INSTANCE)

    def cfg_datasource(self) -> None:
        if not self._config.experiment.db_functionality_enabled:
            logger.info('DB functionality currently disabled. If you would like to instantiate the DB '
                        'and enable full functionality, please see the repo README.')
        else:
            if constants.DEV_MODE and not self._config.data_source.db_conf:
                logger.error('In dev mode, db_conf must be explicitly specified to reduce risk of data loss. Exiting.')
                sys.exit(1)
            else:
                self._config.data_source.db_conf = \
                    self._config.data_source.db_conf or \
                    f'{constants.DEF_DB_PRJ_LOCATION}/{constants.DEF_DB_PRJ_NAME}/{constants.DEF_DB_CONF_NAME}'
        if not self._config.data_source.class_balancing_strategy or \
                self._config.data_source.class_balancing_strategy != "class_weights":
            self._config.data_source.class_balancing_strategy = "oversample_minority_classes"
        if self._config.data_source.class_balancing_strategy == "oversample_minority_classes" and \
                not self._config.data_source.sampling_weights:
            logger.error(f"You must provide a sampling_weights array for your classes when using oversampling")

    def cfg_training(self) -> None:
        if self._config.trainer.fine_tune_scheduler.thaw_schedule:
            self._config.trainer.fine_tune_scheduler.thaw_schedule = \
                os.path.join(self._config.trainer.thaw_schedules_dir,
                             self._config.trainer.fine_tune_scheduler.thaw_schedule)
        if self._config.trainer.restart_training_ckpt:
            if not os.path.exists(self._config.trainer.restart_training_ckpt):
                raise Exception(f"model checkpoint file not found at: {self._config.trainer.restart_training_ckpt}")
            p = re.compile("-(\d*)-")
            x = p.search(self._config.trainer.restart_training_ckpt)
            self._config.trainer.next_epoch = np.int(x.group(1)) + 1
            if self._config.trainer.next_epoch >= self._config.trainer.epochs:
                raise Exception(f"when resuming training, total training epochs for this config must be > the epochs "
                                f"you have already trained for ({self._config.trainer.next_epoch})")
        if self._config.trainer.add_summary:
            assert isinstance(self._config.trainer.add_summary, bool), 'add_graph must be a boolean'
        if len(self._config.inference.pred_inputs) > 0:
            if self._config.experiment.inference_ckpt:
                self._config.experiment.pred_inputs = True
            else:
                raise ValueError("An existing model checkpoint must be provided in order to perform inference")
        if self._config.trainer.earlystopping.monitor_metric:
            if not any(self._config.trainer.earlystopping.monitor_metric in m for m in constants.SUPPORTED_METRICS):
                raise Exception(f"specified earlystopping metric {self._config.trainer.earlystopping.monitor_metric} "
                                f"not currently supported must be in the following "
                                f"metrics: {','.join([m for m in constants.SUPPORTED_METRICS])}")
            else:
                if self._config.trainer.earlystopping.patience < 1:
                    raise ValueError("metric improvement patience should be positive integer.")

    # noinspection PyShadowingNames
    def logging_config(self) -> None:
        file_handler, console_handler = self.conf_handlers()
        logger = logging.getLogger(constants.APP_NAME)
        if self._config.experiment.debug.debug_enabled:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.info(f"Starting {constants.APP_NAME} logger")

    def conf_handlers(self) -> Tuple[logging.FileHandler, logging.StreamHandler]:
        # attach handlers only to the root logger and allow propagation to handle
        # N.B. tensorflow uses absl and attaches to the root logger as well
        formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s: %(message)s")
        file_handler = logging.FileHandler(f'{self._config.experiment.dirs.instance_log_dir}/{constants.APP_NAME}.log')
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        return file_handler, console_handler


def get_args() -> Namespace:
    program_version = f"{__version__}"
    program_version_message = f"{constants.APP_NAME} ({program_version})"
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    # noinspection PyTypeChecker
    parser = ArgumentParser(description=program_shortdesc, formatter_class=RawDescriptionHelpFormatter)
    ex_group = parser.add_mutually_exclusive_group()
    parser.add_argument('--config', dest='config', help='a yaml config file', default='')
    ex_group.add_argument('--pred_inputs', dest='pred_inputs', help='pass inference inputs json file')
    parser.add_argument('-v', '--version', action='version', version=program_version_message)
    args = parser.parse_args()
    if args.config and not os.path.exists(args.config):
        raise CLIError(f"A valid config file was not found at: {args.config}")
    if args.pred_inputs and not os.path.exists(args.pred_inputs):
        raise CLIError(f"A valid inference input file was not found at: {args.pred_inputs}")
    return args


def get_config_from_json(json_file: str) -> Dict:
    """
    Get the config from a json file
    :param json_file:
    :return: config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    return config_dict


def get_config_from_yaml(yaml_file: str = None) -> DotMap:
    """
    Args:
        yaml_file:

    Returns:
        config(namespace) or config(dictionary)
    """
    # load current default config
    os_dc_base = os.getenv('DC_BASE')
    dc_base = f'{os_dc_base}/configs' if os_dc_base else f'{os.getcwd()}/configs'
    defaults_file = os.path.join(dc_base, constants.DEFAULT_CONFIG_NAME)
    defaults_sql_file = os.path.join(dc_base, constants.DEFAULT_CONFIG_SQL_NAME)
    yaml = YAML()
    with open(defaults_file, 'r') as df:
        config_dict = yaml.load(df)
    # parse the configurations from the config json file provided
    if yaml_file:
        with open(yaml_file, 'r') as config_file:
            instance_config_dict = yaml.load(config_file)
        # update keys of default dict with instance-config overrides
        config_dict = recursive_merge_dict(config_dict, instance_config_dict)
    if config_dict['experiment']['db_functionality_enabled']:  # if we're not running in db mode, no need for sql config
        with open(defaults_sql_file, 'r') as df:
            config_sql_dict = yaml.load(df)
        config_dict = recursive_merge_dict(config_dict, config_sql_dict)
    # convert the dictionary to a namespace using DotMap
    config = DotMap(config_dict)
    config.experiment.dc_base = dc_base
    return config


def recursive_merge_dict(orig: MutableMapping, new: Union[MutableMapping, Mapping]) -> MutableMapping:
    for k, v in new.items():
        if isinstance(v, collections.abc.Mapping):
            orig[k] = recursive_merge_dict(orig.get(k, {}), v)
        else:
            orig[k] = v
    return orig


def create_dirs(dirs: List) -> Optional[NoReturn]:
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success 1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except OSError as err:
        print(f"Creating directories error: {err}")
        sys.exit(0)
