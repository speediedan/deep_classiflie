import json
import logging
import os
import sys
import pickle
import datetime
from typing import MutableMapping, NoReturn, Any, Union, Dict, List, Tuple
from pathlib import Path

import psutil
import torch

import utils.constants as constants

logger = logging.getLogger(constants.APP_NAME)


def device_config(config: MutableMapping) -> torch.device:
    # determine CPU vs GPU mode (training is not currently distributed across multiple devices)
    if torch.cuda.is_available():
        if config.experiment.debug.debug_enabled:
            log_device_info()
        # N.B. Another option here would be to mask the default cuda device using the CUDA_VISIBLE_DEVICES env
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:1") if config.experiment.tweetbot.enabled and not constants.DEV_MODE \
                else torch.device("cuda:0")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if device.type == "cpu":
        logger.debug(
            f'{constants.CPU_COUNT} cpus available for training, workers specified is {config.trainer.workers}')
    else:
        logger.debug(
            f'Using the following GPU device. Device node {device.index}, '
            f'name: {torch.cuda.get_device_name(device)}')
    return device


def save_bokeh_tags(bokeh_base: os.PathLike, script: str, script_file: os.PathLike, tags: Tuple,
                    tag_files: List) -> None:
    with open(Path(bokeh_base) / script_file, 'w') as file:
        file.write(script)
    for t, f in zip(tags, tag_files):
        if isinstance(t, Tuple):
            with open(Path(bokeh_base) / f, 'w') as file:
                for subt in t:
                    file.write(subt)
        else:
            with open(Path(bokeh_base) / f, 'w') as file:
                file.write(t)


def create_lock_file() -> Union[str, NoReturn]:
    lock_file = constants.LOCK_FILE
    if not os.path.exists(lock_file):
        with open(lock_file, 'w') as lf:
            lf.write(f'{psutil.Process()}')
    else:
        logger.error(f'Lock file "{lock_file}" exists, first stop existing process')
        sys.exit(0)
    return lock_file


def log_device_info() -> None:
    for d in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(d)
        major = capability[0]
        minor = capability[1]
        name = torch.cuda.get_device_name(d)
        logger.debug(f'Device id {d} is named {name} and has CUDA capability: {major}.{minor}')


def log_config(config: MutableMapping, model_name: str = None) -> None:
    model_name = model_name or "Not specified"
    expdir = config.experiment.dirs.instance_log_dir
    filename = f'{expdir}/{constants.APP_NAME}_{constants.APP_INSTANCE}.json'
    json_data = json.dumps(config, indent=4)
    with open(filename, 'a') as file:
        file.write(f"Using the model: {model_name}\n")
        for line in json_data.split('\n'):
            file.write(line + '\n')


def pickle_obj(path: str, targ_obj: Any) -> str:
    with open(path, "wb+") as f_output:
        pickle.dump(targ_obj, f_output)
    return path


def to_json(var: Any) -> str:
    json_data = json.dumps(var, indent=4, default=datetime_serde)
    return json_data


def save_json(var: Any, filename: str, already_json: bool = False) -> None:
    json_data = json.dumps(var, indent=4, default=datetime_serde) if not already_json else var
    with open(filename, 'w') as file:
        for line in json_data.split('\n'):
            file.write(line + '\n')


def datetime_serde(obj: Any) -> str:
    if isinstance(obj, datetime.date):
        return obj.__str__()


def load_json(filename: os.PathLike) -> Dict:
    json_data = None
    # open the file as read only
    try:
        with open(filename, 'r') as f:
            # read all text
            json_data = json.load(f)
    except FileNotFoundError:
        logger.debug(f'No json data found at {str}, proceeding w/o the cache may result in an error')
    return json_data
