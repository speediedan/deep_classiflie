"""
deep_classiflie: Deep Classiflie is a framework for developing ML models that bolster fact-checking efficiency.

The initial alpha release of Deep Classiflie generates/analyzes a model that continuously classifies a single
individual's statements (Donald Trump)<sup id="a1">[1](#f1)</sup> using a single ground truth labeling source
(The Washington Post). For statements the model deems most likely to be labeled falsehoods.
 (see deepclassiflie.org for more detail), the [@DeepClassiflie](https://twitter.com/DeepClassiflie) twitter bot tweets
 out a statement analysis and model interpretation "report"
@author: Dan Dale, @speediedan
"""
import logging
import os
import sys
from typing import MutableMapping, NoReturn, Optional


import utils.constants as constants
from dataprep.dataprep import DatasetCollection
from utils.core_utils import create_lock_file
from utils.dc_tweetbot import DCTweetBot
from utils.envconfig import EnvConfig
from analysis.inference import Inference
from analysis.model_analysis_rpt import ModelAnalysisRpt
from training.trainer import Trainer


logger = logging.getLogger(constants.APP_NAME)


def main() -> Optional[NoReturn]:
    config = EnvConfig().config
    if config.experiment.dataprep_only:
        _ = DatasetCollection(config)
    elif config.experiment.predict_only and config.inference.pred_inputs:
        Inference(config).init_predict()
    elif config.experiment.tweetbot.enabled:
        init_tweetbot(config)
    elif config.inference.report_mode:
        if not config.experiment.db_functionality_enabled:
            logger.error(f"{constants.DB_WARNING_START} Model analysis reports {constants.DB_WARNING_END}")
            sys.exit(0)
        ModelAnalysisRpt(config)
    else:
        core_flow(config)


def init_tweetbot(config: MutableMapping) -> NoReturn:
    lock_file = None
    try:
        if not config.experiment.db_functionality_enabled:
            logger.error(f"{constants.DB_WARNING_START} The tweetbot {constants.DB_WARNING_END}")
            sys.exit(0)
        lock_file = create_lock_file()
        DCTweetBot(config)
    except KeyboardInterrupt:
        logger.warning('Interrupted bot, removing lock file and exiting...')
        os.remove(lock_file)
        sys.exit(0)


def core_flow(config: MutableMapping) -> None:
    dataset = DatasetCollection(config)
    trainer = Trainer(dataset, config)
    if config.experiment.inference_ckpt:
        # testing mode takes precedence of training if both ckpts specified
        logger.info(f'Testing model weights loaded from {config.experiment.inference_ckpt}...')
        trainer.init_test(config.experiment.inference_ckpt)
    elif config.trainer.restart_training_ckpt:
        # restarting training takes precedence over just building custom swa checkpoints
        logger.info(f'Restarting model training from {config.trainer.restart_training_ckpt}...')
        trainer.train(config.trainer.restart_training_ckpt)
    elif config.trainer.build_swa_from_ckpts:
        logger.info(f'Building swa checkpoint from specified ckpts: {config.trainer.build_swa_from_ckpts}...')
        swa_ckpt = trainer.swa_ckpt_build(mode="custom", ckpt_list=config.trainer.build_swa_from_ckpts)
        logger.info(f'Successfully built SWA checkpoint ({swa_ckpt}) from provided list of checkpoints, '
                    f'proceeding with test')
        trainer.init_test(swa_ckpt)
    else:
        logger.info('Starting model training from scratch...')
        trainer.train()


if __name__ == '__main__':
    repo_base = None
    main()
