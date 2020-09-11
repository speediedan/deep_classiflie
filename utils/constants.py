"""
constants not exposed in config (not generally changed)
"""
import datetime
import os

curr_path = os.path.dirname(os.path.realpath(__file__)).rsplit('/', -1)
curr_parent, curr_base = curr_path[-3], curr_path[-2]
DEF_PRJ_NAME = "deep_classiflie"
DEF_DB_PRJ_NAME = "deep_classiflie_db"
DEV_MODE = True if curr_base != DEF_PRJ_NAME else False
if DEV_MODE:
    LOCK_FILE = f"{os.environ['HOME']}/{curr_base}_dcbot.lock"
    DC_PREDICTIONS_SUBDOMAIN = "predictions-dev"
else:
    LOCK_FILE = f"{os.environ['HOME']}/dcbot.lock"
    DC_PREDICTIONS_SUBDOMAIN = "predictions"
LOCK_FILE = f"{os.environ['HOME']}/{curr_base}_dcbot.lock" if DEV_MODE else f"{os.environ['HOME']}/dcbot.lock"
APP_NAME = curr_base
CPU_COUNT = os.cpu_count()
APP_INSTANCE = f'{datetime.datetime.now():%Y%m%d%H%M%S}'
DB_WARNING_START = "DB functionality is currently disabled (experiment.db_functionality_enabled: False)."
DB_WARNING_END = "cannot be executed w/o first configuring the db. Please see the repo readme for more information. " \
                 "Aborting..."
DEFAULT_CONFIG_NAME = "config_defaults.yaml"
DEFAULT_CONFIG_SQL_NAME = "config_defaults_sql.yaml"
LOCAL_INFSVC_PUB_CACHE_NAME = "dc_infsvc_pub_cache.json"
PINATA_PINJSON_ENDPOINT = "https://api.pinata.cloud/pinning/pinJSONToIPFS"
PINATA_UNPINJSON_ENDPOINT = "https://api.pinata.cloud/pinning/unpin"
CLOUDFLARE_DC_DNS_ENDPOINT = "https://api.cloudflare.com/client/v4/zones"
# default db project location is to be a sibling dir of this project
DEF_DB_PRJ_LOCATION = os.path.abspath(f'{curr_parent}/../../')
DEF_DB_CONF_NAME = f"{DEF_DB_PRJ_NAME}.yaml"
MEM_FACTOR = 1024 * 1024 * 1024
MEM_MAG = "GB"
CKPT_EXT = ".pt"
STMT_EMBED_CACHE_NAME = "stmt_embed_cache.pt"
PRED_EXP_CACHE_NAME = "pred_exp_cache.pt"
PERF_EXP_CACHE_NAME = "perf_exp_cache.pt"
BASE_FALSEHOOD_EMBED_CACHE_NAME = "base_falsehood_embed_cache.pt"
BASE_TRUTH_EMBED_CACHE_NAME = "base_truth_embed_cache.pt"
FALSEHOOD_EMBED_CACHE_NAME = "falsehood_embed_cache.pt"
TRUTH_EMBED_CACHE_NAME = "truth_embed_cache.pt"
TWEET_MODEL_PERF_CACHE_NAME = "tweet_model_perf_cache.pt"
NONTWEET_MODEL_PERF_CACHE_NAME = "nontweet_model_perf_cache.pt"
GLOBAL_MODEL_PERF_CACHE_NAME = "global_model_perf_cache.pt"
SUPPORTED_METRICS = ["val_loss", "acc", "mcc"]
REPORT_TYPES = ['model_rpt_gt']
MAX_STMT_LEN = 30
TWEET_STATUS_MSG = """Explore recent statements & learn about Deep Classiflie: https://deepclassiflie.org  
GitHub: https://github.com/speediedan/deep_classiflie"""