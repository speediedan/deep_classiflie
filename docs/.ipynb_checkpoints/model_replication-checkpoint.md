### Model Replication
N.B. before you begin, the core external dependency is admin access to a mariadb or mysql DB

1. Clone deep_classiflie and deep_classiflie_db (make them peer directories if you want to minimize configuration)
```shell
git clone https://github.com/speediedan/deep_classiflie.git
git clone https://github.com/speediedan/deep_classiflie_db.git
```
2. install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda) if necessary. Then create and activate deep_classiflie virtual env:
```shell
conda env create -f ./deep_classiflie/utils/deep_classiflie.yml
conda activate deep_classiflie
```
3. clone captum and HuggingFace's transformers repos. Install transformers binaries.:
```shell
git clone https://github.com/pytorch/captum.git
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```
4. (temporarily required) Testing of this alpha release occurred before native AMP was integrated into Pytorch with release 1.6. As such, native apex installation is temporarily (as of 2020.08.18) required to replicate the model. Switching from the native AMP api to the pytorch integrated one is planned as part of issue #999 which should obviate the need to install native apex.
```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip uninstall apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
5. [Install mariadb](https://mariadb.com/kb/en/getting-installing-and-upgrading-mariadb/) or mysql DB if necessary.
6. These are the relevant DB configuration settings used for the current release of DeepClassiflie's backend. Divergence from this configuration has not been tested and may result in unexpected behavior.
```mysql
collation-server = utf8mb4_unicode_ci
init-connect='SET NAMES utf8mb4'
character-set-server = utf8mb4
sql_mode = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION,ANSI_QUOTES'
transaction-isolation = READ-COMMITTED
```
7. copy/update relevant Deep Classiflie config file to $HOME dir
```shell
cp ./deep_classiflie_db/db_setup/.dc_config.example ~
mv .dc_config.example .dc_config
vi .dc_config
```

```shell
# configure values appropriate to your environment and move to $HOME
# Sorry I haven't had a chance to write a setup config script for this yet...

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"

export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="${PYTHONPATH}:${HOME}/repos/edification/deep_classiflie:${HOME}/repos/captum:${HOME}/repos/transformers:${HOME}/repos/edification/deep_classiflie_db"

export DC_BASE="$HOME/repos/edification/deep_classiflie"

export DCDB_BASE="$HOME/repos/edification/deep_classiflie_db"
export PYTHONUNBUFFERED=1
export DCDB_PASS="dcbotpasshere"
export DCDB_USER="dcbot"
export DCDB_HOST="hostgoeshere"
export DCDB_NAME="deep_classiflie"
```

8. execute Deep Classiflie DB backend initialization script:
<img src="assets/dc_schema_build.gif" alt="Deep Classiflie logo" align="left" />

Ensure you have access to a DB user with administrator privs. "admin" in the case above.

```shell
cd deep_classiflie_db/db_setup
./deep_classiflie_db_setup.sh deep_classiflie
```

9. login to the backend db and seed historical tweets (necessary as only most recent 3200 can currently be retrieved directly from twitter)
```mysql
mysql -u dcbot -p
use deep_classiflie
source dcbot_tweets_init_20200814.sql
```

10. copy over relevant base model weights to specified model_cache_dir:

```shell
# model_cache_dir default found in configs/config_defaults.yaml
# it defaults to $HOME/datasets/model_cache/deep_classiflie/
cd {PATH_TO_DEEP_CLASSIFLIE_BASE}/deep_classiflie/assets/
cp albert-base-v2-pytorch_model.bin albert-base-v2-spiece.model {MODEL_CACHE_DIR}/
```

11. Run deep_classiflie.py with the provided config necessary to download the raw data from the relevant data sources (factba.se, twitter, washington post), execute the data processing pipeline and generate the dataset collection.
cd deep_classiflie
./deep_classiflie.py --config "{PATH_TO_DEEP_CLASSIFLIE_BASE}/configs/dataprep_only.yaml"
See relevant process diagrams to better understand the dataset generation pipeline and process.

    * While I have set seeds for the majority of randomized processes in the data pipeline, there are a couple points in the pipeline that remain non-deterministic at the moment (see issue #). As such, the dataset generation log messages should approximate those below, but variation within 1% is expected.

    ```
    (...lots of initial data download/parsing message above...) 
    2020-08-14 16:55:22,165:deep_classiflie:INFO: Proceeding with uninitialized base model to generate dist-based duplicate filter
    2020-08-14 16:55:22,501:deep_classiflie:INFO: Predictions from model weights: 
    2020-08-14 16:57:14,215:deep_classiflie:INFO: Generated 385220 candidates for false truth analysis
    2020-08-14 16:57:15,143:deep_classiflie:INFO: Deleted 7073 'truths' from truths table based on similarity with falsehoods enumerated in base_false_truth_del_cands
    2020-08-14 16:57:30,181:deep_classiflie:INFO: saved 50873 rows of a transformed truth distribution to db
    2020-08-14 16:57:30,192:deep_classiflie:DEBUG: DB connection obtained: <mysql.connector.pooling.PooledMySQLConnection object at 0x7f8216056e50>
    2020-08-14 16:57:30,220:deep_classiflie:DEBUG: DB connection closed: <mysql.connector.pooling.PooledMySQLConnection object at 0x7f8216056e50>
    2020-08-14 16:57:30,221:deep_classiflie:INFO: Building a balanced dataset from the following raw class data:
    2020-08-14 16:57:30,221:deep_classiflie:INFO: Label True: 50873 records
    2020-08-14 16:57:30,221:deep_classiflie:INFO: Label False: 19261 records
    2020-08-14 16:57:49,281:deep_classiflie:INFO: Saving features into cached file /home/speediedan/datasets/temp/deep_classiflie/train_converged_filtered.pkl
    2020-08-14 16:58:06,552:deep_classiflie:INFO: Saving features into cached file /home/speediedan/datasets/temp/deep_classiflie/val_converged_filtered.pkl
    2020-08-14 16:58:11,714:deep_classiflie:INFO: Saving features into cached file /home/speediedan/datasets/temp/deep_classiflie/test_converged_filtered.pkl
    2020-08-14 16:58:14,331:deep_classiflie:DEBUG: Metadata update complete, 1 record(s) affected.
    ...
    ```
12. Recursively train the deep classiflie POC model:
```shell
cd deep_classiflie
./deep_classiflie.py --config "{PATH_TO_DEEP_CLASSIFLIE_BASE}/configs/train_albertbase.yaml"
```

13. Generate an swa checkpoint (current release was built using swa torchcontrib module but will switch to the now-integrated pytorch swa api in the next release):
```shell
cd deep_classiflie
./deep_classiflie.py --config "{PATH_TO_DEEP_CLASSIFLIE_BASE}/configs/gen_swa_ckpt.yaml"
```

14. Generate model analysis report(s) using the generated swa checkpoint:
```shell
# NOTE, swa checkpoint generated in previous step must be added to gen_report.yaml
cd deep_classiflie
./deep_classiflie.py --config "{PATH_TO_DEEP_CLASSIFLIE_BASE}/configs/gen_report.yaml"
```

15. Generate model analysis dashboards:
```shell
# NOTE, swa checkpoint generated in previous step must be added to gen_dashboards.yaml
cd deep_classiflie
./deep_classiflie.py --config "{PATH_TO_DEEP_CLASSIFLIE_BASE}/configs/gen_dashboards.yaml"
```
