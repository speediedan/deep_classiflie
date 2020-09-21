#!/bin/bash
source ${HOME}/.bashrc
cd `dirname "$(readlink -f "$0")"`
curr_branch=`git branch | grep '* ' | awk '{print $2}'`
# N.B. if working DEV mode, the branch name should equal the corresponding working tree name
if [ "${curr_branch}" == "master" ]
then
    echo "Starting the deep classiflie inference service in non-dev mode"
    source "${HOME}/.dc_config"
    lock_file=${HOME}/dcbot.lock
    bot_log_name="deep_classiflie_infsvc"
else
    echo "Starting the deep classiflie inference service in dev mode"
    source "${HOME}/.${curr_branch}_config"
    lock_file=${HOME}/${curr_branch}_dcbot.lock
    bot_log_name="${curr_branch}_infsvc"
fi
[ -f $lock_file ] && echo "Lock file ${lock_file} exists, abandoning daemon startup" && exit 0
target_env=$1
conda activate $target_env
prev_log=$HOME/$bot_log_name.out
if test -f $prev_log
then
    d=`date +%Y%m%d%H%M%S`
    mv $prev_log ${prev_log}_${d}.bkp
fi
svc_mode=$2
if [ $svc_mode == "svc" ]
then
  /opt/anaconda/envs/${target_env}/bin/python ${DC_BASE}/deep_classiflie.py --config "${DC_BASE}/configs/infsvc.yaml" 1>"${HOME}/${bot_log_name}.out" 2>&1
else
  nohup /opt/anaconda/envs/${target_env}/bin/python ${DC_BASE}/deep_classiflie.py --config "${DC_BASE}/configs/infsvc.yaml" 1>"${HOME}/${bot_log_name}.out" 2>&1 &
fi