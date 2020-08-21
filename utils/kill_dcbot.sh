#!/bin/bash
target_daemon=$1
kill_pid ()
{
kill_pid=`cat $1| cut -d, -f1 | egrep -o "[[:digit:]]*"`
if [ -v kill_pid ]
then
  echo "attempting to kill pid ${kill_pid}"
  kill -s 9 ${kill_pid}
  if [ $? -eq 0  ]; then echo "kill succeeded!";  rm ${lock_file}; else echo "failed to kill pid ${kill_pid}"; fi
else
  [ -z "$target_daemon" ] && echo "could not find/kill dcbot associated with default dcbot, is it started?" || echo "could not find/kill dcbot associated with ${target_daemon}, is it started?"
  unset kill_pid
  exit 1
fi
}
if [ -z "$target_daemon" ]
then
  lock_file="${HOME}/dcbot.lock"
  [ -f "${lock_file}" ] && kill_pid $lock_file || echo "lock file ${lock_file} not found. Check arguments."
else
  lock_file="${HOME}/${target_daemon}_dcbot.lock"
  [ -f "${lock_file}" ] && kill_pid $lock_file || echo "lock file ${lock_file} not found. Check arguments."
fi

