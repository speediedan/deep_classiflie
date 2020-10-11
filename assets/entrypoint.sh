#!/bin/bash --login
set -e
conda activate $TARGET_ENV
exec "$@"
