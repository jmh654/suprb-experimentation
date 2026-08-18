#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --job-name=mlrun_stat_analysis
#SBATCH --output=stdout/%x-%j.out

JOB_DIR=/home/huettiju/data/oc-compute03/suprb-experimentation

cd ${JOB_DIR}

nix develop --no-pure-eval --command \
  env PYTHONPATH=${JOB_DIR}/runs/mybenchmark:${JOB_DIR} \
  uv run python ${JOB_DIR}/runs/mybenchmark/mlrun4statAnalisys.py

#uv run python ${JOB_DIR}/runs/mybenchmark/mlflow2csv
