#!/usr/bin/env bash
# Run one llama.cpp router on a GPU node for the life of the job, registered to the gateway.
#   sbatch slurm/serve.sbatch                       # edit the #SBATCH lines for your cluster
#   LLM_PRESET=8x-24g sbatch slurm/serve.sbatch     # force a preset instead of picking by GPU count
#SBATCH --job-name=llm-serve
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=5-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

bin/llm join                      # build if needed, start the router + heartbeat (both in tmux)
trap 'bin/llm leave' EXIT TERM    # deregister cleanly when the job ends or is cancelled

# keep the allocation alive while the router runs
while tmux has-session -t =llm 2>/dev/null; do sleep 60; done
