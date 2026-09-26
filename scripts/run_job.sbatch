#!/bin/bash
# Supply your own site-specific resources on the sbatch command line.
# Example only (replace all placeholders with values verified at your site):
# sbatch --partition=<PARTITION> --account=<ACCOUNT> --gpus-per-node=1 \
#   --cpus-per-task=4 --mem=48G --time=<WALLTIME> --output=<LOG_PATH> \
#   scripts/run_job.sbatch python -m engintervene.train --model-index 0 --condition industry --mode train
set -euo pipefail
if [[ "$#" -eq 0 ]]; then
    printf '%s\n' 'Pass the full command after this script.' >&2
    exit 2
fi
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONDONTWRITEBYTECODE=1
srun --cpu-bind=none "$@"
