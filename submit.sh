#!/bin/sh

### Job queue to use (options: batch)
#SBATCH --partition=batch

### Amount of nodes to use
#SBATCH --nodes=1

### Processes per node
#SBATCH --ntasks-per-node=1

### Available cores per node (1-12)
#SBATCH --cpus-per-task=1

### execution time. Format: days-hours:minutes:seconds -- Max: three days
#SBATCH --time 1-00:00:00

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/10.0/extras/CUPTI/lib64/:/opt/cuda/10.0/lib64:/opt/cudnn/v7.6-cu10.0/
export CUDA_HOME=/opt/cuda/10.0
export CUDA_VISIBLE_DEVICES=1

### Enqueue job
srun -o logs/%j.out -e logs/%j.err python trabajo_final.py
