#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-user=antoine.theberge@usherbrooke.ca
#SBATCH --mail-type=ALL

cd /home/thea1603/projects/def-pmjodoin/thea1603/SAMTrack
module load StdEnv python/3.10 httpproxy
source .env/bin/activate

pwd

rsync -rltv /home/thea1603/projects/def-pmjodoin/thea1603/braindata/samtrack/datasets/samtrack_tractoinferno_train_128.hdf5 $SLURM_TMPDIR/
rsync -rltv /home/thea1603/projects/def-pmjodoin/thea1603/braindata/samtrack/datasets/samtrack_tractoinferno_valid_128.hdf5 $SLURM_TMPDIR/
rsync -rltv /home/thea1603/projects/def-pmjodoin/thea1603/braindata/samtrack/datasets/samtrack_tractoinferno_test_128.hdf5 $SLURM_TMPDIR/

srun python samu_train.py samu tractoinferno ds $SLURM_TMPDIR/samtrack_tractoinferno_train_128.hdf5 $SLURM_TMPDIR/samtrack_tractoinferno_valid_128.hdf5 $SLURM_TMPDIR/samtrack_tractoinferno_test_128.hdf5 --batch-size 2 --epochs 1000 --lr 0.0001 --devices 4 --wm_drop_ratio 0 --prompt_strategy attention --n_bundles 32
