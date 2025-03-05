#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
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

# rsync -rltv /home/thea1603/projects/def-pmjodoin/thea1603/braindata/samtrack/datasets/samtrack_hcp105_train_128.hdf5 $SLURM_TMPDIR/
# rsync -rltv /home/thea1603/projects/def-pmjodoin/thea1603/braindata/samtrack/datasets/samtrack_hcp105_valid_128.hdf5 $SLURM_TMPDIR/
rsync -rltv /home/thea1603/projects/def-pmjodoin/thea1603/braindata/samtrack/hcp1200_pretrain.hdf5 $SLURM_TMPDIR/

srun python samu_train.py scilseg hcp105 hcp1200_pretrain_0 $SLURM_TMPDIR/hcp1200_pretrain.hdf5  --batch-size 2 --epochs 1000 --lr 0.0001 --devices 4 --wm_drop_ratio 0 --prompt_strategy attention --volume_size 128 --warmup 50 --pretrain --ds
