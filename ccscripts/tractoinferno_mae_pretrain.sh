#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --mail-user=antoine.theberge@usherbrooke.ca
#SBATCH --mail-type=ALL

cd /home/thea1603/projects/def-pmjodoin/thea1603/SAMTrack
module load StdEnv python/3.10 httpproxy
source .env/bin/activate

pwd

rsync -rltv /home/thea1603/projects/def-pmjodoin/thea1603/braindata/samtrack/datasets/hcp_1200.hdf5 $SLURM_TMPDIR/

srun python mae_fodf_pretrain.py ViT-H_conv hcp_1200 08_mask $SLURM_TMPDIR/hcp_1200.hdf5 --lr 5e-4 --mask_ratio 0.8 --epochs 800 --model_loss sql2 --batch-size 4 --devices 4 --patch_size 8 --num_workers 20 --model_size huge --img_size 96 --mean_mask_loss --decoder_depth 2
