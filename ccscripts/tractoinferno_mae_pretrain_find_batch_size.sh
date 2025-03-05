#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --time=0-00:15:00
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --mail-user=antoine.theberge@usherbrooke.ca
#SBATCH --mail-type=ALL

cd /home/thea1603/projects/def-pmjodoin/thea1603/SAMTrack
module load StdEnv python/3.10 httpproxy
source .env/bin/activate

pwd

rsync -rltv /home/thea1603/projects/def-pmjodoin/thea1603/braindata/samtrack/datasets/hcp_1200.hdf5 $SLURM_TMPDIR/

# srun python mae_fodf_pretrain.py ViT-H hcp_1200 batchsize $SLURM_TMPDIR/hcp_1200.hdf5 --lr 5e-4 --mask_ratio 0.75 --epochs 3 --model_loss sql2 --batch-size 32 --devices 4 --patch_size 8 --num_workers 36 --model_size huge --img_size 96 --mean_mask_loss --decoder_depth 2
# 
# srun python mae_fodf_pretrain.py ViT-H hcp_1200 batchsize $SLURM_TMPDIR/hcp_1200.hdf5 --lr 5e-4 --mask_ratio 0.75 --epochs 3 --model_loss sql2 --batch-size 24 --devices 4 --patch_size 8 --num_workers 36 --model_size huge --img_size 96 --mean_mask_loss --decoder_depth 2
# 
# srun python mae_fodf_pretrain.py ViT-H hcp_1200 batchsize $SLURM_TMPDIR/hcp_1200.hdf5 --lr 5e-4 --mask_ratio 0.75 --epochs 3 --model_loss sql2 --batch-size 16 --devices 4 --patch_size 8 --num_workers 36 --model_size huge --img_size 96 --mean_mask_loss --decoder_depth 2
# 
# srun python mae_fodf_pretrain.py ViT-H hcp_1200 batchsize $SLURM_TMPDIR/hcp_1200.hdf5 --lr 5e-4 --mask_ratio 0.75 --epochs 3 --model_loss sql2 --batch-size 8 --devices 4 --patch_size 8 --num_workers 36 --model_size huge --img_size 96 --mean_mask_loss --decoder_depth 2

# srun python mae_fodf_pretrain.py ViT-H hcp_1200 batchsize $SLURM_TMPDIR/hcp_1200.hdf5 --lr 5e-4 --mask_ratio 0.75 --epochs 3 --model_loss sql2 --batch-size 8 --devices 4 --patch_size 8 --num_workers 36 --model_size huge --img_size 96 --mean_mask_loss --decoder_depth 2

srun python mae_fodf_pretrain.py ViT-H hcp_1200 batchsize $SLURM_TMPDIR/hcp_1200.hdf5 --lr 5e-4 --mask_ratio 0.75 --epochs 3 --model_loss sql2 --batch-size 6 --devices 4 --patch_size 8 --num_workers 36 --model_size huge --img_size 96 --mean_mask_loss --decoder_depth 2
