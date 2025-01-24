echo $(pwd)
src=/home/local/USHERBROOKE/thea1603/data/datasets/tractoinferno/sourcedata/testset
dest=/home/local/USHERBROOKE/thea1603/data/samtrack/results/Tractoinferno_SAMU_32
mkdir -p ${dest}

for sub in $(ls -d ${src}/sub-*/); do
  base_sub=$(basename $sub /)
  echo $sub
  fodf=${src}/${base_sub}/fodf/${base_sub}__fodf.nii.gz
  wm=${src}/${base_sub}/mask/${base_sub}__mask_wm.nii.gz
  out=${dest}/${base_sub}/${base_sub}_

  mkdir -p ${dest}/${base_sub}

  python samu_predict.py ${fodf} ${wm} ${out} --checkpoint samu/tractoinferno/ds/samu/54a74f000ff743e6b5149cc3c7ca9ea8/checkpoints/epoch=159-step=97120.ckpt --img_size 128

done
