echo $(pwd)
src=/home/local/USHERBROOKE/thea1603/data/datasets/tractoinferno/sourcedata/testset
dest=/home/local/USHERBROOKE/thea1603/data/samtrack/results/Tractoinferno_SAMU_72

bundles=(AF_left AF_right ATR_left ATR_right CA CC CC_1 CC_2 CC_3 CC_4 CC_5 CC_6 CC_7 CG_left CG_right CST_left CST_right FPT_left FPT_right FX_left FX_right ICP_left ICP_right IFO_left IFO_right ILF_left ILF_right MCP MLF_left MLF_right OR_left OR_right POPT_left POPT_right SCP_left SCP_right SLF_III_left SLF_III_right SLF_II_left SLF_II_right SLF_I_left SLF_I_right STR_left STR_right ST_FO_left ST_FO_right ST_OCC_left ST_OCC_right ST_PAR_left ST_PAR_right ST_POSTC_left ST_POSTC_right ST_PREC_left ST_PREC_right ST_PREF_left ST_PREF_right ST_PREM_left ST_PREM_right T_OCC_left T_OCC_right T_PAR_left T_PAR_right T_POSTC_left T_POSTC_right T_PREC_left T_PREC_right T_PREF_left T_PREF_right T_PREM_left T_PREM_right UF_left UF_right)

for sub in $(ls -d ${src}/sub-*/); do
  base_sub=$(basename $sub /)
  echo $sub
  fodf=${src}/${base_sub}/fodf/${base_sub}__fodf.nii.gz
  wm=${src}/${base_sub}/mask/${base_sub}__mask_wm.nii.gz
  out=${dest}/${base_sub}/${base_sub}_

  mkdir -p ${dest}/${base_sub}

  python samu_predict.py ${fodf} ${wm} ${out} --checkpoint samu/hcp105/ds/samu/db5475bceeef4fb198e74ddf799bf68d/checkpoints/epoch=169-step=96390.ckpt --img_size 128 --bundles ${bundles[@]}

done
