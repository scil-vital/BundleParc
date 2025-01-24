echo $(pwd)
src=/home/local/USHERBROOKE/thea1603/braindata/databases/Penthera_3T/derivatives/TractoFlow/output/results
dest=/home/local/USHERBROOKE/thea1603/data/samtrack/results/Penthera3T_SAMU_71_pretrain

bundles=(AF_left AF_right ATR_left ATR_right CA CC_1 CC_2 CC_3 CC_4 CC_5 CC_6 CC_7 CG_left CG_right CST_left CST_right FPT_left FPT_right FX_left FX_right ICP_left ICP_right IFO_left IFO_right ILF_left ILF_right MCP MLF_left MLF_right OR_left OR_right POPT_left POPT_right SCP_left SCP_right SLF_III_left SLF_III_right SLF_II_left SLF_II_right SLF_I_left SLF_I_right STR_left STR_right ST_FO_left ST_FO_right ST_OCC_left ST_OCC_right ST_PAR_left ST_PAR_right ST_POSTC_left ST_POSTC_right ST_PREC_left ST_PREC_right ST_PREF_left ST_PREF_right ST_PREM_left ST_PREM_right T_OCC_left T_OCC_right T_PAR_left T_PAR_right T_POSTC_left T_POSTC_right T_PREC_left T_PREC_right T_PREF_left T_PREF_right T_PREM_left T_PREM_right UF_left UF_right)

for sub in $(ls -d ${src}/sub-*/); do
  base_sub=$(basename $sub /)
  echo $sub
  fodf=${sub}/FODF_Metrics/${base_sub}__fodf.nii.gz
  wm=${sub}/Segment_Tissues/${base_sub}__mask_wm.nii.gz
  out=${dest}/${base_sub}/${base_sub}_

  mkdir -p ${dest}/${base_sub}

  # python labelseg_predict.py ${fodf} ${wm} ${out} --checkpoint scilseg/hcp105/ds_ce_dice/scilseg/6db2f9bd813b4a28ac1426ec08b6c990/checkpoints/epoch=99-step=55900.ckpt --bundles ${bundles[@]} --img_size 128 --nb_labels 20
  python labelseg_predict.py ${fodf} ${wm} ${out} --checkpoint labelseg/hcp105/frompretrain/labelsel/7921a15e87b64f9d80ebbb11cce73e76/checkpoints/epoch=429-step=240370.ckpt --bundles ${bundles[@]} --img_size 128 --sh_order 6 --nb_labels 20
  # python labelseg_predict.py ${fodf} ${wm} ${out} --checkpoint labelsel/hcp105/frompretrain/labelsel/44a0342e1fd84ac48ee93a37f9ead76a/checkpoints/epoch=419-step=234780.ckpt --bundles ${bundles[@]} --img_size 128 --sh_order 6 --nb_labels 20

done
