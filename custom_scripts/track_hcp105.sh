# TODO bundle list
# TODO subject list from 5th split

echo $(pwd)
src=/home/local/USHERBROOKE/thea1603/data/samtrack/datasets/hcp_105/testset
dest=/home/local/USHERBROOKE/thea1603/data/samtrack/results/105_71_pretrain_ds
mkdir -p ${dest}

bundles=(AF_left AF_right ATR_left ATR_right CA CC_1 CC_2 CC_3 CC_4 CC_5 CC_6 CC_7 CG_left CG_right CST_left CST_right FPT_left FPT_right FX_left FX_right ICP_left ICP_right IFO_left IFO_right ILF_left ILF_right MCP MLF_left MLF_right OR_left OR_right POPT_left POPT_right SCP_left SCP_right SLF_III_left SLF_III_right SLF_II_left SLF_II_right SLF_I_left SLF_I_right STR_left STR_right ST_FO_left ST_FO_right ST_OCC_left ST_OCC_right ST_PAR_left ST_PAR_right ST_POSTC_left ST_POSTC_right ST_PREC_left ST_PREC_right ST_PREF_left ST_PREF_right ST_PREM_left ST_PREM_right T_OCC_left T_OCC_right T_PAR_left T_PAR_right T_POSTC_left T_POSTC_right T_PREC_left T_PREC_right T_PREF_left T_PREF_right T_PREM_left T_PREM_right UF_left UF_right)

for sub in $(ls -d ${src}/*/); do
  base_sub=$(basename $sub /)
  echo $sub
  fodf=${sub}/${base_sub}__fodf.nii.gz
  wm=${sub}/${base_sub}__mask_wm.nii.gz
  out=${dest}/${base_sub}/${base_sub}_

  mkdir -p ${dest}/${base_sub}

  # python samu_predict.py ${fodf} ${wm} ${out} --checkpoint labelsel/hcp105/frompretrain/labelsel/7921a15e87b64f9d80ebbb11cce73e76/checkpoints/epoch=429-step=240370.ckpt --bundles ${bundles[@]} --img_size 128 --sh_order 6 --nb_labels 10
  python samu_predict.py ${fodf} ${wm} ${out} --checkpoint labelsel/hcp105/frompretrain/labelsel/44a0342e1fd84ac48ee93a37f9ead76a/checkpoints/epoch=419-step=234780.ckpt --bundles ${bundles[@]} --img_size 128 --sh_order 6 --nb_labels 10

done
