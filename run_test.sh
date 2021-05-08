source /home/sf/anaconda3/bin/activate pysot
export PYTHONPATH=/home/sf/Documents/github_proj/carSiam

#data=VOT2018
#data=OTB50
data=ODS

path=./experiments/siamcar_r50
#model=ods_det_vid_e19.pth
model=SiamCAR.pth

#path=./experiments/siamcar_mask_r50
#model=mask_refine_ods_e19.pth
#model=mask_refine_e20_bce.pth
#model=ods_det_vid_e19.pth
#model=mask_base_e20.pth

python -u ./tools/test.py \
  --snapshot ${path}/${model} \
  --dataset ${data} \
  --config ${path}/config.yaml \
#  --config ${path}/config_refine.yaml \
#  --vis