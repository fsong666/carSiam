source /home/sf/anaconda3/bin/activate pysot

data=VOT2018
#data=OTB50
#data=ODS

python -u ./tools/eval.py \
  --tracker_path ./results \
  --dataset ${data} \
  --num 1 \
  --tracker_prefix ''
#  --tracker_prefix 'mask_refine_e20_bce'
#  --tracker_prefix 'SiamMaskCAR'
#  --show_video_level \



