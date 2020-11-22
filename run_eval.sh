source /home/sf/anaconda3/bin/activate pysot

#data=VOT2018
#data=OTB50
data=ODS

python -u ./tools/eval.py \
  --tracker_path ./results \
  --dataset ${data} \
  --num 1 \
  --tracker_prefix 'vid_40_e19.pth0.15_0.1_0.4'


