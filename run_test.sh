source /home/sf/anaconda3/bin/activate pysot

#data=VOT2018
#data=OTB50
data=ODS
path=./experiments/siamcar_r50
model=vid_40_e19.pth
#model=general_model.pth

python -u ./tools/test.py \
  --snapshot ./snapshot/${model} \
  --dataset ${data} \
  --config ${path}/config.yaml \
  --vis