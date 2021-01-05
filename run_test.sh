source /home/sf/anaconda3/bin/activate pysot
export PYTHONPATH=/home/sf/Documents/github_proj/carSiam

#data=VOT2018
#data=OTB50
data=ODS
path=./experiments/siamcar_r50
model=ods_det_vid_e19.pth
#model=general_model.pth

python -u ./tools/test.py \
  --snapshot ${path}/${model} \
  --dataset ${data} \
  --config ${path}/config.yaml \
  --vis