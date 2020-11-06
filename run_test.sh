data=VOT2018
path=./experiments/siamcar_r50
model=checkpoint_e13.pth

python -u ./tools/test.py \
  --snapshot ./snapshot/${model} \
  --dataset ${data} \
  --config ${path}/config.yaml