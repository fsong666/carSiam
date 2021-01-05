source /home/sf/anaconda3/bin/activate pysot
export PYTHONPATH=/home/sf/Documents/github_proj/carSiam

path="./experiments/siamcar_mask_r50"

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=2333 \
    ./tools/train.py --cfg ${path}/config_base.yaml
