
#
source /home/sf/anaconda3/bin/activate pysot
export PYTHONPATH=/home/sf/Documents/github_proj/carSiam

video=merkel.mp4

name=siamcar_r50
path=experiments/${name}

model=SiamCAR.pth


python ../tools/demo2.py \
	--config ../${path}/config.yaml \
	--snapshot ../${path}/${model}  \
	--video_name ${video}

