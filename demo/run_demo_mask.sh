source /home/sf/anaconda3/bin/activate pysot
export PYTHONPATH=/home/sf/Documents/github_proj/carSiam

#video=street.mp4
video=merkel.mp4
#video=mall.mp4
#video=cave.mp4
#video=lion.mp4
#video=Elephants.mp4
#video=dance.mp4
#video=bag.avi

#video=basketball

path=../experiments/siamcar_r50
path_mask=../experiments/siamcar_mask_r50

#model=ods_det_vid_e19.pth
#model=mask_base_e19.pth

#model=ods_refine_e3.pth

#model=mask_refine_e20_bce.pth
#model=mask_refine_ods_e19.pth
model=checkpoint_e19.pth

#python ../tools/demo.py \
#	--config ${path}/config.yaml \
#	--snapshot ${path_mask}/${model}  \
#	--video_name ${video}

python ../tools/demo2.py \
	--config ${path}/config.yaml \
	--snapshot ${path_mask}/${model}  \
	--video_name ${video}

#python ../tools/demo_mask.py \
#	--config ${path_mask}/config_refine.yaml \
#	--snapshot ${path_mask}/${model}  \
#	--video_name ${video}