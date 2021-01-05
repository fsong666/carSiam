source /home/sf/anaconda3/bin/activate pysot
export PYTHONPATH=/home/sf/Documents/github_proj/carSiam

#video=street.mp4
video=mall.mp4
#video=cave.mp4
#video=lion.mp4
#video=Elephants.mp4
#video=dance.mp4
#video=bag.avi


path=../experiments/siamcar_r50
path_mask=../experiments/siamcar_mask_r50

#model=ods_det_vid_e19.pth
#model=mask_base_e20.pth
#model=mask_base_e19.pth
#model=checkpoint_e0_2500.pth # x_left
#model=checkpoint_e0_3000.pth  # x  right跟踪失败
#model=checkpoint_e0_3500.pth  # x_left
#model=checkpoint_e0_4000.pth  # x_right
#model=checkpoint_e0_4500.pth # x  right跟踪失败
#model=checkpoint_e0_5000.pth # x　　left跟踪失败
#model=checkpoint_e0_5500.pth  # x　　left跟踪失败
#model=checkpoint_e0_6000.pth  # x_right ín ods
#model=checkpoint_e0_6500.pth   # x_right
#model=checkpoint_e0_7000.pth  # x right跟踪失败
#model=checkpoint_e0_7500.pth  # x right跟踪失败
#model=checkpoint_e0_8000.pth  # x left跟踪失败
#model=checkpoint_e0_8500.pth  # x left跟踪失败
#model=checkpoint_e0_9000.pth  # left_x
#model=checkpoint_e1_12500.pth  # x_right

model=mask_refine_e20_bce.pth

#python ../tools/demo.py \
#	--config ${path}/config.yaml \
#	--snapshot ${path_mask}/${model}  \
#	--video_name ${video}

python ../tools/demo_mask.py \
	--config ${path_mask}/config_refine.yaml \
	--snapshot ${path_mask}/${model}  \
	--video_name ${video}