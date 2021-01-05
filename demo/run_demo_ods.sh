source /home/sf/anaconda3/bin/activate pysot

#video=street.mp4
video=mall.mp4
#video=cave.mp4
#video=lion.mp4
#video=Elephants.mp4
#video=dance.mp4
#video=bag.avi

# images
#video=../ods/dataset/Data/origin_data/street_low_right
#video=../ods/dataset/Data/origin_data/cave_low_left
#video=../ods/dataset/Data/origin_data/mall_low_right
#video=cube_street
#video=cube_low_right_cave

#cube_video=../ods/dataset/Data/train/street_low_right_train_00
#cube_video=../ods/dataset/Data/train/cave_low_left_train_01
#cube_video=../ods/dataset/Data/train/mall_low_right_train_01

#depth
depth=../ods/dataset/depths

path=../experiments/siamcar_r50
path_mask=../experiments/siamcar_mask_r50

#model=det_35_e19.pth
#model=vid_40_e19.pth
#model=det_vid_e19.pth
model=ods_det_vid_e19.pth
#model=general_model.pth

#model=mask_refine_e20_bce.pth
#model=checkpoint_e0_9000.pth

#python ../tools/demo_label.py \
#	--config ${path}/config.yaml \
#	--snapshot ../snapshot/${model}  \
#	--video ${video} \
#	--cube_video ${cube_video}

## cube_map
#python ../tools/demo_ods.py \
#	--config ${path}/config.yaml \
#	--snapshot ${path}/${model}  \
#	--video_name ${video}

## ods to viewer cube
python ../tools/demo_viewerCube.py \
	--config ${path}/config.yaml \
	--snapshot ${path}/${model}  \
	--video_name ${video}  \
	--depth_img ${depth}

#python ../tools/demo_stereo.py \
#	--config ${path}/config.yaml \
#	--snapshot ${path_mask}/${model}  \
#	--video_name ${video} \
#	--depth_img ${depth}

## mask center cube
#python ../tools/demo_multiObj.py \
#	--config ${path_mask}/config_refine.yaml \
#	--snapshot ${path_mask}/${model}  \
#	--video_name ${video} \
#	--depth_img ${depth}
