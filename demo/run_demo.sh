source /home/sf/anaconda3/bin/activate pysot

video=street.mp4
#video=lion.mp4
#video=Elephants.mp4
#video=dance.mp4
#video=cave.mp4
#video=bag.avi

# images
#video=../ods/dataset/Data/origin_data/street_low_right
#video=../ods/dataset/Data/origin_data/cave_low_left
#video=../ods/dataset/Data/origin_data/mall_low_right
#video=cube_street
#video=cube_low_right_cave

#cube_video=../ods/dataset/Data/train/street_low_right_train_00
#cube_video=../ods/dataset/Data/train/cave_low_left_train_01
cube_video=../ods/dataset/Data/train/mall_low_right_train_01

path=../experiments/siamcar_r50

#model=det_35_e19.pth
#model=vid_20_e19.pthy
#model=vid_40_e19.pth
model=general_model.pth

#python ../tools/demo_label.py \
#	--config ${path}/config.yaml \
#	--snapshot ../snapshot/${model}  \
#	--video ${video} \
#	--cube_video ${cube_video}

python ../tools/demo_viewerCube.py \
	--config ${path}/config.yaml \
	--snapshot ../snapshot/${model}  \
	--video ${video}